import json
import os
import warnings
from collections import Counter
from glob import glob
from typing import Dict

from tqdm.auto import tqdm

from sign_language_translator.utils.tree import tree
from sign_language_translator import Settings


class DataCompletenessChecker:
    def __init__(
        self,
        signs_recordings_dir,
        signs_collection,
        sign_collection_parent_folder="videos",
        cameras=["front", "below", "left", "right"],
        persons=[
            f"person{n}"
            for n in [101, 151, 201, 202, 203, 204, 205, 251, 252, 253, 254, 255]
        ],
        sep=Settings.FILENAME_SEPARATOR,
        extension=".mp4",
        find_filepaths=False,
        load_filepaths=False,
    ) -> None:
        self.signs_recordings_dir = signs_recordings_dir
        self.sign_collection_parent_folder = sign_collection_parent_folder

        with open(os.path.join(signs_recordings_dir, "recordings_labels.json")) as f:
            self.RECORDINGS_LABELS = json.loads(f.read())

        if isinstance(signs_collection, str):
            self.signs_collection = signs_collection
            assert (
                self.signs_collection in self.RECORDINGS_LABELS
            ), "unsupported signs collections provided"
        else:
            NotImplementedError("provide str signs_collection")

        self.cameras = cameras
        self.persons = persons
        self.sep = sep
        self.extension = extension
        self.create_all_possible_filenames()

        self.paths_filename = (
            f"{self.sign_collection_parent_folder}_{self.signs_collection}_paths.txt"
        )

        self.filenames = set()
        self.file_paths = []

        if find_filepaths:
            self.find_filepaths()
        if load_filepaths:
            self.load_filepaths(append=True, load_all=False)

    def find_filepaths(self, append=False):
        patterns = [
            os.path.join(
                    self.signs_recordings_dir,
                    self.sign_collection_parent_folder,
                    self.signs_collection,
                    person,
                    f"*{camera}{self.extension}",
                )
            for person in self.persons
            for camera in self.cameras
        ]

        file_paths = [
            p
            for pattern in tqdm(patterns)
            for p in glob(pattern)
        ]
        file_paths = (
            list(set(file_paths) | set(self.file_paths))
            if append and hasattr(self, "file_paths")
            else file_paths
        )

        self.filenames = self.extract_filenames(file_paths, validate_repetition=True)
        self.file_paths = file_paths

    def load_filepaths(self, append=False, load_all=False):
        file_paths = []
        if os.path.exists(self.paths_filename):
            with open(self.paths_filename) as f:
                all_file_paths = f.read().split("\n")

            if load_all:
                file_paths = all_file_paths
            else:
                for p in all_file_paths:
                    dirname, fname = os.path.split(p)
                    fname = self.split_filename(fname)
                    if (
                        (fname[-1] in self.cameras)
                        and (fname[-2] in self.persons)
                        # and (fname[-2] == os.path.split(dirname)[-1].lower())
                    ):
                        file_paths.append(p)

        file_paths = (
            list(set(file_paths) | set(self.file_paths))
            if append and hasattr(self, "file_paths")
            else file_paths
        )

        self.filenames = self.extract_filenames(file_paths, validate_repetition=True)
        self.file_paths = file_paths

    def extract_filenames(self, file_paths, validate_repetition):
        filenames = [os.path.split(p)[-1] for p in file_paths]
        if validate_repetition:
            self.report_repetition(
                filenames, message="during extract_filenames(file_paths) - "
            )
        return set(filenames)

    def make_file_path(self, filename):
        filename_parts = self.split_filename(filename)
        return os.path.join(
            self.signs_recordings_dir,
            self.sign_collection_parent_folder,
            self.signs_collection,
            filename_parts[-2],  # person
            filename,
        )

    def report_repetition(self, data, message=""):
        counts = Counter(data)
        repeated = {k: v for k, v in counts.items() if v > 1}
        if len(repeated) > 0:
            raise Exception(
                f"{message}Repetition found for {len(repeated)} files:\n{repeated}"
            )

    def save_file_paths(self, overwrite=True):
        if not overwrite:
            self.load_filepaths(append=True, load_all=True)

        with open(self.paths_filename, "w") as f:
            f.write("\n".join(sorted(self.file_paths)))

    def create_all_possible_filenames(self):
        all_filenames = [
            f"{label}{self.sep}{person}{self.sep}{camera}{self.extension}"
            for label in self.RECORDINGS_LABELS[self.signs_collection]
            for person in self.persons
            for camera in self.cameras
        ]
        self.report_repetition(
            all_filenames, message="During create_all_possible_filenames() - "
        )

        self.all_filenames = set(all_filenames)

    def find_missing_files(self):
        return sorted(set(self.all_filenames) - set(self.filenames))

    def find_extra_files(self):
        return sorted(set(self.filenames) - set(self.all_filenames))

    def count_files(self, by: str, filenames=None):
        if filenames is None:
            filenames = self.filenames
        keys, idx = (
            (self.cameras, -1)
            if str(by).lower() in ["cam", "camera", "cameras"]
            else (self.persons, -2)
            if str(by).lower().startswith("person")
            else (self.RECORDINGS_LABELS[self.signs_collection], -3)
            if str(by).lower() in ["word", "words", "label", "labels"]
            else (None, None)
        )
        if keys is None:
            return dict(Counter(filenames))
        counts = {k: 0 for k in keys}
        counts.update(dict(Counter([self.split_filename(f)[idx] for f in filenames])))
        return counts

    def split_filename(self, fname):
        return fname[: -len(self.extension)].split(self.sep)

    def rename(
        self,
        label_replacements: Dict[str, str] = {},
        person_replacements: Dict[str, str] = {},
        camera_replacements: Dict[str, str] = {},
        path_filter=lambda path: True,
        n_files_donot_exist_examples=5,
        verbose=True,
        raise_or_log_FileExistsError="raise",
    ):
        n_renamed = 0
        files_donot_exist = []

        for i, fpath in enumerate(tqdm(self.file_paths, leave=False)):
            if not os.path.exists(fpath):
                files_donot_exist.append(fpath)
                continue
            if not path_filter(fpath):
                continue

            directory, fname = os.path.split(fpath)

            fname_parts = self.split_filename(fname)
            fname_parts[0] = label_replacements.get(fname_parts[0], fname_parts[0])
            fname_parts[1] = person_replacements.get(fname_parts[1], fname_parts[1])
            fname_parts[2] = camera_replacements.get(fname_parts[2], fname_parts[2])

            new_fname = self.sep.join(fname_parts) + self.extension
            new_fpath = os.path.join(directory, new_fname)

            if new_fname == fname:
                continue

            if os.path.exists(new_fpath):  # bloody os.rename on mac (T_T)
                if os.path.samefile(fpath, new_fpath):
                    warnings.warn(f'the filename strings are different ({fname= }, {new_fname= }) but they will effectively point to the same file, rename would likely skip this..')
                    input_txt = ''
                    while input_txt not in ['yes', 'no','y', 'n']:
                        input_txt = input('continue? (y/n)')
                    if input_txt.lower() in ['no', 'n']:
                        break
                else:
                    FileExistsError_message = f"{new_fpath} named file already exists"
                    if raise_or_log_FileExistsError == "log":
                        print(FileExistsError_message)
                        continue
                    else:
                        raise FileExistsError(FileExistsError_message)

            if new_fname in self.filenames:
                FileExistsError_message = (
                    f"{new_fname} named file already exists somewhere in the collection"
                )
                if raise_or_log_FileExistsError == "log":
                    print(FileExistsError_message)
                    continue
                else:
                    raise FileExistsError(FileExistsError_message)

            else:
                os.rename(fpath, new_fpath)
                self.file_paths[i] = new_fpath
                self.filenames.remove(fname)
                self.filenames.add(new_fname)
                n_renamed += 1

        if verbose:
            print(
                f"{n_renamed} files renamed out of total {len(self.file_paths)} files scanned."
            )

        if len(files_donot_exist) > 0:
            examples = "\n\t" + "\n\t".join(
                files_donot_exist[:n_files_donot_exist_examples]
            )
            warnings.warn(
                f"{len(files_donot_exist)} files do not exist. Here are a few examples:{examples}"
            )

    def make_tree(
        self,
        n_levels_above_signs_collection: int = 0,
        directory_only=True,
        extra_line=True,
        ignore=["__pycache__", "temp"],
    ):
        root_dir = os.path.join(
            self.signs_recordings_dir,
            self.sign_collection_parent_folder,
            self.signs_collection,
        )
        for _ in range(n_levels_above_signs_collection):
            root_dir = os.path.split(root_dir)[0]

        tree(
            root_dir,
            directory_only=directory_only,
            extra_line=extra_line,
            ignore=ignore,
        )
