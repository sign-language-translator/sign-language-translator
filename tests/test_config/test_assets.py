import os
import tempfile

import requests

import sign_language_translator as slt
from sign_language_translator.config.assets import Assets
from sign_language_translator.utils.parallel import threaded_map


def test_set_assets_dir():
    default = slt.Assets.ROOT_DIR

    temp_dir = tempfile.gettempdir()
    slt.Assets.set_root_dir(temp_dir)
    slt.Assets.download("text-preprocessing.json")
    download_path = os.path.abspath(os.path.join(temp_dir, "text-preprocessing.json"))
    assert os.path.exists(download_path)
    assert download_path == Assets.get_path("text-preprocessing.json")[0]

    try:
        with tempfile.NamedTemporaryFile() as f:
            slt.Assets.set_root_dir(f.name)
    except ValueError:
        pass

    os.makedirs(default, exist_ok=True)
    slt.Assets.set_root_dir(default)


def test_resource_urls_are_live():
    # load the full list of URLs
    Assets.load_all_urls()  # comment it out to save time

    # send requests
    def get_url_status(name_url_tuple, storage):
        response = requests.head(name_url_tuple[1], timeout=30, allow_redirects=True)
        storage[name_url_tuple] = response.status_code

    name_url_to_status = {}
    threaded_map(
        get_url_status,
        args_list=[
            ((name, url), name_url_to_status)
            for name, url in Assets.FILE_TO_URL.items()
        ],
        time_delay=1e-3,
        max_n_threads=256,
    )

    # check for broken URLs
    broken = {
        name: url for (name, url), status in name_url_to_status.items() if status != 200
    }
    num_broken = len(broken)
    assert num_broken == 0, f"{num_broken} Broken URLs: {broken}"


def test_get_ids():
    # exact match
    ids = Assets.get_ids("text-preprocessing.json")
    assert len(ids) == 1
    assert ids[0] == "text-preprocessing.json"

    # partial match
    ids = Assets.get_ids(r"videos/pk-.*-.*_\d+.mp4")
    assert len(ids) > 0

    # match nothing
    ids = Assets.get_ids(r"match-$^-nothing")
    assert len(ids) == 0


def test_delete_asset():
    Assets.delete("text-preprocessing.json")
    assert not os.path.exists(Assets.get_path("text-preprocessing.json")[0])


def test_infer_archive_name():
    # all info available
    name = Assets._infer_archive_name("videos/pk-hfad-1_1.mp4")
    assert name == r"datasets/pk-hfad-1_videos-?-mp4\.zip"

    # all dictionary videos
    # TODO: test for replications and sentences too
    name = Assets._infer_archive_name(r".*\.mp4")
    assert name == r"datasets/.*_videos-?-mp4\.zip"

    # all landmarks
    name = Assets._infer_archive_name(r"landmarks/.*\.csv")
    assert name == r"datasets/.*_landmarks-.*-csv\.zip"

    # all landmarks for numbers
    name = Assets._infer_archive_name(r"landmarks/.*_\d+(_.*)?\.csv")
    assert name == r"datasets/.*_landmarks-.*-csv\.zip"

    # no info available
    name = Assets._infer_archive_name(r".*\..*")
    assert name == r"datasets/.*_.^-.^-.*\.zip"


def test_fetch_assets():
    # download
    paths = Assets.download(r"(.*[-_])?urls.json", overwrite=True, progress_bar=True)
    assert len(paths) >= 1
    assert all([os.path.exists(path) for path in paths])

    # extract
    paths = Assets.extract(
        r"videos/xx-wordless-1_wordless.mp4", download_archive=True, overwrite=True
    )
    assert len(paths) == 1
    assert os.path.exists(paths[0])

    # fetch
    paths = Assets.fetch(r"videos/xx-wordless-1_wordless.mp4", overwrite=True)
    assert len(paths) == 1
    assert os.path.exists(paths[0])
