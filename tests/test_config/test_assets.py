import os
import tempfile

import pytest
import requests

import sign_language_translator as slt
from sign_language_translator.config.assets import Assets
from sign_language_translator.utils.parallel import threaded_map


def test_set_assets_dir():
    default = slt.Assets.ROOT_DIR
    temp_dir = tempfile.gettempdir()
    try:
        slt.Assets.set_root_dir(temp_dir)
        slt.Assets.download("text-preprocessing.json")
    except Exception as exc:
        slt.Assets.set_root_dir(default)
        raise exc

    download_path = os.path.abspath(os.path.join(temp_dir, "text-preprocessing.json"))
    assert os.path.exists(download_path)
    assert download_path == Assets.get_path("text-preprocessing.json")[0]

    with pytest.raises(ValueError):
        with tempfile.NamedTemporaryFile() as f:
            slt.Assets.set_root_dir(f.name)

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
    name = Assets.infer_archive_name("videos/pk-hfad-1_1.mp4")
    assert name == r"datasets/pk-hfad-1\.videos-?-mp4\.zip$"

    # all dictionary videos
    # TODO: test for replications and sentences too
    name = Assets.infer_archive_name(r".*\.mp4")
    assert name == r"datasets/.*\.videos-?-mp4\.zip$"

    # all landmarks
    name = Assets.infer_archive_name(r"landmarks/.*\.csv")
    assert name == r"datasets/.*\.landmarks-.*-csv\.zip$"

    # all landmarks for numbers
    name = Assets.infer_archive_name(r"landmarks/.*_\d+(_.*)?\.csv")
    assert name == r"datasets/.*\.landmarks-.*-csv\.zip$"

    # no info available
    name = Assets.infer_archive_name(r".*\..*")
    # ToDo: should match all archives or None?
    assert name == r"datasets/.*\..^-.^-.*\.zip$"

    # wrong folder / extension
    name = Assets.infer_archive_name(r"landmarks/pk-hfad-1_airplane.mp4")
    assert name == r"datasets/pk-hfad-1\..^-.*-mp4\.zip$"

    # preprocessing models
    name = Assets.infer_archive_name("pk-hfad-1_airplane.landmarks-mediapipe-world.csv")
    assert name == r"datasets/pk-hfad-1\.landmarks-mediapipe-world-csv\.zip$"

    name = Assets.infer_archive_name(
        "landmarks/pk-hfad-1_chicken-meat.landmarks-mediapipe-image-pose-0.npy"
    )
    assert name == r"datasets/pk-hfad-1\.landmarks-mediapipe-image-pose-0-npy\.zip$"

    name = Assets.infer_archive_name(r"landmarks/.*_.*\.landmarks-mediapipe-.*\.csv")
    assert name == r"datasets/.*\.landmarks-mediapipe-.*-csv\.zip$"


def test_download_assets():
    paths = Assets.download(r"(.*[-_])?urls.json", overwrite=True, progress_bar=True)
    assert len(paths) >= 1
    assert all(os.path.exists(path) for path in paths)


def test_extract_assets():
    # extract video
    paths = Assets.extract(
        "videos/xx-wordless-1_wordless.mp4", download_archive=True, overwrite=True
    )
    assert len(paths) == 1
    assert os.path.exists(paths[0])

    # extract landmarks
    paths = Assets.extract(
        r"xx-shapes-1_.*\.landmarks-testmodel\.csv",
        download_archive=True,
        overwrite=True,
    )
    assert len(paths) == 3
    assert all(os.path.exists(path) for path in paths)
