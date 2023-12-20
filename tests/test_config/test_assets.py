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
    for _, urls_file in Assets.asset_regex_to_urls_file:
        Assets.load_urls(urls_file)

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
