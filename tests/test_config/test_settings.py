import os
import tempfile

import sign_language_translator as slt


def test_set_resource_dir():
    default = slt.Settings.RESOURCES_ROOT_DIRECTORY

    temp_dir = tempfile.gettempdir()
    slt.set_resources_dir(temp_dir)
    slt.utils.download_resource("text_preprocessing.json")
    assert os.path.exists(os.path.join(temp_dir, "text_preprocessing.json"))

    try:
        slt.set_resources_dir("/hbvopiorfg83i24rge8r/fujidh23498r")
    except ValueError:
        pass

    os.makedirs(default, exist_ok=True)
    slt.set_resources_dir(default)
