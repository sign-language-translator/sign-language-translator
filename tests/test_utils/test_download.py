import pytest

from sign_language_translator.utils import download


def test_download():
    download("temp/example.html", "https://example.com", overwrite=True)
    with pytest.raises(FileExistsError):
        download("temp/example.html", "https://example.com", overwrite=False)

    download("temp/example.html", "https://example.com", overwrite=True)
