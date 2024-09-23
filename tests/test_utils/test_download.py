import pytest

from sign_language_translator.utils import download, is_internet_available


@pytest.mark.skipif(not is_internet_available(), reason="No internet available")
def test_download():
    download("temp/example.html", "https://example.com", overwrite=True)
    with pytest.raises(FileExistsError):
        download("temp/example.html", "https://example.com", overwrite=False)

    download("temp/example.html", "https://example.com", overwrite=True)
