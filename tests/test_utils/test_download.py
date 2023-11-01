from sign_language_translator.utils import download


def test_download():
    download("temp/temp.html", "https://example.com", overwrite=True)
    try:
        download("temp/temp.html", "https://example.com", overwrite=False)
    except FileExistsError:
        pass
    download("temp/temp.html", "https://example.com", overwrite=True)
    download("temp.html", "https://example.com", overwrite=True)
