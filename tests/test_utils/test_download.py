from sign_language_translator.utils import download


def test_download():
    download("temp/example.html", "https://example.com", overwrite=True)
    try:
        download("temp/example.html", "https://example.com", overwrite=False)
    except FileExistsError:
        pass
    download("temp/example.html", "https://example.com", overwrite=True)
