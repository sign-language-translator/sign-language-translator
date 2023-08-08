import requests

from sign_language_translator import Settings


def test_resource_urls_are_live():
    for resource_name, url in Settings.FILE_TO_URLS.items():
        assert requests.head(url, timeout=30, allow_redirects=True).status_code == 200, f"{resource_name} URL is broken"
