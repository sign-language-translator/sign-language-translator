"""text scraping - search usable sentences, phrases or ngrams from corpora.
"""

import json
import re
import threading
import urllib.parse
from time import sleep
from typing import Dict, List, Set, Union

import requests
from bs4 import BeautifulSoup
from IPython.display import clear_output
from tqdm.auto import tqdm

captcha_waiting = False


def detect_captcha(page: str) -> bool:
    soup = BeautifulSoup(page, "html.parser")
    divs = soup.find_all("div", {"class": "g-recaptcha"})

    captcha_solving_needed = len(divs) > 0

    return captcha_solving_needed


def multi_threaded_requests(
    target_function, args_list, time_delay=0.02, max_n_threads=None
):
    threads = []
    for args in (bar := tqdm(args_list, desc="sending requests")):
        thread = threading.Thread(target=target_function, args=args)
        thread.start()
        threads.append(thread)
        sleep(time_delay)

        if max_n_threads:
            if len(threads) >= max_n_threads:
                for i, thread in enumerate(threads):
                    bar.set_description(f"waiting for threads {i/len(threads):.0%}")
                    thread.join()
                threads = []
                bar.set_description("sending requests")

    for thread in tqdm(threads, desc="waiting for threads"):
        thread.join()


def request_google(query: str, search_exact: bool = True) -> str:
    if search_exact:
        query = f'"{query}"'

    query = urllib.parse.quote(query)
    url = f"https://www.google.com/search?q={query}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    page = requests.get(url, headers=headers).text

    return page


def extract_google_results_count(page: str) -> int:
    n_results = 0

    soup = BeautifulSoup(page, "html.parser")
    divs = soup.find_all("div", {"id": "result-stats"})

    if divs:
        text = divs[0].next_element.text.replace(",", "")
        numbers = re.findall(r"\d+", text)
        if numbers:
            n_results = int(numbers[0])

    return n_results


def get_google_results_count(query: str, search_exact: bool = True) -> int:
    while True:
        response = request_google(query, search_exact=search_exact)

        if detect_captcha(response):
            print("!!!!!!!CAPTCHA!!!!!!!", query)
            sleep(2)
        else:
            break
    n_results = extract_google_results_count(response)

    return n_results


def request_glosbe(
    query: str, source_language: str, target_language: str, page_no: int
) -> str:

    url = f"https://glosbe.com/{source_language}/{target_language}/{urllib.parse.quote(query)}/fragment/tmem?page={page_no}&mode=MUST&stem=true"
    page = requests.get(url).text

    return page


def extract_glosbe_texts(page: str, element="span", css_class="dense") -> List[str]:
    soup = BeautifulSoup(page, "html.parser")
    cards = soup.find_all(element, {"class": css_class})

    texts = list({card.text for card in cards})

    return texts


def get_glosbe_texts(
    query: str,
    source_language: str,
    target_language: str,
    page_no: int,
    element="span",
    css_class="dense",
):
    global captcha_waiting
    while True:
        if captcha_waiting:
            sleep(10)

        response = request_glosbe(query, source_language, target_language, page_no)

        if detect_captcha(response):
            captcha_waiting = True
            print("!!!!!!!CAPTCHA!!!!!!!", query)
            sleep(10)
        else:
            clear_output()
            captcha_waiting = False
            break
    texts = extract_glosbe_texts(response, element=element, css_class=css_class)

    return texts


def scraping_function_for_glosbe(
    word: str,
    source_language: str,
    target_language: str,
    results: Dict[str, Union[Dict[str, int], Set[str]]],
):
    if results["next_page_number"][word] > 0:
        texts = get_glosbe_texts(
            word, source_language, target_language, results["next_page_number"][word]
        )
        with threading.Lock():
            if len(texts) > 0:
                results["next_page_number"][word] += 1
                results["texts"] |= set(texts)
            else:
                results["next_page_number"][word] = (
                    1 - results["next_page_number"][word]
                )


def request_rekhta(query, category, page_number=1):
    if category in ["nazm", "ghazal", "couplet"]:
        url = f"https://www.rekhta.org/search/{category}?q={urllib.parse.quote(query)}&lang=ur"
    else:
        url = f"https://www.rekhta.org/CollectionSearchLoading?lang=3&pageType=search%D8%B4%D8%B9%D8%B1content&keyword={urllib.parse.quote(query)}&pageIndex={page_number}&_=1676106622519"

    page = requests.get(url).text

    return page


def extract_rekhta_texts(page: str) -> Dict[str, Set[str]]:
    soup = BeautifulSoup(page, "html.parser")
    cards = soup.find_all("div", {"class": "genricMatchCard"})

    results = {}

    for card in cards:
        category = card.h5.text.strip()
        content = "".join(map(str, card.p.a.contents)).replace("<br/>", "\n")
        if category not in results:
            results[category] = set()
        results[category].add(content)

    return results


def merge_rekhta_results(recieving_dict, sending_dict):
    for k, v in sending_dict.items():
        if k not in recieving_dict:
            recieving_dict[k] = set()
        recieving_dict[k] |= v


def get_rekhta_texts(
    query: str,
    category: str,
    page_no: int = 1,
):
    response = request_rekhta(query, category, page_no)
    texts = extract_rekhta_texts(response)

    return texts


def scraping_function_for_rekhta(
    word: str,
    category: str,
    results: Dict[str, Union[Dict[str, int], Dict[str, Set[str]]]],
):
    if results["next_page_number"][word] > 0:
        texts = get_rekhta_texts(word, category, results["next_page_number"][word])
        with threading.Lock():
            if len(texts) > 0:
                results["next_page_number"][word] += 1
                merge_rekhta_results(results["texts"], texts)
            else:
                results["next_page_number"][word] = (
                    1 - results["next_page_number"][word]
                )


def request_wikipedia(word, limit=10000, lang="ur", namespaces=None):
    if namespaces is None:
        namespaces = "ns0=1&ns1=1&ns10=1&ns100=1&ns101=1&ns102=1&ns103=1&ns11=1&ns118=1&ns119=1&ns12=1&ns13=1&ns14=1&ns15=1&ns2=1&ns2300=1&ns2301=1&ns2302=1&ns2303=1&ns3=1&ns4=1&ns5=1&ns6=1&ns7=1&ns710=1&ns711=1&ns8=1&ns828=1&ns829=1&ns9=1"

    url = f"""https://{lang}.wikipedia.org/w/index.php?limit={limit}&fulltext=1&{namespaces}&search={urllib.parse.quote(f'"{word}"')}"""
    page = requests.get(url).text

    return page


def extract_wikipedia_texts(page: str) -> List[str]:
    soup = BeautifulSoup(page, "html.parser")
    cards = soup.find_all("div", {"class": "searchresult"})

    texts = [card.text.strip() for card in cards]

    return texts


def get_wikipedia_texts(word, limit=10000, lang="ur", namespaces=None) -> List[str]:
    page = request_wikipedia(word, limit=limit, lang=lang, namespaces=namespaces)
    texts = extract_wikipedia_texts(page)

    return texts


def scraping_function_for_wikipedia(
    word: str,
    limit: int,
    lang: str,
    results: Dict[str, Union[Dict[str, int], Set[str]]],
) -> None:
    texts = get_wikipedia_texts(word, limit=limit, lang=lang)
    with threading.Lock():
        if word not in results["n_results"]:
            results["n_results"][word] = 0
        results["n_results"][word] += len(texts)
        results["texts"] |= set(texts)
