import multiprocessing
import os
import threading
import time
from typing import List, Tuple

from deep_translator import GoogleTranslator
from deep_translator.exceptions import NotValidPayload
from tqdm.auto import tqdm
from urllib3.exceptions import MaxRetryError, SSLError


def translate_text(text: str, gt: GoogleTranslator, results):

    try:
        translation = gt.translate(text)
    except NotValidPayload:
        translation = ""
    except (MaxRetryError, SSLError) as e:
        print(f"Error - ''{text}''")
        print(e)
        exit()

    results[text] = translation


def translate_file(data: Tuple[str, str, float]):
    """Translate a newline seperated list of words in a file into a target language using Google Translator. The source language is automatically detected.
    The output will be in a new file named "{target_language}_from_{filename}.txt"

    Args:
        data (Tuple[str, str, float]): input_file_path, target_language_code, time_delay: The amount of time in seconds to wait before sending next request in a new thread.
    """
    texts_dir, lang, time_delay = data

    with open(texts_dir, "r") as f:
        texts = f.read().splitlines()

    gt = GoogleTranslator(source="auto", target=lang)
    results = {}

    threads = []
    for txt in texts:
        thread = threading.Thread(target=translate_text, args=(txt, gt, results))
        thread.setName(f"{lang}_{txt}")
        thread.start()
        threads.append(thread)
        time.sleep(time_delay)

    for thread in threads:
        thread.join()

    translations = [results.get(txt, "") for txt in texts]

    parent, fname = os.path.split(texts_dir)
    target_dir = os.path.join(parent, f"{lang}_from_{fname}")
    with open(target_dir, "w") as f:
        f.write("\n".join(map(str, translations)))


def make_translations(args_list: List[Tuple[str, str, float]], n_processes: int = 10):
    with multiprocessing.Pool(processes=n_processes) as pool:
        for _ in tqdm(
            pool.imap_unordered(translate_file, args_list),
            total=len(args_list),
        ):
            pass
