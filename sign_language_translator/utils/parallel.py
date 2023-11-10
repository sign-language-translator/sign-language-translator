"""parallel.py - Parallel Processing Utilities

This module provides utilities for performing parallel processing tasks using multithreading in Python.

Functions:
    - threaded_map: Multi-threaded mapping of a function to an iterable. Useful for I/O bound tasks.
"""

import threading
from time import sleep
from typing import Callable, Iterable

from tqdm.auto import tqdm


def threaded_map(
    target_function: Callable,
    args_list: Iterable,
    time_delay=0.02,
    max_n_threads=None,
    progress_bar=True,
):
    """
    Multi-threaded mapping of a function to an iterable. Useful for I/O bound tasks.

    This function allows you to apply a target function to elements in an iterable concurrently using multiple threads.
    You can control the number of threads, introduce time delays between thread launches, and enable a progress bar.

    Parameters:
        target_function (Callable): The function to apply to each element in the iterable.
        args_list (Iterable): The input data iterable to be processed in parallel.
        time_delay (float, optional): Time delay (in seconds) between launching threads. Default is 0.02.
        max_n_threads (int, optional): The maximum number of threads to run concurrently.
            Default is None, which means entire args_list will be processed concurrently.
        progress_bar (bool, optional): Enable or disable the progress bar. Default is True.

    Note:
        The progress bar provided by the tqdm library gives you visual feedback on the progress of thread launching and
        completion.

    Example:
        .. code-block:: python
            def get_webpage(url, results: dict):
                if not url in results:
                    results[url] = requests.get(url)
            urls = ["https://example.com", "https://example.com", ]
            results = {}
            args = [(url, results) for url in urls]
            threaded_map(get_webpage, args, max_n_threads=2)
            # Threads are launched concurrently, with a maximum of 2 threads at a time.
    """

    threads = []
    tqdm_bar = tqdm(args_list, desc="Launching threads") if progress_bar else None
    for n, args in enumerate(tqdm_bar or args_list):
        thread = threading.Thread(target=target_function, args=args, name=f"{n}_{args}")
        thread.start()
        threads.append(thread)
        sleep(time_delay)

        if max_n_threads:
            if len(threads) >= max_n_threads:
                for i, thread in enumerate(threads):
                    if tqdm_bar:
                        tqdm_bar.set_description(
                            f"Awaiting threads {i/len(threads):.0%}"
                        )
                    thread.join()
                threads = []
                if tqdm_bar:
                    tqdm_bar.set_description("Launching threads")

    for thread in tqdm(threads, desc="Awaiting threads"):
        thread.join()
