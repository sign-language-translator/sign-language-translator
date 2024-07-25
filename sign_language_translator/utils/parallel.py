"""parallel.py - Parallel Processing Utilities

This module provides utilities for performing parallel processing tasks using multithreading in Python.

Functions:
    - threaded_map: Multi-threaded mapping of a function to an iterable. Useful for I/O bound tasks.
"""

import threading
from time import sleep
from typing import Callable, Iterable, Optional

from tqdm.auto import tqdm


def threaded_map(
    target: Callable,
    args_list: Iterable,
    time_delay=0.02,
    timeout: Optional[float] = None,
    max_n_threads: Optional[int] = None,
    progress_bar=True,
    leave=True,
):
    """
    Multi-threaded mapping of a function to an iterable. Useful for I/O bound tasks.

    This function allows you to apply a target function to elements in an iterable concurrently using multiple threads.
    You can control the number of threads, introduce time delays between thread launches, and enable a progress bar.

    Parameters:
        target (Callable): The function to apply to each element in the iterable.
        args_list (Iterable): An iterable of arguments to be passed to the target function in parallel.
        time_delay (float, optional): Time delay (in seconds) between launching threads. Default is 0.02.
        timeout (float, optional): The maximum amount of time (in seconds) to wait for a thread to finish. Default is None, which means wait indefinitely.
        max_n_threads (int, optional): The maximum number of threads to run concurrently. Default is None, which means entire args_list will be processed concurrently.
        progress_bar (bool, optional): Enable or disable the progress bar. Default is True.
        leave (bool, optional): Whether to leave the progress bar after completion. Default is True.

    Example:

        .. code-block:: python

            import requests
            from sign_language_translator.utils import threaded_map

            def get_webpage(url, results: dict):
                if url not in results:
                    results[url] = requests.get(url)

            urls = ["https://example.com", "https://github.com", ...]
            results = {}
            args = [(url, results) for url in urls]

            # process urls concurrently, with a maximum of 2 threads at a time
            threaded_map(get_webpage, args, max_n_threads=2)
    """

    if progress_bar:
        args_list = tqdm(args_list, desc="Launching threads", leave=leave)

    threads = []
    for n, args in enumerate(args_list):
        # launch thread
        thread = threading.Thread(target=target, args=args, name=f"{n}_{args}"[:50])
        thread.start()
        threads.append(thread)
        sleep(time_delay)

        # wait for some threads to finish
        if max_n_threads and len(threads) >= max_n_threads:
            for i, thread in enumerate(threads):
                if isinstance(args_list, tqdm):
                    args_list.set_description(f"Awaiting threads {i/len(threads):.0%}")
                thread.join(timeout=timeout)
                # TODO: progress_callback
            # reset threads list
            threads = []
            if isinstance(args_list, tqdm):
                args_list.set_description("Launching threads")

    # wait for remaining threads to finish
    if progress_bar:
        threads = tqdm(threads, desc="Awaiting threads", leave=leave)
    for thread in threads:
        thread.join(timeout=timeout)
        # TODO: progress_callback
