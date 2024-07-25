"""
Utility Functions for Text Processing

This module contains utility functions for text processing tasks.

Functions:
    make_ngrams: Creates n-grams from a given sequence.
    extract_supported_subsequences_indexes: Extracts the indexes of subsequences based on provided tags and skipped items.
    extract_supported_subsequences: Extracts subsequences from a given sequence based on provided tags and skipped items.
    concatenate_sentence_terminals: Concatenates start and end of sentence tokens to a list of sentences.

Classes:
    ListRegex: A utility class for finding sub-lists within a list of strings that match specified patterns.
"""

import re
from typing import Any, Iterable, List, Optional, Set, Tuple, Union


def make_ngrams(sequence: Iterable, n: int) -> List[Iterable]:
    """Create all possible slices of the given iterable of size n.
    for example, sequence="1234" and n=2 would create ["12","23","34"].

    Args:
        sequence (Iterable): The iterable sequence from which the n-grams will be created.
        n (int): The size of the n-grams.

    Returns:
        List[Iterable]: A list of Iterables representing the n-grams created from the sequence.
            The type of list items is same as sequence argument.
    """

    start = 0
    end = len(sequence) - n  # type: ignore

    ngrams = (
        [sequence[i : i + n] for i in range(start, end + 1)] if end >= start else []  # type: ignore
    )
    return ngrams


def extract_supported_subsequences_indexes(
    sequence: Iterable[Any],
    tags: Iterable[Any],
    supported_tags: Set[Any],
    skipped_items: Set[Any],
) -> List[List[int]]:
    """Extract indexes of supported subsequences from a sequence based on tags and skipped items.

    Args:
        sequence (Iterable[Any]): The input sequence.
        tags (Iterable[Any]): Tags corresponding to each item in the sequence.
        supported_tags (Set[Any]): Set of tags indicating support for a subsequence.
        skipped_items (Set[Any]): Set of items to be skipped.

    Returns:
        List[List[int]]: A list indices of supported subsequences, where each inner list represents a subsequence.

    Examples:

    .. code-block:: python

        sequence = [1, 2, 3, 4, 5, 6]
        tags = ['A', 'A', 'B', 'A', 'A', 'C']
        supported_tags = {'A'}
        skipped_items = {2}
        extract_supported_subsequences(sequence, tags, supported_tags, skipped_items)
        # [[0], [3, 4]]
    """

    all_subsequences = []
    subsequence = []
    for i, (token, tag) in enumerate(zip(sequence, tags)):
        if (tag in supported_tags) and (token not in skipped_items):
            subsequence.append(i)
        else:
            if subsequence:
                all_subsequences.append(subsequence)
                subsequence = []

    if subsequence:
        all_subsequences.append(subsequence)

    return all_subsequences


def extract_supported_subsequences(
    sequence: Iterable[Any],
    tags: Iterable[Any],
    supported_tags: Set[Any],
    skipped_items: Set[Any],
) -> List[List[Any]]:
    """Extract supported subsequences from a sequence based on tags and skipped items.

    Args:
        sequence (Iterable[Any]): The input sequence.
        tags (Iterable[Any]): Tags corresponding to each item in the sequence.
        supported_tags (Set[Any]): Set of tags indicating support for a subsequence.
        skipped_items (Set[Any]): Set of items to be skipped.

    Returns:
        List[List[Any]]: A list of supported subsequences, where each inner list represents a subsequence.

    Examples:

    .. code-block:: python

        sequence = [1, 2, 3, 4, 5, 6]
        tags = ['A', 'A', 'B', 'A', 'A', 'C']
        supported_tags = {'A'}
        skipped_items = {2}
        extract_supported_subsequences(sequence, tags, supported_tags, skipped_items)
        # [[1], [4, 5]]
    """

    indexes = extract_supported_subsequences_indexes(
        sequence=sequence,
        tags=tags,
        supported_tags=supported_tags,
        skipped_items=skipped_items,
    )

    subsequences = [[sequence[i] for i in index] for index in indexes]  # type: ignore

    return subsequences


def concatenate_sentence_terminals(sentences: List, start_token, end_token):
    """
    Inserts start and end tokens between the sentences the input list and
    concatenates them to the sentences (useful when the input is coming from a sentence tokenizer.)

    This function takes a list of sentences and adds a start token to the beginning
    of each sentence except the first and an end token to the end of each sentence except the last.

    Parameters:
        sentences (List): A list of sentences to be processed.
            Sentences can be strings or list of tokens or any type but it must support + operator for concatenation.
        start_token: The token to be added at the start of sentences. Must be same type as a sentence.
        end_token: The token to be added at the end of sentences. Must be same type as a sentence.

    Returns:
        List: A new list of sentences with start and end tokens inserted.

    Example:

    .. code-block:: python

        sentences = ["Hello!", "How are you?", "Goodbye."]
        start_token = "<start>"
        end_token = "<end>"
        result = concatenate_sentence_terminals(sentences, start_token, end_token)
        # Output: ["Hello!<end>", "<start>How are you?<end>", "<start>Goodbye."]
    """

    new_sentences = []
    for i, sentence in enumerate(sentences):
        if i > 0:
            sentence = start_token + sentence
        if i < len(sentences) - 1:
            sentence = sentence + end_token
        new_sentences.append(sentence)

    return new_sentences


class ListRegex:
    """A utility class for finding sub-lists within a list of strings that match specified patterns.

    ListRegex provides methods for
    matching patterns against items in a list,
    searching for the first occurrence of patterns,
    finding all occurrences of patterns,
    and retrieving the starting and ending indices of matches.

    Patterns can be defined using:
        1. regular expressions (str)
        2. lists of patterns (regex (str) or a nested list of patterns)
        3. tuple containing the pattern and its interval quantifier ("\\w+", (2,None)).
    When using regular expressions, each pattern is matched against an individual item in the list.
    When using a list of patterns, any of the patterns in the list can match an item.
    When using a tuple of pattern and counts, items in the specified range can match the pattern.

    Examples:

    .. code-block:: python

        items = ["apple", "banana", "orange", "orange", "grape", "melon", "orange", "kiwi"]

        # Match the patterns against the items
        patterns = ["apple", "\\w+"]
        result = ListRegex.match(items, patterns)
        # Output: (0, 2)

        # Search for the first occurrence of the patterns
        patterns = [r"ba(na){2}", ("orange", (0,3))]
        result = ListRegex.search(items, patterns)
        # Output: (1, 4)

        # Find all occurrences of the patterns
        patterns = ["orange", ["grape", "kiwi"]]
        result = ListRegex.find_all(items, patterns)
        # Output: [['orange', 'grape'], ['orange', 'kiwi']]
    """

    @staticmethod
    def match(
        items: List[str], patterns: List[Union[str, List, Tuple]]
    ) -> Optional[Tuple[int, int]]:
        """
        Matches the given patterns against the items in the list.
        Applies the patterns at the start of the list of string.

        Args:
            items (List[str]): The sequence of strings to be matched.
            patterns (List[str|List]): The patterns to be matched against the items.

        Returns:
            Tuple[int, int] or None: A tuple containing the starting and ending indices of the matched items,
            or None if no match is found.
        """

        does_match = False
        pattern_index = 0
        item_index = 0
        count, counting_state = None, False

        while item_index < len(items) and pattern_index < len(patterns):
            pattern = patterns[pattern_index]
            item = items[item_index]

            # interval quantifiers: *, +, {n,m}
            # TODO: recursion on tuple
            if isinstance(pattern, tuple):
                assert (
                    len(pattern) == 2 and len(pattern[-1]) == 2
                ), "use proper syntax for repetition: (pattern, (min_count, max_count))"
                pattern, (min_count, max_count) = pattern[0], pattern[1]
                min_count = min_count if min_count is not None else float("-inf")
                max_count = max_count if max_count is not None else float("inf")
                if not counting_state:
                    count, counting_state = 0, True

            # Regex: abc, Character set: [abc],
            # TODO: Logical OR (ab|bc)  (sequence of items OR some other sequence of items)
            does_match = ListRegex._match_item(item, pattern)
            # TODO: shouldn't match when max_count is 0
            if does_match:
                item_index += 1
                if counting_state:
                    count += 1  # type: ignore
                if (count is not None) and count >= max_count:  # type: ignore
                    count, counting_state = None, False

                if not counting_state:
                    pattern_index += 1
            else:
                if counting_state:
                    pattern_index += 1
                    does_match = min_count <= count <= max_count  # type: ignore

                    count, counting_state = None, False

            if not does_match:
                break

        return (0, item_index) if does_match else None

    @staticmethod
    def _match_item(item: str, pattern):
        """
        Matches a single item against a pattern.
        A pattern could be a string regex or a list containing regex or list of regex and so on.

        Args:
            item (str): The item to be matched.
            pattern (str|list): The pattern to be matched against the item.

        Returns:
            bool: True if the item matches the pattern, False otherwise.
        """

        does_match = False
        if isinstance(pattern, str):
            does_match = bool(re.match(pattern, item))
        elif isinstance(pattern, list):
            does_match = any(ListRegex._match_item(item, pat) for pat in pattern)
        else:
            raise ValueError("unknown value of pattern, provide str or list")

        return does_match

    @staticmethod
    def search(items: List[str], patterns) -> Optional[Tuple[int, int]]:
        """
        Searches for the first occurrence of the patterns in the list of items.

        Args:
            items (List[str]): The list of strings to be searched.
            patterns (List[str]): The patterns to be searched for in the items.

        Returns:
            Tuple[int, int] or None: A tuple containing the starting and ending indices of the matched items,
            or None if no match is found.
        """

        for i in range(len(items) - len(patterns) + 1):
            matched = ListRegex.match(items[i:], patterns)
            if matched:
                return (i, i + list(matched)[1])  # type: ignore
        return None

    @staticmethod
    def find_all(items: List[str], patterns: List) -> List[List[str]]:
        """Finds all occurrences of the patterns in the list of items.

        Args:
            items (List[str]): The list of strings to be searched.
            patterns (List[str]): The patterns to be searched for in the items.

        Returns:
            List[List[str]]: A list of matched subsequences of items.
        """

        matches = [
            items[start:end] for start, end in ListRegex.find_all_spans(items, patterns)
        ]
        return matches

    @staticmethod
    def find_all_spans(items: List[str], patterns: List) -> List[Tuple[int, int]]:
        """Finds the starting and ending indices of all occurrences of the patterns in the list of items.

        Args:
            items (List[str]): The list of strings to be searched.
            patterns (List[str]): The patterns to be searched for in the items.

        Returns:
            List[Tuple[int,int]]: A list of tuples containing the starting and ending indices of the matched items.
        """

        spans = []
        end = 0
        while end < len(items):
            span = ListRegex.search(items[end:], patterns)
            if span:
                span = end + span[0], end + span[1]
                _, end = span
                spans.append(span)
            else:
                break

        return spans


__all__ = [
    "make_ngrams",
    "extract_supported_subsequences_indexes",
    "extract_supported_subsequences",
    "concatenate_sentence_terminals",
    "ListRegex",
]
