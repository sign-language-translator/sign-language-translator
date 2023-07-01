import re
from typing import Dict, Iterable, List


class SignTokenizer:
    def __init__(
        self,
        word_regex: str = r"\w+",
        compound_words: Iterable[str] = (),
        end_of_sentence_tokens: Iterable[str] = (".", "?", "!"),
        full_stops=(".",),
        non_sentence_end_words: Iterable[str] = ("A", "B", "C"),
    ):
        self.word_regex = word_regex

        self._first_subword_to_full = self._make_compound_word_map(compound_words)

        self.end_of_sentence_tokens = list(
            set(end_of_sentence_tokens) | set(full_stops)
        )
        self.non_sentence_end_words = non_sentence_end_words
        self.full_stops = full_stops

    def tokenize(self, text: str, join_compound_words: bool = True) -> List[str]:
        matches = re.finditer(self.word_regex, text)
        word_spans = [m.span() for m in matches]

        split_indexes = []

        prev_end = -1
        for span in word_spans:
            # break up non-words into single characters
            split_indexes.extend(range(prev_end + 1, span[0]))

            split_indexes.extend(span)
            prev_end = split_indexes[-1]

        split_indexes.extend(range(prev_end + 1, len(text) + 1))

        broken = [
            text[split_indexes[i] : split_indexes[i + 1]]
            for i in range(len(split_indexes) - 1)
            if split_indexes[i] != split_indexes[i + 1]
        ]

        if join_compound_words:
            broken = self._join_subwords(broken)

        # if join_word_sense:
        #     broken = self._join_word_sense(broken)

        return broken

    def sentence_tokenize(self, text: str) -> List[str]:
        tokens = self.tokenize(text)
        sentences = []
        sentence = []
        previous_token = None
        ended = False
        for token in tokens:
            if token in self.end_of_sentence_tokens:
                ended = True
                if (
                    token in self.full_stops
                    and previous_token in self.non_sentence_end_words # type: ignore
                ):
                    ended = False
            else:
                if ended:
                    sentences.append(self.detokenize(sentence))
                    sentence = []
                    ended = False
            sentence.append(token)
            previous_token = token

        if sentence:
            sentences.append(self.detokenize(sentence))

        return sentences

    def detokenize(self, tokens: Iterable[str]) -> str:
        return "".join(tokens)

    def _make_compound_word_map(
        self, word_list: Iterable[str]
    ) -> Dict[str, List[List[str]]]:
        mapper: Dict[str, List] = {}
        for word in set(word_list):
            subwords = self.tokenize(word, join_compound_words=False)
            if len(subwords) > 1:
                mapper.setdefault(subwords[0], [])
                mapper[subwords[0]].append(subwords)

        # sort descending by length so that longest sequence gets joint
        for first_subword in list(mapper.keys()):
            mapper[first_subword] = sorted(mapper[first_subword], key=len, reverse=True)

        return mapper

    def _join_subwords(self, tokens: List[str]) -> List[str]:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if tokens[i] in self._first_subword_to_full:
                compound_found = False
                for compound in self._first_subword_to_full[tokens[i]]:
                    if tokens[i : i + len(compound)] == compound:
                        new_tokens.append(self.detokenize(compound))
                        i += len(compound)
                        compound_found = True
                        break
                if compound_found:
                    continue

            new_tokens.append(tokens[i])
            i += 1

        return new_tokens

    def _join_word_sense(self, tokens: List[str]):
        # :TODO: handle ["word", "word", "(", "word", "-", "sense", ")"] --> ["word", "word(word-sense)"] in tokenization
        return tokens
