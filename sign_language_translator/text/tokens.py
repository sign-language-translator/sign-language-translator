import re
from typing import Any, Dict, Iterable, List, Set, Tuple

# from ..languages.vocab import (
#     END_OF_SENTENCE_MARKS,
#     PUNCTUATION,
#     PUNCTUATION_REGEX,
#     URDU_WORD_REGEX,
#     NON_SENTENCE_END_TOKENS,
#     FULL_STOPS,
# )
from ..config.settings import Settings


class SignTokenizer:
    def __init__(
        self,
        vocab: Set[str] = set(),
        vocab_path: str = None,
        drop_spaces=True,
        start_of_sentence_token = "[",
        end_of_sentence_token = "]",
    ) -> None:
        self.vocab = vocab
        self.punctuation = PUNCTUATION
        self.end_of_sentence_marks = END_OF_SENTENCE_MARKS
        self.punctuation_regex = PUNCTUATION_REGEX
        self.word_regex = URDU_WORD_REGEX
        self.drop_spaces = drop_spaces
        self.sep = " " if self.drop_spaces else ""
        self.eos = end_of_sentence_token
        self.sos = start_of_sentence_token

        self._id_to_token_dict = {i: w for i, w in enumerate(self.vocab)}
        self._token_to_id_dict = {w: i for i, w in self._id_to_token_dict.items()}

        self.first_subword_to_complete = self._make_compound_word_map(vocab)

    def _make_compound_word_map(
        self, vocab: Iterable[str]
    ) -> Dict[str, List[List[str]]]:
        mapper = {}
        for item in vocab:
            subwords = self.split(item)
            if len(subwords) > 1:
                if subwords[0] not in mapper:
                    mapper[subwords[0]] = []
                mapper[subwords[0]].append(subwords)

        # sort descending by length so that longest sequence gets joint
        for first_subword in list(mapper.keys()):
            mapper[first_subword] = sorted(mapper[first_subword], key=len, reverse=True)

        return mapper

    def remove_space_before_punctuation(self, text: str):
        def get_replacement(match: re.Match) -> str:
            matched_string = match.group(0)
            replacement_string = matched_string.strip()

            return replacement_string

        fixed_text = re.sub(self.punctuation_regex, get_replacement, text)

        return fixed_text

    def get_split_indexes(self, text: str) -> List[str]:
        word_spans = self.word_spans(text)
        split_idxs = []

        prev_end = -1
        for span in word_spans:
            # break up non-words into single characters
            split_idxs.extend(range(prev_end + 1, span[0]))

            # break up numbers/acronyms into single characters
            if text[span[0] : span[1]].isdigit() or self.is_acronym(
                text[span[0] : span[1]]
            ):
                span = range(span[0], span[1] + 1)

            split_idxs.extend(span)
            prev_end = split_idxs[-1]

        split_idxs.extend(range(prev_end + 1, len(text) + 1))

        return split_idxs

    def split(self, text: str):
        split_idxs = self.get_split_indexes(text)
        broken = [
            text[split_idxs[i] : split_idxs[i + 1]] for i in range(len(split_idxs) - 1)
        ]
        return broken

    def extract_supported_substrings(
        self, text: str, extra_allowed_symbols: Set[str] = {" "}, break_sentences=True
    ) -> List[str]:
        tokens = self.tokenize(text)
        mask = self.get_supported_mask(
            tokens, extra_allowed_symbols=extra_allowed_symbols
        )
        substrings = self._extract_sublists_according_to_mask(tokens, mask)
        substrings = (
            [
                s
                for sub in substrings
                for s in self.break_sentences(self.insert_sentence_markers(sub))
            ]
            if break_sentences
            else substrings
        )
        substrings = [self.detokenize(toks) for toks in substrings]

        return substrings

    def extract_unsupported_substrings(
        self, text: str, extra_allowed_symbols: Set[str] = {}, break_sentences=True
    ) -> List[str]:
        tokens = self.tokenize(text)
        mask = self.get_supported_mask(
            tokens, extra_allowed_symbols=extra_allowed_symbols
        )
        mask = [not val for val in mask]
        substrings = self._extract_sublists_according_to_mask(tokens, mask)
        substrings = (
            [
                s
                for sub in substrings
                for s in self.break_sentences(self.insert_sentence_markers(sub))
            ]
            if break_sentences
            else substrings
        )
        substrings = [self.detokenize(toks) for toks in substrings]

        return substrings

    def _extract_sublists_according_to_mask(self, tokens: List[Any], mask: List[bool]):
        sublists = []
        a_list = []
        for i in range(len(tokens)):
            if mask[i]:
                a_list.append(tokens[i])
            else:
                if a_list:
                    sublists.append(a_list)
                    a_list = []

        if a_list:
            sublists.append(a_list)

        return sublists

    def get_supported_mask(
        self, tokens: List[str], extra_allowed_symbols: Set[str] = {" "}
    ) -> List[bool]:
        mask = [(tok in self.vocab) or (tok in extra_allowed_symbols) for tok in tokens]
        return mask

    def _is_word(self, token: str) -> bool:
        return bool(re.match(self.word_regex, token))

    def insert_sentence_markers(self, tokens: List[str]) -> List[str]:
        new_tokens = []
        ended = False
        eos_appended = False
        for i, tok in enumerate(tokens):
            if tok in self.end_of_sentence_marks:
                if not ended and (
                    (i == 0)
                    or not (
                        tok in FULL_STOPS
                        and i > 0
                        and tokens[i - 1] in NON_SENTENCE_END_TOKENS
                    )
                ):
                    ended = True
                new_tokens.append(tok)
            else:
                if ended and not eos_appended:
                    new_tokens.append(self.eos)
                    eos_appended = True
                if self._is_word(tok) and ended:
                    new_tokens.append(self.sos)
                    ended = False
                    eos_appended = False
                new_tokens.append(tok)

        if ended and not eos_appended:
            new_tokens.append(self.eos)
        if self._is_word(tok) and ended:
            new_tokens.append(self.sos)

        return new_tokens

    def break_sentences(self, tokens: List[str]):
        sentences = []
        sen10c = []

        for tok in tokens:
            sen10c.append(tok)
            if tok == self.eos:
                sentences.append(sen10c)
                sen10c = []
        if sen10c:
            sentences.append(sen10c)

        return sentences

    def is_acronym(self, text: str):
        return text.isupper() and len(text) <= 6

    def join_subwords(self, tokens: List[str]):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if tokens[i] in self.first_subword_to_complete:
                compound_found = False
                for compound in self.first_subword_to_complete[tokens[i]]:
                    if tokens[i : i + len(compound)] == compound:
                        new_tokens.append("".join(compound))
                        i += len(compound)
                        compound_found = True
                        break
                if compound_found:
                    continue

            new_tokens.append(tokens[i])
            i += 1

        return new_tokens

    def word_spans(self, text: str) -> List[Tuple[int, int]]:
        matches = re.finditer(self.word_regex, text)
        spans = [m.span() for m in matches]

        return spans

    def tokenize(self, text: str) -> List[str]:
        broken = self.split(text)
        tokens = self.join_subwords(broken)

        if self.drop_spaces:
            tokens = [tok for tok in tokens if not re.match(r"\s+", tok)]

        return tokens

    def remove_space_between_numbers(self, text: str) -> str:
        fixed_text = re.sub(
            r"(\d )+\d",
            lambda match: match.group().replace(" ", ""),
            text,
        )
        return fixed_text

    def remove_space_between_acronyms(self, text: str) -> str:
        fixed_text = re.sub(
            r"([A-Z] )+[A-Z]",
            lambda match: match.group().replace(" ", ""),
            text,
        )
        return fixed_text

    def detokenize(self, tokens: List[str]):
        if self.drop_spaces:
            text = " ".join(tokens)
            text = self.remove_space_before_punctuation(text)
            text = self.remove_space_between_numbers(text)
            text = self.remove_space_between_acronyms(text)
            text = re.sub(
                "((" + re.escape(self.sos) + ") )|( (" + re.escape(self.eos) + "))",
                lambda x: x.group().strip(),
                text,
            )
        else:
            text = "".join(tokens)

        return text

    def token_to_id(self, token):
        if isinstance(token, str):
            tokens = self._token_to_id_dict.get(token, -1)
        elif isinstance(token, Iterable):
            tokens = [self._token_to_id_dict.get(tok, -1) for tok in token]
        return tokens

    def id_to_token(self, id_):
        if isinstance(id_, int):
            tokens = self._id_to_token_dict.get(id_, "")
        elif isinstance(id_, Iterable):
            tokens = [self._id_to_token_dict.get(i, "") for i in id_]
        return tokens
