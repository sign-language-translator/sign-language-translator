import enum
import re
from typing import Any, Callable, Iterable, List, Tuple

from sign_language_translator.utils import PrintableEnumMeta

__all__ = [
    "Rule",
    "Tagger",
    "Tags",
]


class Tags(enum.Enum, metaclass=PrintableEnumMeta):
    """Enumeration of token tags used in NLP processing."""

    WORD = "WORD"
    DATE = "DATE"
    TIME = "TIME"
    NAME = "NAME"
    SPACE = "SPACE"
    NUMBER = "NUMBER"
    ACRONYM = "ACRONYM"
    PUNCTUATION = "PUNCTUATION"
    SUPPORTED_WORD = "SUPPORTED_WORD"
    DEFAULT = ""
    END_OF_SEQUENCE = "EOS"
    START_OF_SEQUENCE = "SOS"
    AMBIGUOUS = "AMBIGUOUS"
    WORDLESS = "WORDLESS"
    # WORD_SENSE = "(anything)"
    # PERSON = {
    #     'id': 'PERSON',
    #     'name': 'Person',
    #     'description': 'Represents a person',
    #     'example': 'John Doe',
    # }

    def __repr__(self) -> str:
        return str(self)

    # TODO: rename SUPPORTED to something better e.g. directly_mapped, recognized


class Rule:
    """A rule for token classification based on a matching function.

    Args:
        matcher (Callable[[str], bool]): A function that takes a token (str) as input
            and returns a boolean indicating whether the token matches the rule.
        tag (Any): The tag associated with tokens that match the rule.
        priority (int): The priority level of the rule.

    Methods:
        is_match(token: str) -> bool:
            Checks if the given token matches the rule.

        get_tag() -> str:
            Retrieves the tag associated with the rule.

        get_priority() -> int:
            Retrieves the priority level of the rule.

        from_pattern(pattern: str, tag: str, priority: int) -> Rule:
            Creates a rule from a regular expression pattern, tag, and priority.
            The created rule will use the pattern to match tokens.

    Note:
        - Rules with higher priority are applied first when classifying tokens.
        - The matcher function should return True if the token matches the rule, and False otherwise.
    """

    def __init__(self, matcher: Callable[[str], bool], tag: Any, priority: int):
        self.matcher = matcher
        self.tag = tag
        self.priority = priority

    def is_match(self, token: str):
        return self.matcher(token)

    def get_tag(self):
        return self.tag

    def get_priority(self):
        return self.priority

    @staticmethod
    def from_pattern(pattern: str, tag: Any, priority: int):
        def match(text: str):
            return bool(re.match(pattern, text))

        return Rule(match, tag, priority)


class Tagger:
    """A tagger that applies a set of rules to classify tokens.

    Args:
        rules (List[Rule]): A list of Rule objects representing the classification rules.
            Smaller priority value rules overwrite the others.
        default (Tags, optional): The default tag to assign when no rule matches a token.
            Defaults to Tags.DEFAULT.

    Methods:
        tag(tokens: List[str]) -> List[Tuple[str, Any]]:
            Assigns tags to a list of tokens based on the defined rules.
            Returns a list of tuples containing the token and its corresponding tag.

        get_tags(tokens: List[str]) -> List[Any]:
            Retrieves the tags for a list of tokens based on the defined rules.
            Returns a list of tags corresponding to the input tokens.

    Note:
        - The rules are applied in the order they appear in the list
            but higher priority (smaller value) rules overpower.
        - The default tag is assigned to tokens that do not match any rule.
    """

    def __init__(self, rules: List[Rule], default=Tags.DEFAULT):
        self.rules = rules
        self.default = default

    def tag(self, tokens: Iterable[str]) -> List[Tuple[str, Any]]:
        tagged_tokens = list(zip(tokens, self.get_tags(tokens)))

        return tagged_tokens

    def get_tags(self, tokens: Iterable[str]) -> List[Any]:
        return [self._apply_rules(token) for token in tokens]

    def _apply_rules(self, token: str) -> Any:
        tag = self.default
        priority = float("inf")

        for rule in self.rules:
            if rule.is_match(token):
                if rule.get_priority() < priority:
                    tag = rule.get_tag()
                    priority = rule.get_priority()

        return tag
