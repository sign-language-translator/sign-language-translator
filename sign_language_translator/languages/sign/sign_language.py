from abc import ABC, abstractmethod
from typing import Union
import datetime


class SignLanguage(ABC):
    @abstractmethod
    def _translate_number(self, number: Union[int, float]):
        pass

    @abstractmethod
    def _translate_date(self, date: datetime.date):
        pass

    @abstractmethod
    def _translate_time(self, time: datetime.time):
        pass

    @abstractmethod
    def _translate_word(self, text: str):
        pass

    @abstractmethod
    def _restructure_sentence(self, sentence: str) -> str:
        # adjust the grammar/word sequence
        # drop meaningless words and punctuation
        # normalize synonyms
        pass

    # text_language.supported_word : [[sign_collection.label]]
    # priority/source_id
    # probability(frequency)