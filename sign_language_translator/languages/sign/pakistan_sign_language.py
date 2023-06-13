from .sign_language import SignLanguage


class PakistanSignLanguage(SignLanguage):
    pass
    # if word in vocab.preprocessing_map['words_to_numbers']:
    #     word = vocab.preprocessing_map['words_to_numbers'][word]
    #     self.word_to_label(word)
    # label = [vocab.SUPPORTED_WORD_TO_LABEL[self.country_code.value+"-"+self.organization_code.value][self.text_language].get(word, label)]
    # if random.random() < 0.5:
    #     label = vocab.SUPPORTED_WORD_TO_LABEL_SEQUENCE[self.country_code.value+"-"+self.organization_code.value][self.text_language].get(word, label)
    # return {
    # **preprocessing_map["joint_word_to_split_words"][Urdu.name()],
    # **preprocessing_map["words_to_numbers"][Urdu.name()],
    # }

    # numbers, dates, finger spell (double/single)
