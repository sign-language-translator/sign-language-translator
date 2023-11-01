from sign_language_translator.utils import (
    in_jupyter_notebook,
    sample_one_index,
    search_in_values_to_retrieve_key,
    tree
)


def test_in_jupyter():
    in_jupyter_notebook()


def test_sample_index():
    sample_one_index([1, 2, 3, 4, 5], 2)


def test_search_in_values_to_retrieve_key():
    assert (
        search_in_values_to_retrieve_key("20", {"ur": {"12", "14"}, "en": {"20", "21"}})
        == "en"
    )
    assert (
        search_in_values_to_retrieve_key("20", {"ur": {"12", "14"}, "en": {"21", "22"}})
        is None
    )

def test_tree():
    tree("../..", directory_only=False)
