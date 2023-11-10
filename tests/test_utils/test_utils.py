from sign_language_translator.utils import (
    in_jupyter_notebook,
    sample_one_index,
    search_in_values_to_retrieve_key,
    tree,
)


def test_in_jupyter():
    in_jupyter_notebook()


def test_sample_index():
    data = [1, 2, 3, 4, 5, 0.0]
    for _ in range(20):
        assert 0 <= sample_one_index(data, temperature=2) < len(data) - 1


def test_search_in_values_to_retrieve_key():
    code_to_values = {"ur": {"12", "14"}, "en": {"20", "21"}}
    assert search_in_values_to_retrieve_key("20", code_to_values) == "en"
    assert search_in_values_to_retrieve_key("22", code_to_values) is None


def test_tree():
    tree("../..", directory_only=False, ignore=["__pycache__"], regex=False)
    tree("..", directory_only=True, ignore=[r"temp_.*"], regex=True)
