from sign_language_translator.utils import threaded_map


def test_threaded_map():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    operation = lambda x: x * 2

    def wrapper(data_item, results_dict):
        results_dict[data_item] = operation(data_item)

    results = {}

    threaded_map(wrapper, [(x, results) for x in data], max_n_threads=3)
    assert results == {n: operation(n) for n in data}
