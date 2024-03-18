import os

import torch

from sign_language_translator.models.text_embedding import VectorLookupModel


def test_vector_lookup_model():
    tokens = ["hello", "world"]
    vectors = torch.tensor([[1, 2, 3], [4, 5, 6]]).type(torch.float32)

    model = VectorLookupModel(tokens, vectors)

    assert (model["hello"] == torch.tensor([1, 2, 3])).all()
    assert (model.embed("hello world") == torch.tensor([2.5, 3.5, 4.5])).all()

    # update vocabulary & vectors
    model.update(["world", "sign"], torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]))

    assert (model["world"] == torch.tensor([7, 8, 9])).all()
    assert (model["sign"] == torch.tensor([10, 11, 12])).all()
    assert (model.embed("hello world sign") == torch.tensor([6, 7, 8])).all()

    # Test similarity
    similars, scores = model.similar(torch.tensor([1.0, 2.0, 3.0]), k=1)
    assert similars == ["hello"]
    assert abs(scores[0] - 1) < 1e-4

    # Test saving and loading
    model.save("temp/model.pt")
    assert os.path.exists("temp/model.pt")

    loaded_model = VectorLookupModel.load("temp/model.pt")
    assert (loaded_model["hello"] == torch.tensor([1, 2, 3])).all()
    assert (loaded_model["world"] == torch.tensor([7, 8, 9])).all()
    assert (loaded_model["world sign"] == torch.tensor([8.5, 9.5, 10.5])).all()

    # Test alignment
    model.alignment_matrix = torch.eye(3).flip(0)
    assert (model.embed("hello world", align=True) == torch.tensor([6, 5, 4])).all()

    # Compressed storage
    model.save("temp/model.zip")
    assert os.path.exists("temp/model.zip")

    loaded_model = VectorLookupModel.load("temp/model.zip")
    assert (loaded_model["hello"] == torch.tensor([1, 2, 3])).all()
