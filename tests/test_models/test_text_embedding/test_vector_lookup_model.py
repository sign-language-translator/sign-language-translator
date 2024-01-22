import os

import torch

from sign_language_translator.models.text_embedding import VectorLookupModel


def test_vector_lookup_model():
    tokens = ["hello", "world"]
    vectors = torch.tensor([[1, 2, 3], [4, 5, 6]]).type(torch.float32)

    model = VectorLookupModel(tokens, vectors)

    assert (model["hello"] == torch.tensor([1, 2, 3])).all()
    assert (model.embed("hello world") == torch.tensor([2.5, 3.5, 4.5])).all()

    model.update(["world", "sign"], torch.tensor([[7., 8., 9.], [10., 11., 12.]]))

    assert (model["world"] == torch.tensor([7, 8, 9])).all()
    assert (model["sign"] == torch.tensor([10, 11, 12])).all()
    assert (model.embed("hello world sign") == torch.tensor([6, 7, 8])).all()

    model.save("temp/model.pt")
    assert os.path.exists("temp/model.pt")

    loaded_model = VectorLookupModel.load("temp/model.pt")
    assert (loaded_model["hello"] == torch.tensor([1, 2, 3])).all()
    assert (loaded_model["world"] == torch.tensor([7, 8, 9])).all()
    assert (loaded_model["world sign"] == torch.tensor([8.5, 9.5, 10.5])).all()
