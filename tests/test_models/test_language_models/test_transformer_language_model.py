import torch

from sign_language_translator.models.language_models.transformer_language_model.model import (
    TransformerLanguageModel,
)


def get_tlm_model():
    return TransformerLanguageModel(
        {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3,
            "<sos>": 4,
            "<eos>": 5,
            "<pad>": 6,
            "<unk>": 7,
        },
        vocab_size=8,
        start_of_sequence_token="<sos>",
        unknown_token="<unk>",
        padding_token="<pad>",
        device="cpu",
    )


def test_transformer_language_model_tokens_to_ids():
    model = get_tlm_model()

    assert model.tokens_to_ids(["a", "b", "c", "d"]) == [0, 1, 2, 3]


def test_transformer_language_model_ids_to_tokens():
    model = get_tlm_model()

    assert model.ids_to_tokens(torch.Tensor([0, 1, 2, 3]).type(torch.long)) == [
        "a",
        "b",
        "c",
        "d",
    ]


def test_transformer_language_model_position_ids():
    model = get_tlm_model()

    model.training = False
    model.randomly_shift_position_embedding_during_training = False

    token_ids = torch.Tensor([[3, 1, 0, 2]]).type(torch.long)
    pos_ids = model._make_position_ids(token_ids)
    assert pos_ids.shape == (4,)
    assert pos_ids[..., 0] == 0

    token_ids = torch.Tensor([2, 3, 1, 2]).type(torch.long)
    pos_ids = model._make_position_ids(token_ids)
    assert pos_ids.shape == (4,)
    assert pos_ids[..., 0] == 0
    assert pos_ids[..., 3] == 3

    model.training = True
    model.randomly_shift_position_embedding_during_training = True

    token_ids = torch.Tensor([[3, 1, 0, 2]]).type(torch.long)
    pos_ids = model._make_position_ids(token_ids)
    assert (pos_ids - pos_ids[..., 0])[..., 3] == 3

    token_ids = torch.Tensor([4, 1, 3, 2]).type(torch.long)
    pos_ids = model._make_position_ids(token_ids)
    assert pos_ids[..., 0] == 0


def test_transformer_language_model_predict_next():
    model = get_tlm_model()
    model.next(["<sos>"])
