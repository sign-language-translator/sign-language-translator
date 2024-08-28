"""Custom Layers for Decoder-only Transformers

This module contains custom layers used in the Transformer Decoder architecture.

Classes:
    FeedForward(torch.nn.Module): Implements a simple feedforward neural network module
        with one hidden layer.
    CausalMultiHeadSelfAttention(torch.nn.Module): Implements the causal multi-head self-attention
        mechanism used in transformer decoders.
    DecoderBlock(torch.nn.Module): Implements a single transformer decoder block with
        multi-head self-attention and feedforward neural network layers but no cross-attention.

Example:

.. code-block:: python

    import torch
    from sign_language_translator.models.language_models.transformer_language_model.layers import FeedForward, DecoderBlock, CausalMultiHeadSelfAttention

    model = FeedForward(n_embed=256, hidden_size=512, dropout=0.2, activation='relu')
    input_tensor = torch.randn(32, 256)
    output_tensor = model(input_tensor)

    decoder_block = DecoderBlock(n_embed=256, hidden_size=512, n_heads=8, max_seq_len=32, dropout=0.2, activation='relu')
    input_tensor = torch.randn(16, 32, 256)
    output_tensor = decoder_block(input_tensor)

    attention_layer = CausalMultiHeadSelfAttention(n_heads=8, embed_size=256, dropout=0.2)
    input_tensor = torch.randn(16, 32, 256)
    output_tensor = attention_layer(input_tensor)
"""

import torch


class CausalMultiHeadSelfAttention(torch.nn.Module):
    """
    Causal Multi-Head Self-Attention Module.

    This class implements the causal multi-head self-attention mechanism. It takes an input tensor of shape
    (batch_size, seq_len, embed_size) and applies causal attention, where each token can attend only to
    itself and the previous tokens in the sequence. The input tensor is transformed into queries, keys,
    and values, and then passed through the scaled dot-product attention mechanism. The final output tensor
    is obtained by concatenating the heads and applying a linear projection with dropout.

    Parameters:
        n_heads (int): The number of attention heads.
        embed_size (int): The size of the input embedding dimension. Must be divisible by n_heads.
        dropout (float, optional): The dropout probability applied in the attention and projection layers. Default is 0.25.
        max_seq_len (int, optional): The maximum input sequence length (used only in custom dot-product attention (pytorch<2.0.0)). Default is 64.
        attention_bias (bool, optional): If True, enables trainable bias parameter in the query, key & value layer. Default is False.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_size).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_size).

    Example:

    .. code-block:: python

        model = CausalMultiHeadSelfAttention(n_heads=8, embed_size=256, dropout=0.2)
        input_tensor = torch.randn(16, 32, 256)
        output_tensor = model(input_tensor)
    """

    def __init__(
        self,
        n_heads,
        embed_size,
        dropout=0.25,
        max_seq_len: int = 64,
        attention_bias=False,
    ):
        assert embed_size % n_heads == 0, "embed_dim must be divisible by num_heads"
        super().__init__()

        # save hyper parameters
        self.n_heads = n_heads
        self.embed_size = embed_size
        self.dropout_probability = dropout

        # causal attention mechanism
        self.use_flash_attention = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )
        if not self.use_flash_attention:
            # TODO: this layer should be agnostic of max_seq_len (auto grow)
            self.register_buffer(
                "fill_mask",
                torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                    -1, -1, max_seq_len, max_seq_len
                )
                == 0,
            )
        self.dot_product_scalar = (embed_size // n_heads) ** -0.5

        # learnable parameters
        self.all_queries_keys_values = torch.nn.Linear(
            embed_size, embed_size * 3, bias=attention_bias
        )
        self.attention_dropout = torch.nn.Dropout(dropout)
        self.projection = torch.nn.Linear(embed_size, embed_size)
        self.final_dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the Causal Multi-Head Self-Attention.

        This method applies the forward pass of the causal multi-head self-attention to the input tensor x.
        The input tensor is transformed into queries, keys, and values, which are then passed through the
        scaled dot-product attention mechanism. The final output tensor is obtained by concatenating the
        attention heads and applying a linear projection with dropout.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_size).

        Example:

        .. code-block:: python

            model = CausalMultiHeadSelfAttention(n_heads=8, embed_size=256, dropout=0.2)
            input_tensor = torch.randn(16, 32, 256)
            output_tensor = model.forward(input_tensor)
        """

        # ([batch,] seq_len, embed_size)
        seq_len = x.shape[-2]
        input_had_batch_dim = x.dim() == 2

        # get Q, K, V matrices ([batch,] seq_len, embed_size)
        x = self.all_queries_keys_values(x)  # ([batch,] seq_len, 3 * embed_size)
        querys, keys, values = x.split(self.embed_size, dim=-1)

        # break up embedding dimension ([batch,] seq_len,     n_heads, head_size)
        querys = querys.view(-1, seq_len, self.n_heads, self.embed_size // self.n_heads)
        keys = keys.view(-1, seq_len, self.n_heads, self.embed_size // self.n_heads)
        values = values.view(-1, seq_len, self.n_heads, self.embed_size // self.n_heads)

        # move n_heads dim to batch dim by swapping seq_len & n_heads using transpose
        querys = querys.transpose(-3, -2)  # (batch, n_head, seq_len, head_size)
        keys = keys.transpose(-3, -2)  #   (batch, n_head, seq_len, head_size)
        values = values.transpose(-3, -2)  # (batch, n_head, seq_len, head_size)

        if self.use_flash_attention:
            # GPU go Brrrrrr
            x = torch.nn.functional.scaled_dot_product_attention(
                querys,
                keys,
                values,
                attn_mask=None,
                dropout_p=self.dropout_probability if self.training else 0,
                is_causal=True,
            )  # (batch, n_head, seq_len, head_size)
        else:
            # Manual attention implementation
            # Q @ K = (B, nH, T, HS) @ (B, nH, HS, T) -> (B, nH, T, T)
            attentions = querys @ keys.transpose(-2, -1)
            # normalize the variance to 1
            attentions = attentions * self.dot_product_scalar
            # don't look at future tokens
            attentions = attentions.masked_fill(
                self.fill_mask[..., :seq_len, :seq_len], float("-inf")  # type: ignore
            )
            attentions = torch.nn.functional.softmax(attentions, dim=-1)
            attentions = self.attention_dropout(attentions)

            x = attentions @ values  # (B, nH, T, T) @ (B, nH, T, HS) -> (B, nH, T, HS)

        # concatenate heads
        x = x.transpose(-3, -2).contiguous()  #     (batch, seq_len, n_head, head_size)
        x = x.view(-1, seq_len, self.embed_size)  # (batch, seq_len, embed_size)

        x = self.projection(x)
        x = self.final_dropout(x)

        if input_had_batch_dim and x.shape[0] == 1:
            x = x.squeeze(0)

        return x


class FeedForward(torch.nn.Module):
    """
    FeedForward Neural Network Module.

    This class implements a simple feedforward neural network module with one hidden layer.
    It takes an input tensor of shape (batch_size, n_embed) and applies a linear transformation,
    followed by an activation function (GELU or ReLU), and then another linear transformation
    with dropout applied. The final output tensor has the same shape as the input tensor.

    Parameters:
        n_embed (int): The size of the input feature dimension.
        hidden_size (int): The number of neurons in the hidden layer.
        dropout (float, optional): The dropout probability applied after the second linear layer.
            Default is 0.25.
        activation (str, optional): The activation function to be used. Supported values are 'gelu'
            for GELU activation and 'relu' for ReLU activation. Default is 'gelu'.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, n_embed).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, n_embed).

    Example:

    .. code-block:: python

        model = FeedForward(n_embed=256, hidden_size=512, dropout=0.2, activation='relu')
        input_tensor = torch.randn(32, 256)
        output_tensor = model(input_tensor)
    """

    def __init__(self, n_embed, hidden_size, dropout=0.25, activation="gelu"):
        super().__init__()
        self.fully_connected_1 = torch.nn.Linear(n_embed, hidden_size)
        self.activation = torch.nn.GELU() if activation == "gelu" else torch.nn.ReLU()
        self.fully_connected_2 = torch.nn.Linear(hidden_size, n_embed)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the FeedForward neural network.

        This method applies the forward pass of the feedforward neural network to the input tensor x.
        The forward pass involves passing the input tensor through the hidden layer, followed by an
        activation function, and then through the output layer with dropout applied.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, n_embed).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_embed).

        Example:

        .. code-block:: python

            model = FeedForward(n_embed=256, hidden_size=512, dropout=0.2, activation='relu')
            input_tensor = torch.randn(32, 256)
            output_tensor = model.forward(input_tensor)
        """

        x = self.fully_connected_1(x)
        x = self.activation(x)
        x = self.fully_connected_2(x)
        x = self.dropout(x)
        return x


class DecoderBlock(torch.nn.Module):
    """
    Transformer Decoder Block Module.

    This class implements a single transformer decoder block, consisting of causal multi-head self-attention
    and feedforward neural network layers but no cross-attention. The input tensor x goes through the
    layer norm & attention mechanism and also forms a skip connection over them into another
    layer norm & feedforward neural network. The output also contains a residual connection over these two operations.

    Parameters:
        n_embed (int): The size of the input feature dimension and also the output feature dimension.
        hidden_size (int): The number of neurons in the feedforward neural network's hidden layer.
        n_heads (int): The number of attention heads for multi-head self-attention.
        max_seq_len (int): The maximum sequence length of the input tensor.
        dropout (float, optional): The dropout probability applied in both attention and feedforward
            layers. Default is 0.25.
        activation (str, optional): The activation function to be used in the feedforward neural network.
            Supported values are 'gelu' for GELU activation and 'relu' for ReLU activation. Default is 'gelu'.
        device (torch.device, optional): If provided, the attention and feedforward layers will be moved
            to this device. Default is None.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embed).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embed).

    Example:

    .. code-block:: python
        model = DecoderBlock(n_embed=256, hidden_size=512, n_heads=8, max_seq_len=32, dropout=0.2, activation='relu')
        input_tensor = torch.randn(16, 32, 256)
        output_tensor = model(input_tensor)
    """

    def __init__(
        self,
        n_embed,
        hidden_size,
        n_heads,
        max_seq_len,
        dropout=0.25,
        activation="gelu",
        attention_bias=False,
    ):
        super().__init__()
        self.attention = CausalMultiHeadSelfAttention(
            n_heads,
            n_embed,
            dropout=dropout,
            max_seq_len=max_seq_len,
            attention_bias=attention_bias,
        )
        self.feed_forward = FeedForward(
            n_embed, hidden_size, dropout=dropout, activation=activation
        )
        self.mha_layer_norm = torch.nn.LayerNorm(n_embed)
        self.ff_layer_norm = torch.nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.attention(self.mha_layer_norm(x))
        x = x + self.feed_forward(self.ff_layer_norm(x))
        return x


__all__ = [
    "DecoderBlock",
    "FeedForward",
    "CausalMultiHeadSelfAttention",
]
