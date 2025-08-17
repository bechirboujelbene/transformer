from torch import nn
import torch

def positional_encoding(d_model: int,
                        max_length: int) -> torch.Tensor:
    """
    Computes the positional encoding matrix
    Args:
        d_model: Dimension of Embedding
        max_length: Maximums sequence length

    Shape:
        - output: (max_length, d_model)
    """
    exponent = torch.arange(0, d_model, 2) / d_model
    pos = torch.arange(0, max_length).unsqueeze(1) 
    angle_freq = torch.exp(exponent * -(torch.log(torch.Tensor([10000]))))
    pos_encoding = torch.zeros(max_length, d_model)
    pos_encoding[:, 0::2] = torch.sin(pos * angle_freq)
    pos_encoding[:, 1::2] = torch.cos(pos * angle_freq)

    output = pos_encoding

    return output

class Embedding(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 max_length: int):
        """
        Args:
            vocab_size: Number of elements in the vocabulary
            d_model: Dimension of Embedding
            max_length: Maximum sequence length
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(d_model,max_length)

        # We will convert it into a torch parameter module for you! You can treat it like a normal tensor though!
        if self.pos_encoding is not None:
            self.pos_encoding = nn.Parameter(data=self.pos_encoding, requires_grad=False)

    def forward(self,
                inputs: torch.Tensor) -> torch.Tensor:
        """
        The forward function takes in tensors of token ids and transforms them into vector embeddings. 
        It then adds the positional encoding to the embeddings, and if configured, performs dropout on the layer!

        Args:
            inputs: Batched Sequence of Token Ids

        Shape:
            - inputs: (batch_size, sequence_length)
            - outputs: (batch_size, sequence_length, d_model)
        """

        outputs = self.embedding(inputs)

        # Use fancy indexing to extract the positional encodings until position sequence_length
        sequence_length = inputs.shape[-1]
        pos_encoding = 0
        if self.pos_encoding is not None:
            pos_encoding = self.pos_encoding[:sequence_length]

        outputs = outputs + pos_encoding

        return outputs