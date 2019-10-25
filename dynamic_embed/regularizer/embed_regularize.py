import numpy as np
import torch


def embedded_dropout(embed, words, dropout=0.1):
    """
    In each pass drop some embeddings
    :param embed: nn.Embedding
    :param words: input index
    :param dropout:
    :param scale:
    :return:
    """
    if dropout:
        mask = embed.weight.data.new_empty((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = torch.nn.functional.embedding(words, masked_embed_weight,
                                      padding_idx, embed.max_norm, embed.norm_type,
                                      embed.scale_grad_by_freq, embed.sparse
                                      )
    return X


if __name__ == '__main__':
    V = 50
    h = 4
    bptt = 10
    batch_size = 2

    embed = torch.nn.Embedding(V, h)

    words = np.random.randint(low=0, high=V, size=(batch_size, bptt))
    words = torch.LongTensor(words)

    origX = embed(words)
    X = embedded_dropout(embed, words)

    print(origX)
    print(X)
