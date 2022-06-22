# using pytorch as the "control" vs our numpy experiment
from numpy_transformer.basic_self_attention import sample_batch, simple_self_attention
import numpy as np
import torch
import torch.nn.functional as F


def torch_simple_self_attention(x: np.ndarray) -> np.ndarray:
    mini_batch = torch.tensor(x)
    raw_weights = torch.bmm(mini_batch, mini_batch.transpose(1, 2))
    weights = F.softmax(raw_weights, dim=2)
    y = torch.bmm(weights, mini_batch)

    return y.numpy()


def test_simple_self_attention():
    mini_batch = sample_batch()

    control = torch_simple_self_attention(mini_batch)
    np_impl = simple_self_attention(mini_batch)

    assert np.allclose(control, np_impl)
