# self attention, by default, is permutation equivariant
# (not affected by permuation i.e ignores the sequential nature of the input)

from numpy_transformer.ops import softmax
import numpy as np

MINI_BATCH_SIZE = 2
EMBEDDING_DIMS = 64


def get_embedding(candidate: str) -> np.ndarray:
    return np.array(
        [np.random.rand(1, EMBEDDING_DIMS) for _ in candidate.split(" ")]
    ).squeeze()


def simple_self_attention(mini_batch: np.ndarray) -> np.ndarray:

    # row-wise softmax treating EMBEDDING_DIMS as number of classes K
    # raw_weights = (2 x 6 x 64) * (2 x 64 x 6) = (2 x 6 x 6)
    raw_weights = np.matmul(mini_batch, np.transpose(mini_batch, axes=[0, 2, 1]))
    weights = softmax(raw_weights)

    # y = (2 x 6 x 6) * (2 x 6 x 64) = (2 x 6 x 64)
    y = np.matmul(weights, mini_batch)

    return y


def sample_batch() -> np.ndarray:
    prim_cand = "The quick brown fox jumped over"
    sec_cand = "Squdgy fez, blank jimp crwth vox"
    mini_batch = np.array([get_embedding(prim_cand), get_embedding(sec_cand)])

    return mini_batch


if __name__ == "__main__":
    simple_self_attention(sample_batch())
