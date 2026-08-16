"""PCA/SVD baseline: linear compression at matched compression ratios, for comparison to CsiNet."""

import numpy as np
from sklearn.decomposition import PCA

from csinet.metrics import nmse_db


def flatten_complex(h: np.ndarray) -> np.ndarray:
    """(N,32,32) complex -> (N, 2048) real: concat[Re.flatten, Im.flatten]."""
    n = h.shape[0]
    return np.concatenate([h.real.reshape(n, -1), h.imag.reshape(n, -1)], axis=1).astype(np.float32)


def unflatten_complex(v: np.ndarray, shape: tuple[int, int] = (32, 32)) -> np.ndarray:
    """(N,2048) real -> (N,32,32) complex: inverse of flatten_complex."""
    n = v.shape[0]
    half = v.shape[1] // 2
    real = v[:, :half].reshape(n, *shape)
    imag = v[:, half:].reshape(n, *shape)
    return real + 1j * imag


def run_pca_baseline(train_h: np.ndarray, test_h: np.ndarray, m: int) -> float:
    """Fit PCA (n_components=m) on train, reconstruct test, return NMSE(dB)."""
    x_train = flatten_complex(train_h)
    x_test = flatten_complex(test_h)

    # random_state fixed: for large m, sklearn's "auto" solver heuristic picks a
    # randomized SVD, which is nondeterministic run-to-run (~0.01-0.03 dB jitter
    # observed at m=512) without a fixed seed -- doesn't change any conclusion here,
    # but makes the exact reported numbers unreproducible otherwise.
    pca = PCA(n_components=m, random_state=0)
    pca.fit(x_train)
    x_test_rec = pca.inverse_transform(pca.transform(x_test))

    return nmse_db(unflatten_complex(x_test), unflatten_complex(x_test_rec))
