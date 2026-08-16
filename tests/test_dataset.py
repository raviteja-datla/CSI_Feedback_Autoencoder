import os

from csinet.channel_model import INDOOR
from csinet.dataset import CsiDataset, build_split, save_split


def test_dataset_getitem_shape(tmp_path):
    data = build_split(INDOOR, n_samples=50, seed=0)
    assert data["h"].shape == (50, 32, 32)
    assert data["scale"].shape == (50,)

    path = os.path.join(tmp_path, "indoor_test.npz")
    save_split(data, path)

    ds = CsiDataset(path)
    assert len(ds) == 50
    x, s = ds[0]
    assert tuple(x.shape) == (2, 32, 32)
    assert bool((x.abs() < 1e6).all())  # finite, sane magnitude
    assert s.shape == ()
