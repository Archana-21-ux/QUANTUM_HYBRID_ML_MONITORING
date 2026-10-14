import numpy as np

from quantumguard.drift.preprocess import PCAReducer, iter_windows, subsample_window

RNG = np.random.default_rng(7)


def test_pca_reducer_shapes_and_determinism():
    reference = RNG.normal(size=(1000, 6))
    window = RNG.normal(size=(300, 6))

    reducer = PCAReducer(n_components=4)
    reducer.fit(reference)
    out1 = reducer.transform(window)
    out2 = reducer.transform(window)
    assert out1.shape == (300, 4)
    np.testing.assert_array_equal(out1, out2)


def test_pca_reducer_caps_components_to_data():
    reference = RNG.normal(size=(50, 3))
    reducer = PCAReducer(n_components=8).fit(reference)
    assert reducer.transform(reference).shape == (50, 3)


def test_subsample_deterministic_and_capped():
    X = RNG.normal(size=(1000, 5))
    s1 = subsample_window(X, max_samples=300, seed=11)
    s2 = subsample_window(X, max_samples=300, seed=11)
    assert s1.shape == (300, 5)
    np.testing.assert_array_equal(s1, s2)

    small = RNG.normal(size=(100, 5))
    np.testing.assert_array_equal(subsample_window(small, max_samples=300), small)


def test_iter_windows_counts_and_sizes():
    X = np.arange(2500).reshape(-1, 1)
    windows = list(iter_windows(X, window_size=500, stride=500))
    assert len(windows) == 5
    assert all(w.shape == (500, 1) for _, w in windows)
    assert [wid for wid, _ in windows] == [0, 1, 2, 3, 4]
    # trailing partial window dropped
    windows = list(iter_windows(X[:2400], window_size=500, stride=500))
    assert len(windows) == 4
