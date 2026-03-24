from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyfade import FolderResult, fade, fade_array, fade_image, load_image
from pyfade._compat import (
    border_in,
    border_out,
    conv2_same,
    entropy_uint8_blocks,
    fspecial_gaussian,
    imfilter_replicate,
    matlab_colon,
    nanvar_sample,
    rgb2gray_matlab_uint8,
    split_blocks,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_ONE_DIR = Path(__file__).resolve().parent / "sample_one"


class TensorLike:
    def __init__(self, array: np.ndarray) -> None:
        self._array = np.asarray(array)

    def detach(self) -> "TensorLike":
        return self

    def cpu(self) -> "TensorLike":
        return self

    def numpy(self) -> np.ndarray:
        return self._array


def make_test_image() -> np.ndarray:
    rows, cols = np.indices((16, 16))
    image = np.empty((16, 16, 3), dtype=np.uint8)
    image[:, :, 0] = (rows * 17 + cols * 11) % 256
    image[:, :, 1] = (rows * 7 + cols * 19 + 23) % 256
    image[:, :, 2] = (rows * 29 + cols * 3 + 101) % 256
    return image


def test_matlab_colon_matches_ce_grid() -> None:
    x = matlab_colon(-9.75, 9.75, 1.0)
    assert x.shape == (20,)
    assert x[0] == -9.75
    assert x[-1] == 9.25


def test_border_in_and_out_match_matlab_layout() -> None:
    array = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
    padded = border_in(array, 4)
    expected = np.array(
        [
            [1, 2, 1, 2, 3, 4, 3, 4],
            [5, 6, 5, 6, 7, 8, 7, 8],
            [1, 2, 1, 2, 3, 4, 3, 4],
            [5, 6, 5, 6, 7, 8, 7, 8],
            [9, 10, 9, 10, 11, 12, 11, 12],
            [13, 14, 13, 14, 15, 16, 15, 16],
            [9, 10, 9, 10, 11, 12, 11, 12],
            [13, 14, 13, 14, 15, 16, 15, 16],
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(padded, expected)
    np.testing.assert_array_equal(border_out(padded, 4), array)


def test_entropy_and_rgb2gray_are_deterministic() -> None:
    image = np.array(
        [
            [[0, 0, 0], [255, 255, 255]],
            [[255, 0, 0], [0, 255, 0]],
        ],
        dtype=np.uint8,
    )
    gray = rgb2gray_matlab_uint8(image)
    np.testing.assert_array_equal(gray, np.array([[0.0, 255.0], [76.0, 150.0]]))

    block = np.array([[[[0, 0], [255, 255]]]], dtype=np.uint8)
    entropy = entropy_uint8_blocks(block)
    np.testing.assert_allclose(entropy, np.array([[1.0]]))


def test_conv2_same_matches_matlab_even_kernel_alignment() -> None:
    image = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    kernel = np.array([[1.0, 2.0]])
    expected = np.array([[4.0, 7.0, 6.0], [13.0, 16.0, 12.0]])
    np.testing.assert_array_equal(conv2_same(image, kernel), expected)


def test_imfilter_replicate_matches_matlab_separable_gaussian_accumulation() -> None:
    image = np.full((16, 16), 255.0, dtype=np.float64)
    kernel = fspecial_gaussian((7, 7), 7.0 / 6.0)
    filtered = imfilter_replicate(image, kernel)
    np.testing.assert_allclose(filtered[8, 8], 255.00000000000009, rtol=0.0, atol=1e-13)


def test_split_blocks_matches_im2col_distinct_patch_order() -> None:
    array = np.arange(1, 17, dtype=np.float64).reshape(4, 4, order="F")
    blocks = split_blocks(array, 2)
    columns = [blocks[i, j].reshape(-1, order="F") for j in range(blocks.shape[1]) for i in range(blocks.shape[0])]
    expected = np.array(
        [
            [1, 3, 9, 11],
            [2, 4, 10, 12],
            [5, 7, 13, 15],
            [6, 8, 14, 16],
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(np.column_stack(columns), expected)


def test_nanvar_sample_matches_matlab_singleton_behavior() -> None:
    blocks = np.array([[1.0, np.nan], [np.nan, np.nan], [1.0, 1.0]])
    result = nanvar_sample(blocks, axis=(1,))
    np.testing.assert_array_equal(result, np.array([0.0, np.nan, 0.0]))


def test_fade_matches_regression_fixture() -> None:
    image = make_test_image()
    score, density_map = fade_array(image, return_map=True)
    assert density_map.shape == (2, 2)
    np.testing.assert_allclose(score, 0.1813003412312952, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        density_map,
        np.array(
            [
                [0.215179855687064, 0.171280844331513],
                [0.222215457146475, 0.134828067185145],
            ]
        ),
        rtol=0.0,
        atol=1e-12,
    )


def test_general_fade_accepts_image_path() -> None:
    image_path = FIXTURES_DIR / "grid16.png"
    score_from_path = fade(image_path)
    score_from_array = fade(load_image(image_path))
    np.testing.assert_allclose(score_from_path, score_from_array, rtol=0.0, atol=1e-12)


def test_general_fade_accepts_folder_path_with_progress_and_workers() -> None:
    result = fade(SAMPLE_ONE_DIR, workers=2, progress=True)
    assert isinstance(result, FolderResult)
    assert list(result.scores) == ["AM_Bing_211.png"]
    expected = fade_image(SAMPLE_ONE_DIR / "AM_Bing_211.png")
    np.testing.assert_allclose(result.scores["AM_Bing_211.png"], expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(result.mean_score, expected, rtol=0.0, atol=1e-12)


def test_general_fade_accepts_npy_path(tmp_path: Path) -> None:
    image = make_test_image()
    npy_path = tmp_path / "sample.npy"
    np.save(npy_path, image)
    score_from_npy = fade(npy_path)
    score_from_array = fade(image)
    np.testing.assert_allclose(score_from_npy, score_from_array, rtol=0.0, atol=1e-12)


def test_fade_accepts_normalized_numpy_batch_with_workers() -> None:
    image_a = make_test_image().astype(np.float32) / 255.0
    image_b = np.flip(make_test_image(), axis=1).astype(np.float32) / 255.0
    batch = np.stack([np.moveaxis(image_a, -1, 0), np.moveaxis(image_b, -1, 0)], axis=0)
    scores, maps = fade(batch, workers=2, progress=True, return_map=True)
    expected_score_a, expected_map_a = fade(image_a, return_map=True)
    expected_score_b, expected_map_b = fade(image_b, return_map=True)
    np.testing.assert_allclose(scores, np.array([expected_score_a, expected_score_b]), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(maps, np.stack([expected_map_a, expected_map_b], axis=0), rtol=0.0, atol=1e-12)


def test_fade_accepts_tensor_like_input() -> None:
    image = make_test_image()
    tensor_like = TensorLike(np.expand_dims(np.moveaxis(image, -1, 0), axis=0))
    scores, maps = fade(tensor_like, return_map=True)
    expected_score, expected_map = fade(image, return_map=True)
    np.testing.assert_allclose(scores, np.array([expected_score]), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(maps, np.expand_dims(expected_map, axis=0), rtol=0.0, atol=1e-12)


def test_zero_batch_does_not_emit_runtime_warnings(recwarn: pytest.WarningsRecorder) -> None:
    batch = np.zeros((2, 3, 16, 16), dtype=np.uint8)
    scores = fade(batch, workers=2, progress=False)
    assert scores.shape == (2,)
    assert len(recwarn) == 0


def test_fade_accepts_actual_torch_tensor_if_available() -> None:
    torch = pytest.importorskip("torch")
    image = make_test_image()
    batch = torch.from_numpy(np.expand_dims(np.moveaxis(image, -1, 0), axis=0)).to(torch.float32) / 255.0
    scores, maps = fade(batch, workers=2, progress=True, return_map=True)
    expected_score, expected_map = fade(image, return_map=True)
    np.testing.assert_allclose(scores, np.array([expected_score]), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(maps, np.expand_dims(expected_map, axis=0), rtol=0.0, atol=1e-12)
