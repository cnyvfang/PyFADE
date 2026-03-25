from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
from numba import njit
from scipy import ndimage, signal


@njit(cache=True)
def _rgb2hsv_matlab_uint8_numba(image: np.ndarray) -> np.ndarray:
    height, width, _ = image.shape
    hsv = np.empty((height, width, 3), dtype=np.float64)
    for row in range(height):
        for col in range(width):
            r = image[row, col, 0] / 255.0
            g = image[row, col, 1] / 255.0
            b = image[row, col, 2] / 255.0

            maxc = r if r >= g and r >= b else (g if g >= b else b)
            minc = r if r <= g and r <= b else (g if g <= b else b)
            delta = maxc - minc

            h = 0.0
            s = 0.0
            if maxc > 0.0:
                s = delta / maxc

            if delta > 0.0:
                if maxc == r:
                    h = ((g - b) / delta) % 6.0
                elif maxc == g:
                    h = ((b - r) / delta) + 2.0
                else:
                    h = ((r - g) / delta) + 4.0
                h = (h / 6.0) % 1.0

            hsv[row, col, 0] = h
            hsv[row, col, 1] = s
            hsv[row, col, 2] = maxc
    return hsv


@njit(cache=True)
def _nanvar_last2_numba(blocks: np.ndarray) -> np.ndarray:
    patch_rows, patch_cols, block_rows, block_cols = blocks.shape
    out = np.empty((patch_rows, patch_cols), dtype=np.float64)
    for patch_row in range(patch_rows):
        for patch_col in range(patch_cols):
            count = 0
            total = 0.0
            for row in range(block_rows):
                for col in range(block_cols):
                    value = blocks[patch_row, patch_col, row, col]
                    if not np.isnan(value):
                        count += 1
                        total += value

            if count == 0:
                out[patch_row, patch_col] = np.nan
                continue
            if count == 1:
                out[patch_row, patch_col] = 0.0
                continue

            mean = total / count
            sumsq = 0.0
            for row in range(block_rows):
                for col in range(block_cols):
                    value = blocks[patch_row, patch_col, row, col]
                    if not np.isnan(value):
                        delta = value - mean
                        sumsq += delta * delta
            out[patch_row, patch_col] = sumsq / (count - 1.0)
    return out


@njit(cache=True)
def _nanvar_axis0_2d_numba(blocks: np.ndarray) -> np.ndarray:
    rows, cols = blocks.shape
    out = np.empty(cols, dtype=np.float64)
    for col in range(cols):
        count = 0
        total = 0.0
        for row in range(rows):
            value = blocks[row, col]
            if not np.isnan(value):
                count += 1
                total += value

        if count == 0:
            out[col] = np.nan
            continue
        if count == 1:
            out[col] = 0.0
            continue

        mean = total / count
        sumsq = 0.0
        for row in range(rows):
            value = blocks[row, col]
            if not np.isnan(value):
                delta = value - mean
                sumsq += delta * delta
        out[col] = sumsq / (count - 1.0)
    return out


@njit(cache=True)
def _entropy_blocks_u8_numba(blocks: np.ndarray) -> np.ndarray:
    patch_rows, patch_cols, block_rows, block_cols = blocks.shape
    block_area = block_rows * block_cols
    entropy = np.empty((patch_rows, patch_cols), dtype=np.float64)
    terms = np.zeros(block_area + 1, dtype=np.float64)
    for count in range(1, block_area + 1):
        prob = count / block_area
        terms[count] = -(prob * np.log2(prob))

    hist = np.empty(256, dtype=np.int64)
    for patch_row in range(patch_rows):
        for patch_col in range(patch_cols):
            for idx in range(256):
                hist[idx] = 0
            for row in range(block_rows):
                for col in range(block_cols):
                    hist[int(blocks[patch_row, patch_col, row, col])] += 1

            acc = 0.0
            for idx in range(256):
                count = hist[idx]
                if count != 0:
                    acc += terms[count]
            entropy[patch_row, patch_col] = acc
    return entropy


def matlab_colon(start: float, stop: float, step: float = 1.0) -> np.ndarray:
    """Replicate MATLAB's colon operator for a scalar step."""
    start = float(start)
    stop = float(stop)
    step = float(step)
    if step == 0:
        raise ValueError("step must be non-zero")
    if (step > 0 and start > stop) or (step < 0 and start < stop):
        return np.array([], dtype=np.float64)
    span = (stop - start) / step
    count = int(np.floor(span + 1e-12)) + 1
    return start + step * np.arange(count, dtype=np.float64)


def fspecial_gaussian(hsize: Iterable[int], sigma: float) -> np.ndarray:
    h, w = [int(v) for v in hsize]
    y = np.arange(-(h - 1) / 2.0, (h - 1) / 2.0 + 1.0, dtype=np.float64)
    x = np.arange(-(w - 1) / 2.0, (w - 1) / 2.0 + 1.0, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-((xx * xx) + (yy * yy)) / (2.0 * sigma * sigma))
    kernel_sum = kernel.sum()
    if kernel_sum != 0:
        kernel = kernel / kernel_sum
    return kernel


def imfilter_replicate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image_f = image.astype(np.float64, copy=False)
    kernel_f = kernel.astype(np.float64, copy=False)
    factors = _rank1_kernel_factors(kernel_f)
    if factors is not None:
        column_kernel, row_kernel = factors
        filtered = ndimage.correlate1d(image_f, column_kernel, axis=0, mode="nearest")
        return ndimage.correlate1d(filtered, row_kernel, axis=1, mode="nearest")
    return ndimage.correlate(image_f, kernel_f, mode="nearest")


@lru_cache(maxsize=16)
def _rank1_kernel_factors_cached(shape: tuple[int, int], raw: bytes) -> tuple[np.ndarray, np.ndarray] | None:
    kernel = np.frombuffer(raw, dtype=np.float64).reshape(shape)
    u, s, vh = np.linalg.svd(kernel, full_matrices=False)
    if s.size > 1 and s[1] > (np.finfo(np.float64).eps * max(shape) * s[0]):
        return None

    column_kernel = np.sqrt(s[0]) * u[:, 0]
    row_kernel = np.sqrt(s[0]) * vh[0, :]
    center_row = shape[0] // 2
    center_col = shape[1] // 2
    if column_kernel[center_row] < 0 or (
        column_kernel[center_row] == 0 and row_kernel[center_col] < 0
    ):
        column_kernel = -column_kernel
        row_kernel = -row_kernel
    return column_kernel.copy(), row_kernel.copy()


def _rank1_kernel_factors(kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    return _rank1_kernel_factors_cached(kernel.shape, kernel.tobytes())


def conv2_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image = image.astype(np.float64, copy=False)
    kernel = kernel.astype(np.float64, copy=False)
    if kernel.shape[0] == 1:
        return ndimage.convolve1d(image, kernel.reshape(-1), axis=1, mode="constant", cval=0.0, origin=0)
    if kernel.shape[1] == 1:
        return ndimage.convolve1d(image, kernel.reshape(-1), axis=0, mode="constant", cval=0.0, origin=0)
    full = signal.convolve2d(image, kernel, mode="full")
    row_start = kernel.shape[0] // 2
    col_start = kernel.shape[1] // 2
    row_end = row_start + image.shape[0]
    col_end = col_start + image.shape[1]
    return full[row_start:row_end, col_start:col_end]


def _round_like_matlab_integer(values: np.ndarray, dtype: np.dtype | type[np.uint8] | type[np.uint16]) -> np.ndarray:
    target_dtype = np.dtype(dtype)
    if not np.issubdtype(target_dtype, np.unsignedinteger):
        raise TypeError("target dtype must be an unsigned integer type")
    rounded = np.floor(values + 0.5)
    max_value = np.iinfo(target_dtype).max
    return np.clip(rounded, 0, max_value).astype(target_dtype)


def matlab_uint8_cast(values: np.ndarray) -> np.ndarray:
    """Match MATLAB's uint8(...) conversion, including saturation for uint16 inputs."""
    return _round_like_matlab_integer(np.asarray(values, dtype=np.float64), np.uint8)


def rgb2gray_matlab(image: np.ndarray) -> np.ndarray:
    """Match double(rgb2gray(RGB)) for uint8 and uint16 MATLAB images."""
    if image.dtype not in (np.uint8, np.uint16):
        raise TypeError("rgb2gray_matlab expects a uint8 or uint16 RGB image")
    weights = np.array(
        [0.298936021293775, 0.587043074451121, 0.114020904255103],
        dtype=np.float64,
    )
    gray = np.tensordot(image.astype(np.float64, copy=False), weights, axes=([2], [0]))
    return _round_like_matlab_integer(gray, image.dtype).astype(np.float64)


def rgb2hsv_matlab(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return _rgb2hsv_matlab_uint8_numba(image)
    if np.issubdtype(image.dtype, np.integer):
        scale = float(np.iinfo(image.dtype).max)
    else:
        image_max = float(np.nanmax(image))
        scale = 1.0 if image_max <= 1.0 else 255.0
    rgb = image.astype(np.float64, copy=False) / scale
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    h = np.zeros_like(maxc)
    s = np.zeros_like(maxc)
    v = maxc

    nonzero = maxc > 0
    s[nonzero] = delta[nonzero] / maxc[nonzero]

    delta_nonzero = delta > 0
    rmask = delta_nonzero & (maxc == r)
    gmask = delta_nonzero & (maxc == g) & ~rmask
    bmask = delta_nonzero & (maxc == b) & ~(rmask | gmask)

    h[rmask] = np.mod((g[rmask] - b[rmask]) / delta[rmask], 6.0)
    h[gmask] = ((b[gmask] - r[gmask]) / delta[gmask]) + 2.0
    h[bmask] = ((r[bmask] - g[bmask]) / delta[bmask]) + 4.0
    h = (h / 6.0) % 1.0

    hsv = np.stack([h, s, v], axis=-1)
    return hsv


def border_in(image: np.ndarray, ps: int) -> np.ndarray:
    if ps % 2 == 0:
        uc = ps // 2
        dc = (ps // 2) - 1
    else:
        uc = ps // 2
        dc = uc

    ucb = image[:uc, :]
    dcb = image[-(dc + 1) :, :]
    temp = np.concatenate([ucb, image, dcb], axis=0)

    lcb = temp[:, :uc]
    rcb = temp[:, -(dc + 1) :]
    return np.concatenate([lcb, temp, rcb], axis=1)


def border_out(image: np.ndarray, ps: int) -> np.ndarray:
    if ps % 2 == 0:
        uc = ps // 2
        dc = (ps // 2) - 1
    else:
        uc = ps // 2
        dc = uc

    return image[uc : image.shape[0] - (dc + 1), uc : image.shape[1] - (dc + 1)]


def split_blocks(image: np.ndarray, ps: int) -> np.ndarray:
    row, col = image.shape
    if row % ps != 0 or col % ps != 0:
        raise ValueError("image dimensions must be divisible by block size")
    patch_row_num = row // ps
    patch_col_num = col // ps
    return image.reshape(patch_row_num, ps, patch_col_num, ps).transpose(0, 2, 1, 3)


def nanvar_sample(blocks: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
    blocks = np.asarray(blocks, dtype=np.float64)
    normalized_axis = tuple(sorted(int(value) for value in axis))
    if blocks.ndim == 4 and normalized_axis == (2, 3):
        return _nanvar_last2_numba(blocks)
    if blocks.ndim == 2 and normalized_axis == (0,):
        return _nanvar_axis0_2d_numba(blocks)
    finite_mask = np.isfinite(blocks)
    counts_keepdims = np.sum(finite_mask, axis=axis, keepdims=True)
    counts = np.sum(finite_mask, axis=axis)

    sums = np.nansum(blocks, axis=axis, keepdims=True)
    means = np.divide(sums, counts_keepdims, out=np.zeros_like(sums), where=counts_keepdims > 0)
    centered = np.where(finite_mask, blocks - means, 0.0)
    sumsq = np.sum(centered * centered, axis=axis)

    out = np.full(counts.shape, np.nan, dtype=np.float64)
    out[counts == 1] = 0.0
    valid = counts > 1
    out[valid] = sumsq[valid] / (counts[valid] - 1.0)
    return out


def sample_std(blocks: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.std(blocks, axis=axis, ddof=1)


@lru_cache(maxsize=8)
def _entropy_terms(block_area: int) -> np.ndarray:
    terms = np.zeros(block_area + 1, dtype=np.float64)
    counts = np.arange(1, block_area + 1, dtype=np.float64)
    probs = counts / float(block_area)
    terms[1:] = -(probs * np.log2(probs))
    return terms


def entropy_integer_blocks(blocks: np.ndarray) -> np.ndarray:
    if not np.issubdtype(blocks.dtype, np.integer):
        raise TypeError("entropy_integer_blocks expects integer-valued blocks")
    if blocks.dtype == np.uint8:
        return _entropy_blocks_u8_numba(blocks)
    patch_row_num, patch_col_num, _, _ = blocks.shape
    flat_blocks = blocks.reshape(patch_row_num * patch_col_num, -1)
    block_area = flat_blocks.shape[1]
    sorted_blocks = np.sort(flat_blocks, axis=1)
    starts = np.ones_like(sorted_blocks, dtype=bool)
    starts[:, 1:] = sorted_blocks[:, 1:] != sorted_blocks[:, :-1]
    rows, cols = np.nonzero(starts)
    next_cols = np.empty_like(cols)
    next_cols[:-1] = cols[1:]
    next_cols[-1] = block_area
    row_change = np.empty_like(rows, dtype=bool)
    row_change[:-1] = rows[1:] != rows[:-1]
    row_change[-1] = True
    next_cols[row_change] = block_area
    run_lengths = next_cols - cols
    entropy = np.bincount(
        rows,
        weights=_entropy_terms(block_area)[run_lengths],
        minlength=flat_blocks.shape[0],
    )
    return entropy.reshape(patch_row_num, patch_col_num)
