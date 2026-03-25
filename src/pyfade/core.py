from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import struct
from typing import Iterable, Sequence
import zlib

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.io import loadmat
from scipy.linalg import solve
import turbojpeg
from tqdm.auto import tqdm

from ._compat import (
    border_in,
    border_out,
    conv2_same,
    entropy_integer_blocks,
    fspecial_gaussian,
    imfilter_replicate,
    matlab_uint8_cast,
    matlab_colon,
    nanvar_sample,
    rgb2gray_matlab,
    rgb2hsv_matlab,
    sample_std,
    split_blocks,
)

PATCH_SIZE = 8
MODEL_DIR = Path(__file__).resolve().parent / "models"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
FLOAT_UNIT_RANGE_TOL = 1e-6


@dataclass(frozen=True)
class ReferenceModel:
    mu: np.ndarray
    cov: np.ndarray
    half_cov: np.ndarray
    half_cov_solve_ones: np.ndarray
    half_cov_ones_quad: float


@dataclass(frozen=True)
class FolderResult:
    scores: dict[str, float]
    density_maps: dict[str, np.ndarray] | None
    mean_score: float
    min_score: float
    max_score: float


def load_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    suffix = image_path.suffix.lower()
    header = _read_png_header(image_path) if suffix == ".png" else None
    if header is not None and header["bit_depth"] == 16:
        return _load_png_preserve_precision(image_path)
    if suffix in (".jpg", ".jpeg"):
        decoded = _load_jpeg_with_turbojpeg(image_path)
        if decoded is not None:
            return decoded
    with Image.open(image_path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def _load_jpeg_with_turbojpeg(path: Path) -> np.ndarray | None:
    try:
        decoded = turbojpeg.decompress(path.read_bytes())
    except Exception:
        return None

    colorspace = getattr(decoded, "colorspace", None)
    colorspace_name = getattr(colorspace, "name", None)
    array = np.asarray(decoded)
    if colorspace_name in {"YCbCr", "RGB"} and array.ndim == 3 and array.shape[2] == 3:
        return np.array(array, dtype=np.uint8, copy=True)
    if colorspace_name == "GRAY" and array.ndim == 2:
        return np.repeat(array[:, :, None], 3, axis=2).astype(np.uint8, copy=False)
    return None


def _read_png_header(path: Path) -> dict[str, int] | None:
    with path.open("rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            return None
        length = struct.unpack(">I", handle.read(4))[0]
        chunk_type = handle.read(4)
        if chunk_type != b"IHDR" or length != 13:
            return None
        width, height, bit_depth, color_type, compression, flt, interlace = struct.unpack(">IIBBBBB", handle.read(length))
    return {
        "width": int(width),
        "height": int(height),
        "bit_depth": int(bit_depth),
        "color_type": int(color_type),
        "compression": int(compression),
        "filter": int(flt),
        "interlace": int(interlace),
    }


def _png_channel_count(color_type: int) -> int:
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise ValueError(f"unsupported PNG color type: {color_type}")
    return channels


def _png_paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _load_png_preserve_precision(path: Path) -> np.ndarray:
    header = _read_png_header(path)
    if header is None:
        raise ValueError(f"{path} is not a valid PNG")
    if header["compression"] != 0 or header["filter"] != 0 or header["interlace"] != 0:
        raise ValueError("unsupported PNG encoding; expected standard non-interlaced PNG")

    idat_chunks: list[bytes] = []
    with path.open("rb") as handle:
        handle.read(8)
        while True:
            raw_length = handle.read(4)
            if not raw_length:
                break
            length = struct.unpack(">I", raw_length)[0]
            chunk_type = handle.read(4)
            chunk_data = handle.read(length)
            handle.read(4)
            if chunk_type == b"IDAT":
                idat_chunks.append(chunk_data)
            elif chunk_type == b"IEND":
                break

    channels = _png_channel_count(header["color_type"])
    bytes_per_sample = 2 if header["bit_depth"] == 16 else 1
    bytes_per_pixel = channels * bytes_per_sample
    row_bytes = header["width"] * bytes_per_pixel
    decompressed = zlib.decompress(b"".join(idat_chunks))
    expected_size = header["height"] * (row_bytes + 1)
    if len(decompressed) != expected_size:
        raise ValueError("unexpected decompressed PNG size")

    decoded = np.empty((header["height"], row_bytes), dtype=np.uint8)
    prev_row = np.zeros(row_bytes, dtype=np.uint8)
    offset = 0
    for row_idx in range(header["height"]):
        filter_type = decompressed[offset]
        offset += 1
        row = np.frombuffer(decompressed[offset : offset + row_bytes], dtype=np.uint8).copy()
        offset += row_bytes
        if filter_type == 0:
            pass
        elif filter_type == 1:
            for idx in range(bytes_per_pixel, row_bytes):
                row[idx] = (int(row[idx]) + int(row[idx - bytes_per_pixel])) & 0xFF
        elif filter_type == 2:
            row = ((row.astype(np.uint16) + prev_row.astype(np.uint16)) & 0xFF).astype(np.uint8)
        elif filter_type == 3:
            for idx in range(row_bytes):
                left = int(row[idx - bytes_per_pixel]) if idx >= bytes_per_pixel else 0
                up = int(prev_row[idx])
                row[idx] = (int(row[idx]) + ((left + up) // 2)) & 0xFF
        elif filter_type == 4:
            for idx in range(row_bytes):
                left = int(row[idx - bytes_per_pixel]) if idx >= bytes_per_pixel else 0
                up = int(prev_row[idx])
                up_left = int(prev_row[idx - bytes_per_pixel]) if idx >= bytes_per_pixel else 0
                row[idx] = (int(row[idx]) + _png_paeth_predictor(left, up, up_left)) & 0xFF
        else:
            raise ValueError(f"unsupported PNG filter type: {filter_type}")
        decoded[row_idx] = row
        prev_row = row

    dtype = ">u2" if header["bit_depth"] == 16 else np.uint8
    image = np.frombuffer(decoded.tobytes(), dtype=dtype).reshape(header["height"], header["width"], channels)
    if header["bit_depth"] == 16:
        image = image.astype(np.uint16, copy=False)
    if header["color_type"] in (4, 6):
        image = image[:, :, :-1]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    return image

def _to_numpy_array(image: object) -> np.ndarray:
    if hasattr(image, "detach") and hasattr(image, "cpu") and hasattr(image, "numpy"):
        return np.asarray(image.detach().cpu().numpy())
    return np.asarray(image)


def _float_to_integer_image(array: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(array)
    if np.any(finite_mask):
        finite_values = array[finite_mask]
        if float(np.min(finite_values)) >= -FLOAT_UNIT_RANGE_TOL and float(np.max(finite_values)) <= (1.0 + FLOAT_UNIT_RANGE_TOL):
            array = array * 255.0
            upper = 255.0
            dtype = np.uint8
        else:
            upper = 255.0 if float(np.max(finite_values)) <= (255.0 + FLOAT_UNIT_RANGE_TOL) else 65535.0
            dtype = np.uint8 if upper <= 255.0 else np.uint16
    else:
        upper = 255.0
        dtype = np.uint8
    array = np.nan_to_num(array, nan=0.0, posinf=upper, neginf=0.0)
    array = np.clip(array, 0.0, upper)
    return np.clip(np.floor(array + 0.5), 0, upper).astype(dtype)


def _ensure_rgb_image(image: object) -> np.ndarray:
    array = _to_numpy_array(image)
    if array.ndim != 3:
        raise ValueError("expected an RGB image with shape (H, W, 3)")
    channels_last = array.shape[2] in (3, 4)
    channels_first = array.shape[0] in (3, 4)
    if channels_first and not channels_last:
        array = np.moveaxis(array[:3], 0, -1)
    elif channels_last:
        array = array[:, :, :3]
    else:
        raise ValueError("expected an RGB image with shape (H, W, 3) or (3, H, W)")

    if np.issubdtype(array.dtype, np.floating):
        array = _float_to_integer_image(array)
    elif array.dtype == np.uint8 or array.dtype == np.uint16:
        return array
    elif np.issubdtype(array.dtype, np.integer):
        if np.iinfo(array.dtype).max <= 255:
            array = np.clip(array, 0, 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 65535).astype(np.uint16)
    else:
        raise TypeError("expected an integer or floating-point RGB image")
    return array


def _normalize_image_batch(image: object) -> tuple[list[np.ndarray], bool]:
    array = _to_numpy_array(image)
    if array.ndim == 4:
        channels_first = array.shape[1] in (3, 4)
        channels_last = array.shape[3] in (3, 4)
        if channels_first and not channels_last:
            images = [np.moveaxis(array[idx, :3], 0, -1) for idx in range(array.shape[0])]
            return [_ensure_rgb_image(item) for item in images], True
        if channels_last:
            images = [array[idx, :, :, :3] for idx in range(array.shape[0])]
            return [_ensure_rgb_image(item) for item in images], True
        raise ValueError("expected a batch with shape (B, 3, H, W) or (B, H, W, 3)")

    return [_ensure_rgb_image(array)], False


def _trim_to_patch_grid(image: np.ndarray, patch_size: int = PATCH_SIZE) -> np.ndarray:
    row, col, _ = image.shape
    patch_row_num = row // patch_size
    patch_col_num = col // patch_size
    return image[: patch_row_num * patch_size, : patch_col_num * patch_size, :3]


@lru_cache(maxsize=None)
def _load_reference_model(name: str, model_dir: str) -> ReferenceModel:
    mat = loadmat(Path(model_dir) / name)
    if "fogfree" in name:
        mu_key = "mu_fogfreeparam"
        cov_key = "cov_fogfreeparam"
    else:
        mu_key = "mu_foggyparam"
        cov_key = "cov_foggyparam"
    cov = np.asarray(mat[cov_key], dtype=np.float64)
    half_cov = cov / 2.0
    ones = np.ones(cov.shape[0], dtype=np.float64)
    half_cov_solve_ones = solve(half_cov, ones, assume_a="sym", check_finite=False)
    return ReferenceModel(
        mu=np.asarray(mat[mu_key], dtype=np.float64).reshape(-1),
        cov=cov,
        half_cov=half_cov,
        half_cov_solve_ones=half_cov_solve_ones,
        half_cov_ones_quad=float(ones @ half_cov_solve_ones),
    )


def _load_models(model_dir: str | Path | None) -> tuple[ReferenceModel, ReferenceModel]:
    model_dir_str = str(Path(model_dir) if model_dir is not None else MODEL_DIR)
    fogfree = _load_reference_model("natural_fogfree_image_features_ps8.mat", model_dir_str)
    foggy = _load_reference_model("natural_foggy_image_features_ps8.mat", model_dir_str)
    return fogfree, foggy


@lru_cache(maxsize=1)
def _mscn_window() -> np.ndarray:
    window = fspecial_gaussian((7, 7), 7.0 / 6.0)
    return window / window.sum()


def _mscn_filter_replicate(image: np.ndarray) -> np.ndarray:
    return imfilter_replicate(image.astype(np.float64, copy=False), _mscn_window())


@lru_cache(maxsize=1)
def _constant_window_lookup() -> tuple[np.ndarray, np.ndarray]:
    data = loadmat(MODEL_DIR / "matlab_constant_window_lookup.mat")
    return (
        np.asarray(data["mu_offsets"], dtype=np.float64).reshape(-1),
        np.asarray(data["m2_offsets"], dtype=np.float64).reshape(-1),
    )


def _apply_uint8_constant_window_lookup(
    gray: np.ndarray,
    mu: np.ndarray,
    second_moment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    local_min = ndimage.minimum_filter(gray, size=7, mode="nearest")
    local_max = ndimage.maximum_filter(gray, size=7, mode="nearest")
    constant_window = local_min == local_max
    if not np.any(constant_window):
        return mu, second_moment

    gray_uint8 = matlab_uint8_cast(gray)
    mu_offsets, m2_offsets = _constant_window_lookup()
    adjusted_mu = mu.copy()
    adjusted_second_moment = second_moment.copy()
    adjusted_mu[constant_window] = gray[constant_window] + mu_offsets[gray_uint8[constant_window]]
    adjusted_second_moment[constant_window] = (
        gray[constant_window] * gray[constant_window]
        + m2_offsets[gray_uint8[constant_window]]
    )
    return adjusted_mu, adjusted_second_moment


@lru_cache(maxsize=1)
def _ce_kernels() -> tuple[np.ndarray, np.ndarray]:
    sigma = 3.25
    break_off_sigma = 3.0
    filter_size = break_off_sigma * sigma
    x = matlab_colon(-filter_size, filter_size, 1.0)
    gauss = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp((x * x) / (-2.0 * sigma * sigma))
    gauss = gauss / gauss.sum()
    gx = ((x * x) / (sigma**4) - (1.0 / (sigma**2))) * gauss
    gx = gx - (gx.sum() / x.size)
    gx = gx / np.sum(0.5 * x * x * gx)
    return gx.reshape(1, -1), gx.reshape(-1, 1)


def _ce(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    semisaturation = 0.1
    t1 = 9.225496406318721e-004 * 255.0
    t2 = 8.969246659629488e-004 * 255.0
    t3 = 2.069284034165411e-004 * 255.0
    border_s = 20
    gx_row, gx_col = _ce_kernels()

    image_f = image.astype(np.float64, copy=False)
    r = image_f[:, :, 0]
    g = image_f[:, :, 1]
    b = image_f[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    by = 0.5 * r + 0.5 * g - b
    rg = r - g
    row, col, _ = image.shape

    ce_gray = np.zeros((row, col), dtype=np.float64)
    ce_by = np.zeros((row, col), dtype=np.float64)
    ce_rg = np.zeros((row, col), dtype=np.float64)

    def _channel_ce(channel: np.ndarray, threshold: float, out: np.ndarray) -> None:
        padded = border_in(channel, border_s)
        cx = conv2_same(padded, gx_row)
        cy = conv2_same(padded, gx_col)
        contrast = np.sqrt((cx * cx) + (cy * cy))
        contrast = border_out(contrast, border_s)
        contrast_max = np.max(contrast)
        with np.errstate(divide="ignore", invalid="ignore"):
            response = (contrast * contrast_max) / (contrast + (contrast_max * semisaturation))
        response = response - threshold
        mask = response > 1e-7
        out[mask] = response[mask]

    _channel_ce(gray, t1, ce_gray)
    _channel_ce(by, t2, ce_by)
    _channel_ce(rg, t3, ce_rg)
    return ce_gray, ce_by, ce_rg


def _mscn_from_moments(gray: np.ndarray, mu: np.ndarray, second_moment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu_sq = mu * mu
    sigma = np.sqrt(np.abs(second_moment - mu_sq))
    return sigma, (gray - mu) / (sigma + 1.0)


def _extract_features(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    image = _ensure_rgb_image(image)
    image = _trim_to_patch_grid(image, PATCH_SIZE)
    row, col, _ = image.shape
    if row == 0 or col == 0:
        raise ValueError("image must be at least 8x8 pixels")

    image_f = image.astype(np.float64, copy=False)
    r = image_f[:, :, 0]
    g = image_f[:, :, 1]
    b = image_f[:, :, 2]
    ig = rgb2gray_matlab(image)
    ig_uint8 = matlab_uint8_cast(ig)

    irn = r / 255.0
    ign = g / 255.0
    ibn = b / 255.0
    idark = np.minimum(np.minimum(irn, ign), ibn)
    isat = rgb2hsv_matlab(image)[:, :, 1]

    mu = _mscn_filter_replicate(ig)
    second_moment = _mscn_filter_replicate(ig * ig)
    if image.dtype == np.uint8:
        mu, second_moment = _apply_uint8_constant_window_lookup(ig, mu, second_moment)
    sigma, mscn = _mscn_from_moments(ig, mu, second_moment)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = sigma / mu

    rg = r - g
    by = 0.5 * (r + g) - b

    mscn_blocks = split_blocks(mscn, PATCH_SIZE)
    mscn_var = nanvar_sample(mscn_blocks, axis=(2, 3))

    mscn_pair = mscn * np.roll(mscn, shift=1, axis=0)
    mscn_pair_l = mscn_pair.copy()
    mscn_pair_l[mscn_pair_l > 0] = np.nan
    mscn_pair_r = mscn_pair.copy()
    mscn_pair_r[mscn_pair_r < 0] = np.nan
    mscn_v_pair_l_var = nanvar_sample(split_blocks(mscn_pair_l, PATCH_SIZE), axis=(2, 3))
    mscn_v_pair_r_var = nanvar_sample(split_blocks(mscn_pair_r, PATCH_SIZE), axis=(2, 3))

    mean_sigma = split_blocks(sigma, PATCH_SIZE).mean(axis=(2, 3))
    mean_cv = split_blocks(cv, PATCH_SIZE).mean(axis=(2, 3))

    ce_gray, ce_by, ce_rg = _ce(image)
    mean_ce_gray = split_blocks(ce_gray, PATCH_SIZE).mean(axis=(2, 3))
    mean_ce_by = split_blocks(ce_by, PATCH_SIZE).mean(axis=(2, 3))
    mean_ce_rg = split_blocks(ce_rg, PATCH_SIZE).mean(axis=(2, 3))

    ie = entropy_integer_blocks(split_blocks(ig_uint8, PATCH_SIZE))
    mean_id = split_blocks(idark, PATCH_SIZE).mean(axis=(2, 3))
    mean_is = split_blocks(isat, PATCH_SIZE).mean(axis=(2, 3))

    rg_blocks = split_blocks(rg, PATCH_SIZE)
    by_blocks = split_blocks(by, PATCH_SIZE)
    cf = np.sqrt(sample_std(rg_blocks, axis=(2, 3)) ** 2 + sample_std(by_blocks, axis=(2, 3)) ** 2)
    cf += 0.3 * np.sqrt(rg_blocks.mean(axis=(2, 3)) ** 2 + by_blocks.mean(axis=(2, 3)) ** 2)

    feature_maps = [
        mscn_var,
        mscn_v_pair_r_var,
        mscn_v_pair_l_var,
        mean_sigma,
        mean_cv,
        mean_ce_gray,
        mean_ce_by,
        mean_ce_rg,
        ie,
        mean_id,
        mean_is,
        cf,
    ]
    feat = np.column_stack([feature_map.reshape(-1, order="F") for feature_map in feature_maps])
    feat = np.log1p(feat)
    return feat, (row // PATCH_SIZE, col // PATCH_SIZE)


def _distance_to_model(
    feat: np.ndarray,
    patch_grid_shape: tuple[int, int],
    model: ReferenceModel,
) -> tuple[float, np.ndarray]:
    patch_scalar_var = nanvar_sample(feat.T, axis=(0,))
    mu_matrix = model.mu[None, :] - feat
    solved_mu = solve(model.half_cov, mu_matrix.T, assume_a="sym", check_finite=False).T
    base_distance_sq = np.einsum("ij,ij->i", mu_matrix, solved_mu)
    alpha = patch_scalar_var / 2.0
    mu_ones = mu_matrix @ model.half_cov_solve_ones
    correction = alpha * (mu_ones * mu_ones) / (1.0 + (alpha * model.half_cov_ones_quad))
    distance_sq = base_distance_sq - correction
    tiny_negative = (distance_sq < 0) & (np.abs(distance_sq) < 1e-12)
    if np.any(tiny_negative):
        distance_sq = distance_sq.copy()
        distance_sq[tiny_negative] = 0.0
    distance_patch = np.sqrt(distance_sq)

    if np.any(np.isfinite(distance_patch)):
        mean_distance = float(np.nanmean(distance_patch))
    else:
        mean_distance = float("nan")
    return mean_distance, distance_patch.reshape(patch_grid_shape, order="F")


def _score_single_uint8_rgb(
    image: np.ndarray,
    *,
    fogfree: ReferenceModel,
    foggy: ReferenceModel,
    return_map: bool,
) -> float | tuple[float, np.ndarray]:
    feat, patch_grid_shape = _extract_features(image)
    df, df_map = _distance_to_model(feat, patch_grid_shape, fogfree)
    dff, dff_map = _distance_to_model(feat, patch_grid_shape, foggy)
    score = df / (dff + 1.0)
    density_map = df_map / (dff_map + 1.0)
    if return_map:
        return score, density_map
    return score


def _normalize_workers(workers: int | None) -> int:
    if workers is None:
        return 1
    workers_int = int(workers)
    if workers_int < 1:
        raise ValueError("workers must be >= 1")
    return workers_int


def _progress_iter(iterable: Iterable[object], *, total: int, enabled: bool, desc: str) -> Iterable[object]:
    if not enabled:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def _evaluate_many_images(
    images: Sequence[np.ndarray],
    *,
    model_dir: str | Path | None,
    return_map: bool,
    workers: int,
    progress: bool,
    desc: str,
) -> list[float | tuple[float, np.ndarray]]:
    fogfree, foggy = _load_models(model_dir)
    workers = _normalize_workers(workers)
    if workers == 1 or len(images) <= 1:
        iterator = _progress_iter(images, total=len(images), enabled=progress and len(images) > 1, desc=desc)
        return [
            _score_single_uint8_rgb(image, fogfree=fogfree, foggy=foggy, return_map=return_map)
            for image in iterator
        ]

    results: list[float | tuple[float, np.ndarray] | None] = [None] * len(images)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_score_single_uint8_rgb, image, fogfree=fogfree, foggy=foggy, return_map=return_map): idx
            for idx, image in enumerate(images)
        }
        iterator = _progress_iter(as_completed(futures), total=len(futures), enabled=progress, desc=desc)
        for future in iterator:
            idx = futures[future]
            results[idx] = future.result()

    return [result for result in results if result is not None]


def _evaluate_many_paths(
    paths: Sequence[Path],
    *,
    model_dir: str | Path | None,
    return_map: bool,
    workers: int,
    progress: bool,
    desc: str,
) -> list[float | tuple[float, np.ndarray]]:
    fogfree, foggy = _load_models(model_dir)
    workers = _normalize_workers(workers)

    def _score_path(path: Path) -> float | tuple[float, np.ndarray]:
        image = load_image(path)
        return _score_single_uint8_rgb(image, fogfree=fogfree, foggy=foggy, return_map=return_map)

    if workers == 1 or len(paths) <= 1:
        iterator = _progress_iter(paths, total=len(paths), enabled=progress and len(paths) > 1, desc=desc)
        return [_score_path(path) for path in iterator]

    results: list[float | tuple[float, np.ndarray] | None] = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_score_path, path): idx for idx, path in enumerate(paths)}
        iterator = _progress_iter(as_completed(futures), total=len(futures), enabled=progress, desc=desc)
        for future in iterator:
            idx = futures[future]
            results[idx] = future.result()

    return [result for result in results if result is not None]


def fade_array(
    image: object,
    *,
    model_dir: str | Path | None = None,
    return_map: bool = False,
    workers: int = 1,
    progress: bool = False,
) -> float | np.ndarray | tuple[float, np.ndarray] | tuple[np.ndarray, np.ndarray]:
    images, is_batch = _normalize_image_batch(image)
    results = _evaluate_many_images(
        images,
        model_dir=model_dir,
        return_map=return_map,
        workers=workers,
        progress=progress,
        desc="pyfade",
    )

    if not is_batch:
        single_result = results[0]
        if return_map:
            score, density_map = single_result
            return float(score), density_map
        return float(single_result)

    if return_map:
        scores = np.asarray([result[0] for result in results], dtype=np.float64)
        density_maps = np.stack([result[1] for result in results], axis=0)
        return scores, density_maps

    return np.asarray(results, dtype=np.float64)


def fade_image(
    path: str | Path,
    *,
    model_dir: str | Path | None = None,
    return_map: bool = False,
) -> float | tuple[float, np.ndarray]:
    image = load_image(path)
    return fade_array(image, model_dir=model_dir, return_map=return_map)


def _folder_result_from_items(
    image_paths: Sequence[Path],
    items: Sequence[float | tuple[float, np.ndarray]],
    *,
    return_map: bool,
) -> FolderResult:
    score_items: dict[str, float] = {}
    map_items: dict[str, np.ndarray] | None = {} if return_map else None
    for path, item in zip(image_paths, items, strict=True):
        if return_map:
            score, density_map = item
            score_items[path.name] = float(score)
            map_items[path.name] = density_map
        else:
            score_items[path.name] = float(item)

    values = np.asarray(list(score_items.values()), dtype=np.float64)
    return FolderResult(
        scores=score_items,
        density_maps=map_items,
        mean_score=float(values.mean()),
        min_score=float(values.min()),
        max_score=float(values.max()),
    )


def fade_folder(
    folder: str | Path,
    *,
    model_dir: str | Path | None = None,
    return_map: bool = False,
    workers: int = 1,
    progress: bool = False,
    extensions: Iterable[str] = IMAGE_EXTENSIONS,
) -> FolderResult:
    folder = Path(folder)
    suffixes = {ext.lower() for ext in extensions}
    image_paths = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )
    if not image_paths:
        raise ValueError(f"no supported images found in {folder}")

    items = _evaluate_many_paths(
        image_paths,
        model_dir=model_dir,
        return_map=return_map,
        workers=workers,
        progress=progress,
        desc="pyfade",
    )
    return _folder_result_from_items(image_paths, items, return_map=return_map)


def fade(
    input_data: object,
    *,
    model_dir: str | Path | None = None,
    return_map: bool = False,
    workers: int = 1,
    progress: bool = False,
    extensions: Iterable[str] = IMAGE_EXTENSIONS,
) -> float | np.ndarray | tuple[float, np.ndarray] | tuple[np.ndarray, np.ndarray] | FolderResult:
    if isinstance(input_data, (str, Path)):
        path = Path(input_data)
        if path.suffix.lower() == ".npy" and path.is_file():
            return fade_array(np.load(path, allow_pickle=False), model_dir=model_dir, return_map=return_map, workers=workers, progress=progress)
        if path.is_dir():
            return fade_folder(
                path,
                model_dir=model_dir,
                return_map=return_map,
                workers=workers,
                progress=progress,
                extensions=extensions,
            )
        return fade_image(path, model_dir=model_dir, return_map=return_map)

    return fade_array(input_data, model_dir=model_dir, return_map=return_map, workers=workers, progress=progress)
