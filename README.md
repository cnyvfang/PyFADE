# PyFADE

`PyFADE` is a MATLAB-aligned Python package for Fog Aware Density Evaluator (FADE) fog density estimation.
It packages the optimized Python implementation together with the original reference model parameters. The implementation is designed to match the MATLAB FADE code path closely.


## Installation

Installed from PyPI with:

```bash
pip install fade-python
```

Install from a local checkout:

```bash
git clone https://github.com/cnyvfang/PyFADE
cd pyfade
# Option 1:
pip install .              # standard install
# Option 2:
pip install -e ".[dev]"    # editable install for development
# Option 3:
pip install ".[tensor]"    # with PyTorch input support
```

The published distribution name is
`fade-python`, while the import package and CLI command remain `pyfade`.

## Usage

```python
from pyfade import fade
```

### Accepted inputs

- Image path: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`
- Folder path: a flat directory of supported image files
- `.npy` path: loaded and processed as a NumPy array
- NumPy arrays:
  `(H, W, 3)`, `(3, H, W)`, `(B, H, W, 3)`, or `(B, 3, H, W)`
- PyTorch tensors:
  `(H, W, 3)`, `(3, H, W)`, `(B, H, W, 3)`, or `(B, 3, H, W)`

### Input conventions

- `uint8` inputs are interpreted as `0..255` images
- Floating-point inputs in `[0, 1]` are treated as normalized images and scaled to `0..255`
- Other floating-point inputs are treated as already being in `0..255`
- Tensor inputs are copied to CPU and processed through the same MATLAB-aligned NumPy path

### Return values

- Single image input:
  `float`, or `(score, density_map)` when `return_map=True`
- Batch array/tensor input:
  `scores` with shape `(B,)`, or `(scores, density_maps)` when `return_map=True`
- Folder input:
  `FolderResult` with `scores`, optional `density_maps`, and `mean_score`, `min_score`, `max_score`

Density maps always follow MATLAB-style patch trimming, so their shape is based
on the trimmed `(H // 8, W // 8)` patch grid.

### Examples

Single image path:

```python
score = fade("/path/to/image.png")
score, density_map = fade("/path/to/image.png", return_map=True)
```

Folder path:

```python
result = fade("/path/to/folder", workers=4, progress=True)

print(result.mean_score)
print(result.scores["example.png"])
```

NumPy array:

```python
import numpy as np

image = np.zeros((256, 256, 3), dtype=np.uint8)
score = fade(image)

batch = np.random.randint(0, 256, size=(8, 3, 256, 256), dtype=np.uint8)
scores, density_maps = fade(batch, workers=4, progress=True, return_map=True)
```

PyTorch tensor:

```python
import torch

batch = torch.randint(0, 256, (8, 3, 256, 256), dtype=torch.uint8)
scores = fade(batch, workers=4, progress=True)

normalized = batch.to(torch.float32) / 255.0
scores, density_maps = fade(normalized, workers=4, progress=True, return_map=True)
```

## Parallelism and Progress

- `workers` controls image-level concurrency
- `progress=True` enables a `tqdm` progress bar for folder and batch evaluation

Example:

```python
result = fade("/path/to/folder", workers=8, progress=True)
scores = fade(batch_tensor, workers=4, progress=True)
```

## CLI

`PyFADE` installs a `pyfade` command and is imported with `import pyfade`:

```bash
pyfade /path/to/image.png
pyfade /path/to/folder --workers 4 --progress
pyfade /path/to/array.npy --workers 4
pyfade /path/to/image.png --return-map
```

## Benchmark and Precision Summary

The table below uses the `RTTS` dataset and the dehazing results from `PRISM` with `4322` images.

| Runtime | Workers | Total Time (s) | Throughput (img/s) | Mean score | Mean score diff vs. MATLAB  | Max score abs diff |
| --- |--------:|---------------:| ---: | ---: | ---: | ---: |
| MATLAB |       1 |         809.04 | 5.3909 | 0.470454676887910 | 0.000e+00  | 0.000e+00 |
| MATLAB |       4 |         286.69 | 15.5351 | 0.470454676887910 | N/A | N/A  |
| MATLAB |       8 |         216.34 | 20.7242 | 0.470454676887910 | N/A | N/A  |
| Python pre-optimization |       1 |        1785.01 | 2.4226 | 0.470454676908741 | 2.083e-11 | 1.346e-08  |
| Python optimized |       1 |         546.15 | 7.9248 | 0.470454676908513 | 2.060e-11 | 1.346e-08  |
| Python optimized |       4 |         169.47 | 25.6466 | 0.470454676908513 | N/A | N/A  |
| Python optimized |       8  |         131.98 | 32.9707 | 0.470454676908513 | N/A | N/A  |

The initial Python port was already highly consistent with MATLAB, but slower. The optimized Python version preserves MATLAB-level numerical agreement while substantially improving single-thread performance. On this machine (Apple M2 Pro, 32GB RAM), the optimized Python version is faster than MATLAB with 1, 4, and 8 image-level workers.

## MATLAB vs. Initial Python Port

- Both versions implement the same FADE algorithm and use the same original
  reference models.
- MATLAB relies on built-in operators such as `rgb2gray`, `rgb2hsv`,
  `fspecial`, `imfilter`, `im2col`, `entropy`, `nanvar/std`, and `mrdivide`.
- The Python port reproduces those semantics explicitly, including patch order,
  border handling, convolution alignment, variance rules, and entropy behavior.
- The Python package provides a broader interface surface than the original
  MATLAB function: it supports folder paths, image paths, `.npy` files, NumPy
  arrays, and tensor-like inputs.
- Precision of the initial Python version was already close to MATLAB:
  `Score MAE = 2.469e-11`, `Score max abs diff = 1.346e-08`,
  `Map global MAE = 2.370e-09`.
- Performance of the initial Python single-thread version was much worse than
  MATLAB single-thread: `1784.01s` vs. `801.72s`.

## Optimized Python vs. Initial Python

- The score definition, feature definition, bundled model parameters, and
  MATLAB-alignment rules are unchanged.
- The optimized version improves the implementation, not the algorithm:
  - vectorized rank-1 update for model-distance computation
  - cached MSCN and CE kernels
  - faster 1D convolution path for `1xN` and `Nx1` kernels
  - vectorized packed-`bincount` entropy evaluation
- Precision is effectively unchanged:
  - `Score MAE`: `2.469e-11 -> 2.488e-11`
  - `Score RMSE`: `2.723e-10 -> 2.722e-10`
  - `Score max abs diff`: unchanged at `1.346e-08`
  - `Map global MAE`: unchanged at `2.370e-09`
- Single-thread performance improved from `1784.01s` to `545.38s`,
  about `3.27x` faster than the initial Python version.

