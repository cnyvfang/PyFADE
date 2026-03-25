# PyFADE

FADE (Fog Aware Density Evaluator) is a method for estimating haze density in an image, and it is often used as an indicator for evaluating the dehazing performance of methods in image dehazing.

PyFADE is an unofficial Python implementation of FADE. The implementation is designed to closely match the MATLAB results, while providing higher efficiency and stronger plug-and-play compatibility with existing experimental code.



## ⚙️ Installation

Install from PyPI:

```bash
pip install fade-python
```

The published distribution name is `fade-python`, while the import package and
CLI command remain `pyfade`.

Install from a local checkout:

```bash
git clone https://github.com/cnyvfang/PyFADE.git
cd PyFADE
# Option 1:
pip install .              # standard install
# Option 2:
pip install -e ".[dev]"    # editable install for development
# Option 3:
pip install ".[tensor]"    # with PyTorch input support
```

The base install already includes the current runtime dependencies used by the
latest implementation, including `numba` and `turbojpeg`.

## 🔨 Usage

```python
from pyfade import fade
```

### Accepted inputs

- Image path: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`
- Folder path: a flat, non-recursive directory of supported image files
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

np.save("sample.npy", image)
score_from_npy = fade("sample.npy")

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

## 🚀 Parallelism and Progress

- `workers` controls image-level concurrency
- `progress=True` enables a `tqdm` progress bar for folder and batch evaluation

Example:

```python
result = fade("/path/to/folder", workers=8, progress=True)
scores = fade(batch_tensor, workers=4, progress=True)
```

## 👨‍💻 CLI

```bash
pyfade /path/to/image.png
pyfade /path/to/folder --workers 4 --progress
pyfade /path/to/array.npy --workers 4
pyfade /path/to/image.png --return-map
```

## 📖 Reference

To use PyFADE to validate your methods, please cite the original FADE paper and note the use of the PyFADE implementation.

```bibtex
@article{choi2015referenceless,
  title={Referenceless prediction of perceptual fog density and perceptual image defogging},
  author={Choi, Lark Kwon and You, Jaehee and Bovik, Alan Conrad},
  journal={IEEE Transactions on Image Processing},
  volume={24},
  number={11},
  pages={3888--3901},
  year={2015},
  publisher={IEEE}
}
```

If this implementation is useful in your work, we also hope you will consider citing our
related dehazing work.

```bibtex
@article{fang2024real,
  title={Real-world image dehazing with coherence-based pseudo labeling and cooperative unfolding network},
  author={Fang, Chengyu and He, Chunming and Xiao, Fengyang and Zhang, Yulun and Tang, Longxiang and Zhang, Yuelin and Li, Kai and Li, Xiu},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={97859--97883},
  year={2024}
}
```

## 📈 Benchmark and Precision Summary

| Runtime       | Workers | Total Time (s) | Throughput (img/s) | Runtime       | Workers | Total Time (s) | Throughput (img/s) |
|---------------|--------:|---------------:|-------------------:|---------------|--------:|---------------:|-------------------:|
| MATLAB        |       1 |         809.04 |             5.3909 | Ours          |       1 |         393.47 |            10.9844 |
| MATLAB        |       4 |         286.69 |            15.5351 | Ours          |       4 |         142.04 |            30.4283 |
| MATLAB        |       8 |         216.34 |            20.7242 | Ours          |       8 |         111.89 |            38.6282 |

| Dataset            | Runtime  |            Mean Score | Runtime |            Mean Score | Mean Diff | Max Diff |
|--------------------|----------|----------------------:|---------|----------------------:|----------:|---------:|
| RTTS Hazy Images   | MATLAB   |        2.514381896977 | Ours |        2.514345050406 | -3.685e-05 | 4.163e-03 |
| RTTS PRISM Dehazed | MATLAB   |        0.470454676887 | Ours |        0.470454676888 | 1.002e-12 | 1.346e-08 |
| RTTS CORUN Dehazed | MATLAB   |        0.824629833124 | Ours |        0.824629832977 | -1.468e-10 | 6.253e-07 |
| RTTS RIDCP Dehazed | MATLAB   |        0.913704350828 | Ours |        0.913702944397 | -1.406e-06 | 2.952e-03 |
| Mini-ImageNet Testset | MATLAB   |        0.484498390241 | Ours |        0.484496958259 | -1.432e-06 | 2.358e-03 |

Precision statistics can also depend slightly on image save-time preprocessing and decoder behavior, especially when images are re-saved, normalized, or converted across formats before evaluation.



## 🔍 MATLAB Version vs. Current PyFADE

- Both versions implement the same FADE algorithm and use the same original
  reference models.
- MATLAB relies on built-in operators such as `rgb2gray`, `rgb2hsv`,`fspecial`, `imfilter`, `im2col`, `entropy`, `nanvar/std`, and `mrdivide`.
- PyFADE reproduces those semantics explicitly, including patch order, border handling, convolution alignment, variance rules, and entropy behavior.
- PyFADE provides a broader interface surface than the original MATLAB function: it supports folder paths, image paths, `.npy` files, NumPy arrays, and tensor-like inputs.
- PyFADE also preserves higher-precision image I/O paths such as 16-bit PNG input and normalized float array/tensor input.
- Relative to the original MATLAB code, the current package adds:
  - image-level multi-threaded execution
  - vectorized rank-1 update for model-distance computation
  - cached MSCN and CE kernels
  - faster 1D convolution path for `1xN` and `Nx1` kernels
  - vectorized packed-`bincount` entropy evaluation
  - `numba`-accelerated hotspots for HSV conversion, entropy, and variance
  - faster JPEG decoding through `turbojpeg`
- On the original paired `RTTS` workload benchmark in this repository, the current PyFADE implementation is faster than MATLAB at `1`, `4`, and `8` image-level workers.
