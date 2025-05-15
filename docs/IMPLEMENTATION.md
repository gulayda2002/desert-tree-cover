# Implementation Notes and Profiling

This document details key challenges encountered during development, profiling results, and solutions applied.

## 1. Profiling Environment
- **Machine:** MacBook Pro M1 2020, 8-core CPU, 8-core GPU (MPS), 16 GB unified RAM
- **Software Versions:** Python 3.10, PyTorch 1.13 with MPS support, rasterio 1.3.6

## 2. Profiling Tools and Methods
- **CPU Profiling:** Python `cProfile` with snakeviz for visualization
- **GPU Profiling:** PyTorch `torch.utils.bottleneck` and MPS activity logs

## 3. Processor Hotspots (Preprocessing)
| Function                 | Time % | Optimizations Applied                       |
|--------------------------|-------:|---------------------------------------------|
| rasterio.open            |   30 % | Reuse open file handles via global cache    |
| src.read (windowed)      |   45 % | Adjust tile size to 512×512 for balanced I/O|
| array normalization      |   10 % | In-place division (`arr /= max_val`)        |
| mask thresholding        |    5 % | Vectorized NumPy operations                 |
| file write (`np.savez`)  |   10 % | Compressed `.npz` instead of `.npy`         |

## 4. GPU Bottlenecks (Training)
- Initial training with batch_size=16 caused MPS kernel stalls and OOM errors.
- Reducing `batch_size` to 6 eliminated memory errors while fully utilizing GPU.
- `torch.utils.data.DataLoader` with `pin_memory=True` improved host-to-device transfer by 15 %.

## 5. Solutions to Key Challenges

### 5.1 File Descriptor Limits
- **Issue:** Workers hitting `Too many open files` due to concurrent rasterio file openings.
- **Solution:** Implemented a singleton file-handle manager that caches rasterio `DatasetReader` objects per-band and per-scene.

### 5.2 MPS Stability
- **Issue:** PyTorch MPS backend crashes on large convolution kernels.
- **Solution:** Upgraded to latest PyTorch release and split large batches into smaller sub-batches within training loop.

### 5.3 Load Imbalance
- **Issue:** Some tiles take longer due to edge conditions (smaller tile sizes).
- **Solution:** Switched from `Pool.map` to `Pool.imap_unordered` for dynamic task scheduling; faster processes pick up remaining tasks immediately.

## 6. Summary of Improvements
- **End-to-End Preprocessing Time:** Reduced from ~120 s (serial) to ~18 s (parallel, 6 workers)
- **Training Epoch Time:** Reduced from ~45 min (CPU-only) to ~7 min (MPS with batch_size=6)

