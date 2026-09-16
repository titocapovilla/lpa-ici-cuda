# GPU LPA ICI 2D

This repository contains the code for the implementation and CUDA acceleration of the LPA-ICI 2D algorithm (Local Polynomial Approximation - Intersection of Confidence Intervals) with anisotropic kernels.

## Compilation
The code is compiled via Cmake.
Make sure you have CMake and a C++17 compiler installed.
If nvcc is also present, the .cu files are compiled as well. Without it only the two CPU versions are built.

the command
```bash
./build.sh
```
executes the Cmake and builds the executables inside /build

## Run
From the repository root, run

```bash
# C++ version
./build/v0_0_lpa_ici_2D_naive

# CUDA version
./build/v0_0_lpa_ici_2D_naive_gpu
```
or any of the other compiled versions.



# ALGORITHM
The algorithm imports the image via `stbi_loadf()` with `desired_channels = 1`, so stb converts it to gray scale. The conversion is computed as follows

```math
\text{Gray} = \frac{77R + 150G + 29B}{256}
```

It then adds noise to the image, Gaussian with $`\sigma = 20/255`$. The seed is fixed, so every version filters exactly the same noisy image and the results are comparable.

The kernels are read from a text file: 16 directions by 7 scales, so 112 anisotropic kernels. Each one is a wedge pointing in its direction. Every direction is cropped to the sizes of direction 0, which are 1, 3, 7, 15, 31, 47 and 63 pixels square. The variance of each kernel is precomputed as

```math
\sigma^2 \sum_{i=1}^{n} w_i^2
```

For each direction the algorithm convolves the image with all the kernels from smaller to bigger. After every convolution it builds the interval

```math
\left[ \text{estimate} - \gamma \sqrt{\text{variance}}, \, \text{estimate} + \gamma \sqrt{\text{variance}} \right]
```

with $`\gamma = 2`$, and intersects it with the intervals from all the smaller kernels.
While the intersection is not empty, the algorithm accepts the new estimate.
When it becomes empty, the last estimate with overlapping intervals is kept, together with its variance. This is the ICI rule, and it selects the largest kernel that is still consistent with every smaller one.

Because the stopping point depends on the noise and on the image content, the amount of convolutions per pixel varies. A pixel in a flat area reaches a large kernel. A pixel close to an edge stops early.

The 16 directional estimates are combined into the final image as a weighted average with weight $`1/\text{variance}`$, so a direction that reached a larger kernel counts more.

Finally the PSNR is computed against the clean image, and the result is written as .hdr, which is the only format stbi writes in floating point.

![Clean, noisy and denoised](data/barbara_clean_noisy_denoised.png)

## Files and Implementations

##### v0_0_lpa_ici_2D_naive.cpp
Standard algorithm implementation, follows the python/matlab original.
Performs the convolutions via convolve2D() on all the image's pixels.
Processes the intervals, if the interval is still valid updates the estimate.
nb: all convolution sizes for a pixel are computed, even those discarded later.

##### v0_0_lpa_ici_2D_naive.cu
First GPU version. Parallelizes ```convolve2D_gpu()``` across pixels, one thread per pixel.
The ICI test and the accumulation stay on the host, so the estimate of every scale is copied back.
The filter of each scale is also allocated and freed inside the loop.

##### v0_1_lpa_ici_2D_naive_single_kernel_alloc.cu
Same as above, with the whole filter data allocated once before the loop.


##### v1_0_lpa_ici_2D_fused.cpp
Improves performance by calculating the convolution only for pixels whose confidence intervals have not already lost overlap.

##### v1_0_lpa_ici_2D_fused.cu
GPU implementation of the fused version. Performs the convolution and interval checking on device.
Only the directional estimate is copied back to host.

##### v1_1_lpa_ici_2D_fused_aggregation.cu
Moves directional and final accumulation loops to device to avoid copy and iteration over the directional image estimate for 16 directions.

##### v1_2_lpa_ici_2D_fused_aggregation_constant.cu
Moves kernel weights to __constant__ memory for faster reading and idle time reduction.
The kernel is transferred once per direction, as MAX_DIR_KERNEL_ELEMENTS floats (8192 * 4bytes = 32,768bytes < 64 KB)
Currently the max size of kernels for one direction is 29,692 bytes < 32,768 (when normalized)
Two directions could technically fit but enough room is left for different kernel sizes that might be used.


##### v2_0_lpa_ici_2D_fused_streams.cu
Starts from v1_1_lpa_ici_2D_fused_aggregation (constant memory would be overwritten by streams).
Introduces streams for the directions.
Allocates num_dir * img_size vectors for the directional estimates and parameters.
Computes each direction in a separate stream
Performs accumulation with atomicAdd().
Synchronizes streams and calls normalization kernel
No improvement is seen given GPU is already saturated.

##### v3_0_lpa_ici_2D_fused_ac_sparse.cu
Stores only the kernel values that are not zero.
The non zero values of every kernel row are contiguous, so a row is described by row offset, start column and run length.
Seven __constant__ arrays hold all 112 filters: c_weights with the non zero values, then c_run_row, c_run_col and c_run_len describing each row, and c_kernel_first_run, c_kernel_num_runs and c_kernel_first_weight saying which rows belong to which filter.
All 112 filters together take 38,000 bytes, so they all fit in constant memory and are uploaded once instead of one direction at a time.

##### v3_0_2_lpa_ici_2D_fused_ac_sparse_coarsened.cu
Starts from sparse. Each thread computes 4 pixels instead of 1. The 4 pixels are selected far apart, so threads next to each other read memory next to each other.
The reduction in the number of blocks makes it slower.

##### v3_1_lpa_ici_2D_fused_ac_sparse_coalesced.cu
Block changed from 16x16 to 32x8 (same number of threads).
A warp now covers 32 pixels of one image row instead of 16 of one row and 16 of the next, and a read does one memory access instead of two.
Time change is minimal.

##### v4_0_lpa_ici_2D_sparse_scale_fused.cu
All 7 scales of a direction run in one kernel. The confidence interval of a pixel dpes not need to be read from global memory.
d_lower_bounds, d_upper_bounds and initializeBoundsKernel are removed. Variance and threshold move to __constant__.

##### v4_1_lpa_ici_2D_sparse_shared.cu
Introduces tiling by moving patches of the image to __shared__ memory for faster loading of those values during convolutions.


## Results

512x512 image, 16 directions, 7 scales, NVIDIA T4 on Google Colab.
Times are the median of five runs of the directional loop.
All versions produce the same image, PSNR 25.0736.

| Version | Time | Speedup |
|---|---|---|
| v0_0_lpa_ici_2D_naive.cpp | 57.4393 s | 1x |
| v1_0_lpa_ici_2D_fused.cpp | 28.2776 s | 2x |
| v0_0_lpa_ici_2D_naive.cu | 0.263951 s | 218x |
| v0_1_lpa_ici_2D_naive_single_kernel_alloc.cu | 0.247642 s | 232x |
| v1_0_lpa_ici_2D_fused.cu | 0.112544 s | 510x |
| v1_1_lpa_ici_2D_fused_aggregation.cu | 0.103192 s | 557x |
| v1_2_lpa_ici_2D_fused_aggregation_constant.cu | 0.080458 s | 714x |
| v2_0_lpa_ici_2D_fused_streams.cu | 0.101345 s | 567x |
| v3_0_lpa_ici_2D_fused_ac_sparse.cu | 0.009138 s | 6286x |
| v3_0_2_lpa_ici_2D_fused_ac_sparse_coarsened.cu | 0.010851 s | 5294x |
| v3_1_lpa_ici_2D_fused_ac_sparse_coalesced.cu | 0.009090 s | 6319x |
| v4_0_lpa_ici_2D_sparse_scale_fused.cu | 0.007477 s | 7682x |
| v4_1_lpa_ici_2D_sparse_shared.cu | 0.005800 s | 9903x |

The two CPU versions are single runs.




## References

The LPA-ICI algorithm and its original MATLAB implementation come from the LASIP toolkit (Local Approximations in Signal and Image Processing), Tampere University:
https://webpages.tuni.fi/lasip/2D/

The directional kernels in `kernels/` were generated with a modified version of the demo_CreateLPAKernels.m script.

Python notebooks by Prof. Giacomo Boracchi (Politecnico di Milano) were also used as a reference.
They are dowloadable at https://boracchi.faculty.polimi.it/teaching/MMMIP.htm

The images are imported and processed via stb_image.
https://github.com/nothings/stb

The code was based off of the examples in
Programming Massively Parallel processors 5th edition by Wen-mei W. Hwu, David B. Kirk, and Izzat H. El Hajj