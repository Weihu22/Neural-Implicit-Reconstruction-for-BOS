
## License

This project is licensed under the **PolyForm Noncommercial License 1.0.0**.

The code may be used, modified, and distributed for noncommercial purposes,
including academic, educational, and research use.

**Commercial use is not permitted under this license.**
For commercial licensing inquiries, please contact the authors.

See the [LICENSE](LICENSE) file for details.


## Citation

If you use this code in your research, please cite our paper:

**X. Lu, W. Hu, Z. Liao, Z. Wang, Y. Zhang, J. Li,  
Neural refractive index primitives for flame field reconstruction using background-oriented schlieren,  
Combustion and Flame 290 (2026) 115082.**

https://doi.org/10.1016/j.combustflame.2026.115082

```bibtex
@article{lu2026neural,
  title   = {Neural refractive index primitives for flame field reconstruction using background-oriented schlieren},
  author  = {Lu, Xinyi and Hu, Wei and Liao, Zizhou and Wang, Zheng and Zhang, Yue and Li, Jingxuan},
  journal = {Combustion and Flame},
  volume  = {290},
  pages   = {115082},
  year    = {2026},
  doi     = {10.1016/j.combustflame.2026.115082}
}
```

## Reference Implementations

To facilitate reproducibility, this code was developed based in part on the following open-source implementations. We sincerely thank the authors for making their work publicly available.

**[1] TIGRE**

Biguri, A., Dosanjh, M., Hancock, S., et al.  
*TIGRE: A MATLAB-GPU toolbox for CBCT image reconstruction.*  
Biomedical Physics & Engineering Express, 2016, 2(5): 055010.  
https://github.com/CERN/TIGRE

**[2] torch-ngp**

Tang, J., Chen, X., Wang, J., et al.  
*Compressible-composable NeRF via rank-residual decomposition.*  
Advances in Neural Information Processing Systems, 2022, 35: 14798–14809.  
https://github.com/ashawkey/torch-ngp

## Important Notice

### Limitation of the Current CGLS Implementation

We have identified a mismatch between the forward and backward operators in the current CGLS-based reconstruction code. Specifically, `Ax` uses the **Siddon ray-tracing and interpolation operators**, whereas `A^T b` uses a **pseudo-matched backprojection weighting scheme**.

Although the mismatch is typically below approximately **1%**, it can still affect CGLS convergence, potentially causing **non-convergence** and **artificial stripe-like artifacts near the reconstruction boundaries**.

Based on our current tests, this implementation appears to be suitable mainly for **strictly radial and sufficiently dense multi-view configurations**. We currently do **not recommend it for limited-view or non-radial BOST configurations**.

This issue has been addressed in our new **IRN-TV-CGLS** implementation. The updated BOST code, supporting **limited-view and non-radial view distributions**, will be released after the associated paper is published.

**Note:** The **Neural Implicit Reconstruction for BOS** implementation does **not** suffer from this forward/backprojection mismatch issue and can still be used normally.

## Implementation and Usage Instructions

At present, the code does not support CPU-only execution. The ray integration and its backward-propagation operations were implemented exclusively as custom CUDA code, so an NVIDIA GPU with CUDA support is required to run the current version.

### 1. MATLAB Initialization and Compilation

For the first run:

1. Run `step1_InitBOSLAB.m`
2. Run `step2_Compile.m`

After successful compilation:

Run `step1_InitBOSLAB.m`, then directly run:

`step3_generate_phantom1_synthetic_data.m`

in the demo folder.

Copy the dataset from:

```text
MATLAB/Test_data/Phantom 1
```

to:

```text
PYTHON/NIR-BOS/data
```

Set up the Python environment using:

```text
environment.yml
```

### 2. Configuration Before Running `main_BOS.py`

Set `sys.argv` based on the outputs of:

```text
matlab/step3_generate_phantom1_synthetic_data.m
```

Example configuration:

```text
--scale: 0.00054421
--ROIsize: [0.95237, 2, 0.95237]
--ROInum: [140, 294, 140]
--ROIvoxelsize: 0.013605
--valbound: [-1, 2.9339]
```

Due to uncertainties in practical usage, the `valbound` range can be appropriately relaxed.

Set the output directory for saving results.

Example:

```text
results/phantom1/freencode_disc_mask
```

### 3. Mask Configuration

If `--maskflag` is removed from `sys.argv`, the 3D mask is disabled, while the 2D mask-based ray sampling strategy remains enabled.

To disable the 2D mask-based ray sampling strategy, remove `UVROIs` and `masks` from the input of the `collate` function in `provider.py`:

```python
rays = get_rays(
    poses,
    self.intrinsics,
    self.H,
    self.W,
    self.num_rays,
    error_map,
    self.opt.patch_size
)
```

### 4. Encoding Configuration

Select the encoding method in NeRF network initialization:

```python
model = NeRFNetwork(
    encoding="Hash", #or Fourier
    bound=opt.bound,
    cuda_ray=opt.cuda_ray,
    density_scale=1,
    min_near=opt.min_near,
    density_thresh=opt.density_thresh,
    mask3Ddata=mask3Ddata,
    ROIsize=opt.ROIsize,
    ROInum=opt.ROInum,
    ROIvoxelsize=opt.ROIvoxelsize,
    valbound=opt.valbound
)
```

More detailed encoding parameters can be configured in `encoding.py` (`get_encoder` function).

`multires` controls the number of frequency encoding levels, and `log2_hashmap_size` defines the hash table size for hash encoding.

### 5. Evaluation and Export

The `eval_interval` parameter in the `Trainer` controls the evaluation frequency.

Every `eval_interval` steps/epochs, the current predicted refractive index (`sigma0`) and the automatically differentiated refractive index gradients (`dsigma_dxyz_auto0`) are exported as `sigmas0_epochMATLAB` (`.mat`) files named as:

```text
sigmas0_epoch number
```

### 6. Gradient Computation in the Loss Function

The gradient computation in the loss function is controlled in `utils.py` (`train_step` function).

It is defined as:

```python
loss = 1 * loss.mean() + 0 * loss_auto.mean()
```

where `loss` represents the discrete gradient loss, and `loss_auto` represents the gradient loss computed via automatic differentiation.

### 7. CUDA Compilation

On the first run of `main_BOS.py`, CUDA kernels will be compiled (JIT compilation), which may take some time.

This is a normal and expected step.

If any issues occur and cannot be resolved, please contact us for support.

### 8. Geometry Verification

After successful compilation, two figures will be displayed:

The first shows the test set camera and flow field geometry, and the second shows the validation set geometry.

If the result is inconsistent with the expected setup, it is likely due to an issue in the MATLAB geometry definition and should be checked.

Close the figure to proceed to the next step.

After training is completed, an additional figure will be displayed, showing the spatial configuration of the camera and flow field for the test set.

Close the figure to proceed to the next step.

# It should be noted that the encoding module is not implemented with CUDA acceleration.
# As a result, hash-based encoding is significantly slower than frequency-based encoding.
# A CUDA-accelerated version of the encoding will be released in the future.

