# Copyright (c) 2025 Rohit Jena. All rights reserved.
#
# This file is part of FireANTs, distributed under the terms of
# the FireANTs License version 1.0. A copy of the license can be found
# in the LICENSE file at the root of this repository.
#
# IMPORTANT: This code is part of FireANTs and its use, reproduction, or
# distribution must comply with the full license terms, including:
# - Maintaining all copyright notices and bibliography references
# - Using only approved (re)-distribution channels
# - Proper attribution in derivative works
#
# For full license details, see: https://github.com/rohitrango/FireANTs/blob/main/LICENSE


"""
Utilities for loading and processing timeseries (4D) medical images.
"""

from typing import List, Optional, Tuple
from pathlib import Path
import SimpleITK as sitk
import torch
import numpy as np
import logging

from fireants.io.image import Image

logger = logging.getLogger(__name__)


def load_4d_nifti(file_path: str, dtype: torch.dtype = torch.float32,
                  device: str = 'cuda:0') -> List[Image]:
    """Load a 4D NIfTI file and split it into a list of 3D Image objects.

    Args:
        file_path: Path to 4D NIfTI file (e.g., 'timeseries.nii.gz')
        dtype: PyTorch dtype for the image data
        device: Device to load images onto ('cuda:0', 'cpu', etc.)

    Returns:
        List of Image objects, one per timepoint

    Example:
        >>> frames = load_4d_nifti('bold.nii.gz')
        >>> print(f"Loaded {len(frames)} timepoints")
        >>> print(f"Frame shape: {frames[0].shape}")
    """
    logger.info(f"Loading 4D NIfTI file: {file_path}")

    # Load with SimpleITK
    itk_image_4d = sitk.ReadImage(str(file_path))

    # Check dimensionality
    ndim = itk_image_4d.GetDimension()
    if ndim != 4:
        raise ValueError(f"Expected 4D image, got {ndim}D image")

    # Get size along time dimension (last dimension in SimpleITK)
    size_4d = itk_image_4d.GetSize()
    num_frames = size_4d[3]

    logger.info(f"Found {num_frames} timepoints with shape {size_4d[:3]}")

    # Extract each 3D frame
    frames = []
    extractor = sitk.Extract()

    for t in range(num_frames):
        # Set extraction region: all spatial dimensions, single timepoint
        extraction_region = list(size_4d)
        extraction_region[3] = 0  # Collapse time dimension
        extraction_index = [0, 0, 0, t]  # Start at timepoint t

        extractor.SetSize(extraction_region)
        extractor.SetIndex(extraction_index)

        # Extract 3D volume
        itk_image_3d = extractor.Execute(itk_image_4d)

        # Create Image object from the 3D volume
        # We pass the already-loaded itk_image instead of a file path
        img = Image(itk_image=itk_image_3d, dtype=dtype, device=device)
        frames.append(img)

    logger.info(f"Successfully loaded {len(frames)} frames")
    return frames


def load_frame_directory(directory: str, pattern: str = '*.nii.gz',
                        dtype: torch.dtype = torch.float32,
                        device: str = 'cuda:0') -> List[Image]:
    """Load a timeseries from a directory of 3D volumes.

    Frames are sorted alphabetically by filename.

    Args:
        directory: Path to directory containing frame files
        pattern: Glob pattern for matching files (default: '*.nii.gz')
        dtype: PyTorch dtype for the image data
        device: Device to load images onto

    Returns:
        List of Image objects sorted by filename

    Example:
        >>> frames = load_frame_directory('frames/', pattern='frame_*.nii.gz')
        >>> print(f"Loaded {len(frames)} frames")
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise ValueError(f"Directory not found: {directory}")

    # Find all matching files and sort
    frame_files = sorted(dir_path.glob(pattern))

    if len(frame_files) == 0:
        raise ValueError(f"No files matching pattern '{pattern}' found in {directory}")

    logger.info(f"Found {len(frame_files)} frames in {directory}")

    # Load each frame
    frames = []
    for frame_file in frame_files:
        img = Image.load_file(str(frame_file), dtype=dtype, device=device)
        frames.append(img)

    logger.info(f"Successfully loaded {len(frames)} frames")
    return frames


def estimate_chunk_size(frame_shape: Tuple[int, ...],
                       registration_type: str,
                       dtype: torch.dtype = torch.float32,
                       device: str = 'cuda:0',
                       safety_factor: float = 0.6) -> int:
    """Estimate optimal chunk size based on available GPU memory.

    Args:
        frame_shape: Shape of a single frame [C, H, W, D] or [H, W, D]
        registration_type: Type of registration ('Rigid', 'Affine', 'Greedy', 'SyN')
        dtype: Data type of frames
        device: Target device for estimation
        safety_factor: Fraction of free memory to use (default: 0.6 for safety)

    Returns:
        Recommended chunk size (number of frames to process simultaneously)

    Example:
        >>> chunk_size = estimate_chunk_size((128, 128, 128), 'Greedy')
        >>> print(f"Process {chunk_size} frames at a time")
    """
    if device == 'cpu':
        # CPU has more memory, use larger chunks
        return 50

    # Get available GPU memory
    if torch.cuda.is_available():
        mem_free, mem_total = torch.cuda.mem_get_info(device=device)
    else:
        logger.warning(f"CUDA not available, defaulting to chunk_size=10")
        return 10

    # Calculate memory per frame
    if len(frame_shape) == 3:
        frame_shape = (1,) + frame_shape  # Add channel dimension

    bytes_per_element = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else 4
    frame_bytes = int(np.prod(frame_shape) * bytes_per_element)

    # Estimate overhead based on registration type
    if registration_type in ['Rigid', 'Affine']:
        # Small parameter count: just transformation matrices
        # Overhead: ~100 bytes per frame for parameters
        overhead_per_frame = frame_bytes * 2 + 100  # 2x for downsampled copies
    elif registration_type in ['Greedy']:
        # Warp field: [H, W, D, 3] ≈ 3x frame size (assuming single channel)
        # Plus downsampled copies and gradients
        overhead_per_frame = frame_bytes * 6  # Conservative estimate
    elif registration_type == 'SyN':
        # Two warp fields (forward + reverse) plus downsampled copies
        overhead_per_frame = frame_bytes * 8
    else:
        # Unknown type, be conservative
        overhead_per_frame = frame_bytes * 10

    # Calculate chunk size
    usable_memory = mem_free * safety_factor
    chunk_size = int(usable_memory / overhead_per_frame)

    # Clamp to reasonable range
    chunk_size = max(1, min(chunk_size, 50))

    logger.info(f"GPU memory: {mem_free / 1e9:.2f} GB free / {mem_total / 1e9:.2f} GB total")
    logger.info(f"Estimated chunk size for {registration_type}: {chunk_size} frames")

    return chunk_size


def save_transform_bundle(transforms: List[np.ndarray], output_path: str,
                          transform_type: str = 'warp', format: str = 'npz') -> None:
    """Save a bundle of transformations to a single file.

    Args:
        transforms: List of transformation arrays (affine matrices or warp fields)
        output_path: Output file path
        transform_type: Type of transformation ('affine', 'warp')
        format: Output format ('npz', 'h5')

    Example:
        >>> # Save affine matrices
        >>> affines = [reg.get_affine_matrix() for reg in registrations]
        >>> save_transform_bundle(affines, 'output/affines.npz', 'affine')
        >>>
        >>> # Save warp fields
        >>> warps = [reg.get_warp() for reg in registrations]
        >>> save_transform_bundle(warps, 'output/warps.npz', 'warp')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'npz':
        # NumPy archive format
        data_dict = {}
        for i, transform in enumerate(transforms):
            data_dict[f'{transform_type}_{i:04d}'] = transform

        np.savez_compressed(output_path, **data_dict)
        logger.info(f"Saved {len(transforms)} {transform_type} transforms to {output_path}")

    elif format == 'h5':
        # HDF5 format (requires h5py)
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py not installed. Install with: pip install h5py")

        with h5py.File(output_path, 'w') as f:
            for i, transform in enumerate(transforms):
                f.create_dataset(f'{transform_type}_{i:04d}', data=transform,
                               compression='gzip', compression_opts=4)

        logger.info(f"Saved {len(transforms)} {transform_type} transforms to {output_path} (HDF5)")

    else:
        raise ValueError(f"Unsupported format: {format}. Use 'npz' or 'h5'")


def load_transform_bundle(bundle_path: str, format: str = 'npz') -> List[np.ndarray]:
    """Load a bundle of transformations from a file.

    Args:
        bundle_path: Path to transform bundle file
        format: File format ('npz', 'h5')

    Returns:
        List of transformation arrays in order

    Example:
        >>> transforms = load_transform_bundle('output/warps.npz')
        >>> print(f"Loaded {len(transforms)} transforms")
    """
    bundle_path = Path(bundle_path)

    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle file not found: {bundle_path}")

    if format == 'npz':
        data = np.load(bundle_path)
        # Sort keys to maintain order
        keys = sorted(data.keys())
        transforms = [data[k] for k in keys]
        logger.info(f"Loaded {len(transforms)} transforms from {bundle_path}")
        return transforms

    elif format == 'h5':
        import h5py
        transforms = []
        with h5py.File(bundle_path, 'r') as f:
            # Sort keys to maintain order
            keys = sorted(f.keys())
            for key in keys:
                transforms.append(f[key][:])
        logger.info(f"Loaded {len(transforms)} transforms from {bundle_path} (HDF5)")
        return transforms

    else:
        raise ValueError(f"Unsupported format: {format}")


if __name__ == '__main__':
    # Test loading
    import sys

    if len(sys.argv) < 2:
        print("Usage: python timeseries.py <4d_file_or_directory>")
        sys.exit(1)

    input_path = sys.argv[1]

    if Path(input_path).is_dir():
        frames = load_frame_directory(input_path)
    else:
        frames = load_4d_nifti(input_path)

    print(f"Loaded {len(frames)} frames")
    print(f"Frame shape: {frames[0].shape}")

    # Test chunk size estimation
    chunk_size = estimate_chunk_size(frames[0].shape[1:], 'Greedy')
    print(f"Recommended chunk size for Greedy: {chunk_size}")
