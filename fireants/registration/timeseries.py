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
Timeseries registration engine for efficient batch processing of 4D volumes.
"""

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import torch
import numpy as np
import logging
from tqdm import tqdm

from fireants.io.image import Image, BatchedImages
from fireants.io.timeseries import estimate_chunk_size, save_transform_bundle
from fireants.registration.rigid import RigidRegistration
from fireants.registration.affine import AffineRegistration
from fireants.registration.greedy import GreedyRegistration
from fireants.registration.syn import SyNRegistration
from fireants.interpolator import fireants_interpolator

logger = logging.getLogger(__name__)


class TimeseriesRegistration:
    """Ultra-fast registration engine for timeseries (4D) volumes.

    Registers multiple timepoints to a reference frame using chunked batch processing
    to maximize GPU utilization while managing memory efficiently.

    Supports two registration modes:
    - **Parallel**: All frames register to reference independently (maximum parallelism)
    - **Sequential**: Chain registrations (0→1, 1→2, ...) with warm-start initialization

    Supports multi-stage pipelines (e.g., Rigid→SyN) where each stage initializes
    from the previous stage's result.

    Args:
        frames: List of Image objects (one per timepoint)
        reference: Reference Image to register all frames to
        transform_types: List of registration types in order (e.g., ['Rigid', 'SyN'])
        mode: Registration strategy - 'parallel' or 'sequential'
        chunk_size: Number of frames to process simultaneously (None = auto-detect)
        registration_params: Dict of parameters for each transform type
        output_prefix: Base path for saving results
        device: Device to run on ('cuda:0', 'cpu')
        progress_bar: Whether to show progress bars

    Example:
        >>> from fireants.io.timeseries import load_4d_nifti
        >>> frames = load_4d_nifti('bold.nii.gz')
        >>> reference = frames[0]
        >>>
        >>> ts_reg = TimeseriesRegistration(
        ...     frames=frames,
        ...     reference=reference,
        ...     transform_types=['Rigid', 'Greedy'],
        ...     mode='parallel',
        ...     output_prefix='output/timeseries'
        ... )
        >>> ts_reg.register()
    """

    def __init__(self,
                 frames: List[Image],
                 reference: Image,
                 transform_types: List[str],
                 mode: str = 'parallel',
                 chunk_size: Optional[int] = None,
                 registration_params: Optional[Dict[str, Dict[str, Any]]] = None,
                 output_prefix: str = 'output/timeseries',
                 device: str = 'cuda:0',
                 progress_bar: bool = True,
                 save_warped_timeseries: bool = False):

        self.frames = frames
        self.reference = reference
        self.transform_types = transform_types
        self.mode = mode
        self.device = device
        self.progress_bar = progress_bar
        self.output_prefix = output_prefix
        self.save_warped_timeseries = save_warped_timeseries

        # Storage for warped volumes (if saving)
        self.warped_volumes = [] if save_warped_timeseries else None

        # Auto-detect chunk size if not provided
        if chunk_size is None:
            # Use the last (most memory-intensive) transform type for estimation
            last_transform = transform_types[-1]
            self.chunk_size = estimate_chunk_size(
                frames[0].shape[1:],  # Remove batch dim
                last_transform,
                frames[0].array.dtype,
                device
            )
        else:
            self.chunk_size = chunk_size

        # Registration parameters for each transform type
        self.registration_params = registration_params or {}

        # Validate parameters
        self._validate_params()

        logger.info(f"TimeseriesRegistration initialized:")
        logger.info(f"  Frames: {len(frames)}")
        logger.info(f"  Transform pipeline: {' → '.join(transform_types)}")
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Chunk size: {self.chunk_size}")

    def _validate_params(self):
        """Validate initialization parameters."""
        valid_modes = ['parallel', 'sequential']
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be one of {valid_modes}")

        valid_transforms = ['Rigid', 'Affine', 'Greedy', 'SyN']
        for t in self.transform_types:
            if t not in valid_transforms:
                raise ValueError(f"Invalid transform type '{t}'. Must be one of {valid_transforms}")

        if len(self.frames) == 0:
            raise ValueError("No frames provided")

        # Validate reference shape matches frames
        ref_shape = self.reference.shape[1:]  # Exclude batch dim
        for i, frame in enumerate(self.frames):
            frame_shape = frame.shape[1:]
            if frame_shape != ref_shape:
                raise ValueError(f"Frame {i} shape {frame_shape} doesn't match reference shape {ref_shape}")

    def register(self) -> Dict[str, List[Any]]:
        """Run timeseries registration.

        Returns:
            Dictionary mapping transform type to list of transformations
            Example: {'Rigid': [affine0, affine1, ...], 'Greedy': [warp0, warp1, ...]}
        """
        if self.mode == 'parallel':
            return self._register_parallel()
        elif self.mode == 'sequential':
            return self._register_sequential()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _register_parallel(self) -> Dict[str, List[Any]]:
        """Register all frames to reference independently with chunked batching."""
        logger.info("Starting parallel registration...")

        all_transforms = {t: [] for t in self.transform_types}
        num_frames = len(self.frames)
        num_chunks = (num_frames + self.chunk_size - 1) // self.chunk_size

        # Process each chunk
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, num_frames)
            chunk_frames = self.frames[chunk_start:chunk_end]
            n_frames_in_chunk = len(chunk_frames)

            logger.info(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} (frames {chunk_start}-{chunk_end-1})")

            # Create batched images
            # Reference stays as single image - registration will broadcast automatically
            fixed_batch = BatchedImages([self.reference])
            moving_batch = BatchedImages(chunk_frames)

            # Multi-stage registration pipeline
            prev_transform = None
            prev_transform_type = None

            for transform_type in self.transform_types:
                logger.info(f"  Stage: {transform_type}")

                # Get parameters for this transform type
                params = self.registration_params.get(transform_type, {})

                # Initialize from previous stage if available
                if prev_transform is not None:
                    params = self._prepare_init_transform(params, prev_transform, prev_transform_type, transform_type)

                # Create and run registration
                reg = self._create_registration(
                    transform_type,
                    fixed_batch,
                    moving_batch,
                    params
                )

                reg.optimize()

                # Extract transformations from this chunk
                chunk_transforms = self._extract_transforms(reg, transform_type)
                all_transforms[transform_type].extend(chunk_transforms)

                # Save chunk results progressively
                self._save_chunk_transforms(
                    chunk_transforms,
                    transform_type,
                    chunk_start,
                    chunk_end
                )

                prev_transform = reg
                prev_transform_type = transform_type

            # Apply warps and save warped volumes if requested
            if self.save_warped_timeseries:
                logger.info(f"  Applying warps to chunk...")
                warped_chunk = self._apply_warps_to_chunk(reg, moving_batch, transform_type)
                self.warped_volumes.extend(warped_chunk)

            # Free GPU memory
            del fixed_batch, moving_batch, reg
            torch.cuda.empty_cache()

        # Save final bundled transforms
        for transform_type in self.transform_types:
            self._save_final_bundle(all_transforms[transform_type], transform_type)

        # Save warped timeseries if accumulated
        if self.save_warped_timeseries and self.warped_volumes:
            self._save_warped_4d_timeseries()

        logger.info("\nParallel registration completed successfully")
        return all_transforms

    def _register_sequential(self) -> Dict[str, List[Any]]:
        """Register frames sequentially: 0→1, 1→2, 2→3, ...

        Each registration initializes from the previous one for temporal coherence.
        """
        logger.info("Starting sequential registration...")

        all_transforms = {t: [] for t in self.transform_types}
        num_frames = len(self.frames)

        # Initialize with identity/None
        prev_transforms = {t: None for t in self.transform_types}

        # Register each consecutive pair
        pbar = tqdm(range(num_frames - 1), desc="Sequential registration") if self.progress_bar else range(num_frames - 1)

        for i in pbar:
            frame_i = self.frames[i]
            frame_j = self.frames[i + 1]

            # Create single-image batches
            fixed_batch = BatchedImages([frame_i])
            moving_batch = BatchedImages([frame_j])

            # Multi-stage pipeline for this pair
            for stage_idx, transform_type in enumerate(self.transform_types):
                params = self.registration_params.get(transform_type, {}).copy()

                # Initialize from previous frame's transform (temporal coherence)
                if prev_transforms[transform_type] is not None:
                    params = self._prepare_init_transform(
                        params,
                        prev_transforms[transform_type],
                        transform_type,
                        transform_type
                    )

                # Create and run registration
                reg = self._create_registration(
                    transform_type,
                    fixed_batch,
                    moving_batch,
                    params
                )

                reg.optimize()

                # Extract single transform
                transform = self._extract_transforms(reg, transform_type)[0]
                all_transforms[transform_type].append(transform)

                # Update prev_transform for next pair
                prev_transforms[transform_type] = reg

                # Free memory from previous stage
                if stage_idx > 0:
                    torch.cuda.empty_cache()

            # Free memory after processing this pair
            del fixed_batch, moving_batch
            torch.cuda.empty_cache()

        # Save final bundled transforms
        for transform_type in self.transform_types:
            self._save_final_bundle(all_transforms[transform_type], transform_type)

        logger.info("\nSequential registration completed successfully")
        return all_transforms

    def _create_registration(self, transform_type: str,
                            fixed_batch: BatchedImages,
                            moving_batch: BatchedImages,
                            params: Dict[str, Any]):
        """Create registration object of specified type."""
        common_params = {
            'fixed_images': fixed_batch,
            'moving_images': moving_batch,
            'progress_bar': False,  # We handle our own progress bars
            **params
        }

        if transform_type == 'Rigid':
            return RigidRegistration(**common_params)
        elif transform_type == 'Affine':
            return AffineRegistration(**common_params)
        elif transform_type == 'Greedy':
            return GreedyRegistration(**common_params)
        elif transform_type == 'SyN':
            return SyNRegistration(**common_params)
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

    def _prepare_init_transform(self, params: Dict, prev_reg, prev_type: str, current_type: str) -> Dict:
        """Prepare initialization from previous registration stage."""
        params = params.copy()

        if current_type in ['Rigid', 'Affine']:
            # Initialize from previous affine transform
            if prev_type in ['Rigid', 'Affine']:
                params['init_rigid'] = prev_reg.get_affine_matrix(homogenous=False)
        elif current_type in ['Greedy', 'SyN']:
            # Initialize from previous affine or deformable transform
            if prev_type in ['Rigid', 'Affine']:
                params['init_affine'] = prev_reg.get_affine_matrix(homogenous=True)
            # TODO: Support initializing deformable from deformable (init_warp)

        return params

    def _extract_transforms(self, reg, transform_type: str) -> List[np.ndarray]:
        """Extract batch of transformations from registration object."""
        if transform_type in ['Rigid', 'Affine']:
            # Get affine matrices [N, D, D+1] or [N, D+1, D+1]
            affine_batch = reg.get_affine_matrix(homogenous=False).detach().cpu().numpy()
            # Split into list of individual matrices
            return [affine_batch[i] for i in range(affine_batch.shape[0])]
        elif transform_type in ['Greedy']:
            # Get warp field [N, H, W, D, 3]
            warp_batch = reg.warp.warp.detach().cpu().numpy()
            return [warp_batch[i] for i in range(warp_batch.shape[0])]
        elif transform_type == 'SyN':
            # Get forward warp field [N, H, W, D, 3]
            fwd_warp_batch = reg.fwd_warp.warp.detach().cpu().numpy()
            return [fwd_warp_batch[i] for i in range(fwd_warp_batch.shape[0])]
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

    def _save_chunk_transforms(self, transforms: List[np.ndarray],
                              transform_type: str, start_idx: int, end_idx: int):
        """Save transforms from a single chunk (progressive saving)."""
        chunk_path = f"{self.output_prefix}_{transform_type}_chunk_{start_idx:04d}_{end_idx:04d}.npz"
        save_transform_bundle(transforms, chunk_path, transform_type=transform_type.lower())

    def _save_final_bundle(self, transforms: List[np.ndarray], transform_type: str):
        """Save final bundled transforms for all frames."""
        bundle_path = f"{self.output_prefix}_{transform_type}_all.npz"
        save_transform_bundle(transforms, bundle_path, transform_type=transform_type.lower())
        logger.info(f"Saved {len(transforms)} {transform_type} transforms to {bundle_path}")

    def _apply_warps_to_chunk(self, reg, moving_batch: BatchedImages, transform_type: str) -> List[np.ndarray]:
        """Apply transformations to moving images and return warped volumes.

        Args:
            reg: Registration object with computed transformations
            moving_batch: BatchedImages of moving frames
            transform_type: Type of transformation

        Returns:
            List of warped volume arrays (CPU numpy arrays) [H, W, D]
        """
        # Get the transformation parameters
        if transform_type in ['Rigid', 'Affine']:
            # For rigid/affine, we need to apply the transformation
            warp_params = reg.get_warp_parameters(
                BatchedImages([self.reference]),
                moving_batch
            )
            # Apply affine transformation
            moving_arrays = moving_batch().to(self.device)  # [N, C, H, W, D]
            warped = fireants_interpolator(
                moving_arrays,
                affine=warp_params['affine'],
                out_shape=warp_params['out_shape'],
                mode='bilinear',
                align_corners=True
            )
        elif transform_type == 'Greedy':
            # Get warp field and apply
            warp_field = reg.warp.warp  # [N, H, W, D, 3]
            moving_arrays = moving_batch().to(self.device)
            warped = fireants_interpolator(
                moving_arrays,
                affine=None,
                grid=warp_field.contiguous(),
                mode='bilinear',
                align_corners=True,
                is_displacement=True
            )
        elif transform_type == 'SyN':
            # Get forward warp field and apply
            fwd_warp_field = reg.fwd_warp.warp  # [N, H, W, D, 3]
            moving_arrays = moving_batch().to(self.device)
            # Need to also apply initial affine if present
            affine_mat = reg.affine  # [N, D+1, D+1]
            moving_p2t = moving_batch.get_phy2torch().to(reg.dtype)
            fixed_t2p = BatchedImages([self.reference]).get_torch2phy().to(reg.dtype)
            affine_map = (torch.matmul(moving_p2t, torch.matmul(affine_mat, fixed_t2p))[:, :-1]).contiguous()

            warped = fireants_interpolator(
                moving_arrays,
                affine=affine_map,
                grid=fwd_warp_field.contiguous(),
                mode='bilinear',
                align_corners=True,
                is_displacement=True
            )
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

        # Convert to list of numpy arrays [H, W, D]
        warped_list = []
        for i in range(warped.shape[0]):
            vol = warped[i, 0].detach().cpu().numpy()  # Remove batch and channel dims
            warped_list.append(vol)

        return warped_list

    def _save_warped_4d_timeseries(self):
        """Save accumulated warped volumes as 4D NIfTI."""
        import SimpleITK as sitk

        output_path = f"{self.output_prefix}_warped.nii.gz"
        logger.info(f"Saving warped 4D timeseries to {output_path}")

        # Stack into 4D array
        # Each volume is [D, H, W] (z, y, x from SimpleITK's GetArrayFromImage ordering)
        # Stack along last axis: [D, H, W, T]
        warped_4d = np.stack(self.warped_volumes, axis=-1)
        logger.debug(f"Warped 4D shape before transpose: {warped_4d.shape}")

        # SimpleITK expects [T, D, H, W] for 4D data (time, z, y, x)
        # Transpose from [D, H, W, T] -> [T, D, H, W]
        warped_4d_sitk = np.transpose(warped_4d, (3, 0, 1, 2))  # [z,y,x,T] -> [T,z,y,x]
        logger.debug(f"Warped 4D shape after transpose: {warped_4d_sitk.shape}")

        itk_4d = sitk.GetImageFromArray(warped_4d_sitk)

        # Copy spatial metadata from reference
        ref_itk = self.reference.itk_image
        spacing_4d = list(ref_itk.GetSpacing()) + [1.0]  # Add time spacing (TR)
        origin_4d = list(ref_itk.GetOrigin()) + [0.0]    # Add time origin

        # Direction matrix: for 4D images (3D+time), use 3x3 for spatial dims only
        # Time dimension doesn't have a direction component
        direction_3d = ref_itk.GetDirection()  # Already flattened list of 9 elements

        itk_4d.SetSpacing(spacing_4d)
        itk_4d.SetOrigin(origin_4d)
        itk_4d.SetDirection(direction_3d)  # Use 3x3 spatial direction matrix

        # Save
        sitk.WriteImage(itk_4d, str(output_path))
        logger.info(f"Saved {len(self.warped_volumes)} warped volumes to {output_path}")


if __name__ == '__main__':
    # Test timeseries registration
    import sys
    from fireants.io.timeseries import load_4d_nifti

    if len(sys.argv) < 2:
        print("Usage: python timeseries.py <4d_file>")
        sys.exit(1)

    # Load frames
    frames = load_4d_nifti(sys.argv[1])
    reference = frames[0]

    # Test registration
    ts_reg = TimeseriesRegistration(
        frames=frames[:10],  # Test with first 10 frames
        reference=reference,
        transform_types=['Rigid'],
        mode='parallel',
        registration_params={
            'Rigid': {
                'scales': [4, 2, 1],
                'iterations': [100, 50, 25],
                'loss_type': 'cc',
                'cc_kernel_size': 5
            }
        },
        output_prefix='test_output/timeseries'
    )

    results = ts_reg.register()
    print(f"Registration complete. Saved transforms: {list(results.keys())}")
