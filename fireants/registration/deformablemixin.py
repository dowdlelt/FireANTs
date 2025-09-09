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


## Class to inherit common functions to Greedy and SyN

import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.nn import functional as F
import numpy as np
from typing import List, Optional, Union, Callable
from tqdm import tqdm
import SimpleITK as sitk

from fireants.utils.globals import MIN_IMG_SIZE
from fireants.io.image import BatchedImages
from fireants.registration.abstract import AbstractRegistration
from fireants.registration.deformation.svf import StationaryVelocity
from fireants.registration.deformation.compositive import CompositiveWarp
from fireants.losses.cc import gaussian_1d, separable_filtering
from fireants.utils.imageutils import downsample
from fireants.utils.util import compose_warp
from fireants.utils.warputils import compositive_warp_inverse


class DeformableMixin:
    """Mixin class providing common functionality for deformable registration classes.

    This mixin implements shared methods used by both GreedyRegistration and SyNRegistration
    classes, particularly for saving deformation fields in a format compatible with ANTs
    (Advanced Normalization Tools) and other widely used registration tools.

    The mixin assumes the parent class has:

    - opt_size: Number of registration pairs
    - dims: Number of spatial dimensions
    - fixed_images: BatchedImages object containing fixed images
    - moving_images: BatchedImages object containing moving images
    - get_warped_coordinates(): Method to get transformed coordinates
    """

    def save_as_ants_transforms(reg, filenames: Union[str, List[str]], cache_inverse: bool = True):
        """Save deformation fields in ANTs-compatible format.

        Converts the learned deformation fields to displacement fields in physical space
        and saves them in a format that can be used by ANTs registration tools.
        The displacement fields are saved as multi-component images where each component
        represents the displacement along one spatial dimension.

        Args:
            filenames (Union[str, List[str]]): Path(s) where the transform(s) should be saved.
                If a single string is provided for multiple transforms, it will be treated
                as the first filename. For multiple transforms, provide a list of filenames
                matching the number of transforms.

        Raises:
            AssertionError: If number of filenames doesn't match number of transforms (opt_size)

        !!! caution "Physical Space Coordinates"
            The saved transforms are in physical space coordinates, not normalized [-1,1] space.
            The displacement fields are saved with the same orientation and spacing as the 
            fixed images.
        """
        if isinstance(filenames, str):
            filenames = [filenames]
        assert len(filenames) == reg.opt_size, "Number of filenames should match the number of warps"
        # get the warp field
        fixed_image: BatchedImages = reg.fixed_images
        moving_image: BatchedImages = reg.moving_images

        # Determine if this is a symmetric (SyN-like) registration needing composition of forward and inverse warps.
        if hasattr(reg, 'fwd_warp') and hasattr(reg, 'rev_warp'):
            # Build affine components (similar logic to Greedy/SyN)
            fixed_arrays = fixed_image()
            fixed_t2p = fixed_image.get_torch2phy()
            moving_p2t = moving_image.get_phy2torch()
            affine_map_init = torch.matmul(moving_p2t, torch.matmul(reg.affine, fixed_t2p))[:, :-1]
            fixed_image_affinecoords = F.affine_grid(affine_map_init, fixed_image.shape, align_corners=True)
            # Identity grid in normalized space
            init_grid = F.affine_grid(torch.eye(reg.dims, reg.dims+1, device=fixed_arrays.device)[None], \
                                      fixed_image.shape, align_corners=True)
            # Current forward / reverse warps
            fwd_warp_field = reg.fwd_warp.get_warp().detach()
            rev_warp_field = reg.rev_warp.get_warp().detach()
            if fwd_warp_field.shape[1:-1] != init_grid.shape[1:-1]:
                fwd_warp_field = F.interpolate(fwd_warp_field.permute(0, reg.dims+1, *range(1, reg.dims+1)), size=init_grid.shape[1:-1], mode='trilinear' if reg.dims==3 else 'bilinear', align_corners=True).permute(0, *range(2, reg.dims+2), 1)
            if rev_warp_field.shape[1:-1] != init_grid.shape[1:-1]:
                rev_warp_field = F.interpolate(rev_warp_field.permute(0, reg.dims+1, *range(1, reg.dims+1)), size=init_grid.shape[1:-1], mode='trilinear' if reg.dims==3 else 'bilinear', align_corners=True).permute(0, *range(2, reg.dims+2), 1)
            # Full reverse deformation (phi_rev = v + id)
            rev_full = rev_warp_field + init_grid
            # Cached inverse of reverse warp to avoid recomputation
            if hasattr(reg, '_invert_compositive_iterative') and getattr(reg, 'inverse_method', 'iterative') == 'iterative':
                if cache_inverse and getattr(reg, '_cached_rev_inv_warp', None) is not None:
                    rev_inv_warp_field = reg._cached_rev_inv_warp
                else:
                    rev_inv_warp_field = reg._invert_compositive_iterative(rev_full, init_grid, iters=getattr(reg, 'inverse_iters', 20))
                    if cache_inverse:
                        reg._cached_rev_inv_warp = rev_inv_warp_field.detach()
            else:
                if cache_inverse and getattr(reg, '_cached_rev_inv_warp', None) is not None:
                    rev_inv_warp_field = reg._cached_rev_inv_warp
                else:
                    rev_inv_warp_field = compositive_warp_inverse(fixed_image, rev_full, displacement=True)
                    if cache_inverse:
                        reg._cached_rev_inv_warp = rev_inv_warp_field.detach()
            # Optional smoothing if configured at higher level (SyN sets smooth_warp_sigma=0 inside compositive warp usually)
            if getattr(reg, 'smooth_warp_sigma', 0) > 0:
                warp_gaussian = [gaussian_1d(s, truncated=2) for s in (torch.zeros(reg.dims, device=fixed_arrays.device) + reg.smooth_warp_sigma)]
                fwd_warp_field = separable_filtering(fwd_warp_field.permute(0, reg.dims+1, *range(1, reg.dims+1)), warp_gaussian).permute(0, *range(2, reg.dims+2), 1)
                rev_inv_warp_field = separable_filtering(rev_inv_warp_field.permute(0, reg.dims+1, *range(1, reg.dims+1)), warp_gaussian).permute(0, *range(2, reg.dims+2), 1)
            # Compose (forward ◦ inverse(reverse)) in displacement form relative to init_grid
            composed_warp = compose_warp(fwd_warp_field, rev_inv_warp_field, init_grid)
            moved_coords = fixed_image_affinecoords + composed_warp
        else:
            # Fallback to generic path (e.g., Greedy / single warp models)
            moved_coords = reg.get_warped_coordinates(fixed_image, moving_image)   # [B, H, W, [D], dim]
            init_grid = F.affine_grid(torch.eye(reg.dims, reg.dims+1, device=moved_coords.device)[None], \
                                        fixed_image.shape, align_corners=True)
        # this is now moved displacements
        moving_t2p = moving_image.get_torch2phy()
        fixed_t2p = fixed_image.get_torch2phy()

        moved_coords = torch.einsum('bij, b...j->b...i', moving_t2p[:, :reg.dims, :reg.dims], moved_coords) + moving_t2p[:, :reg.dims, reg.dims]
        init_grid = torch.einsum('bij, b...j->b...i', fixed_t2p[:, :reg.dims, :reg.dims], init_grid) + fixed_t2p[:, :reg.dims, reg.dims]
        moved_coords = moved_coords - init_grid
        # save 
        for i in range(reg.opt_size):
            moved_disp = moved_coords[i].detach().cpu().numpy()  # [H, W, D, 3]
            savefile = filenames[i]
            # get itk image
            if len(fixed_image.images) < i:     # this image is probably broadcasted then
                itk_data = fixed_image.images[0].itk_image
            else:
                itk_data = fixed_image.images[i].itk_image
            # copy itk data
            warp = sitk.GetImageFromArray(moved_disp)
            warp.CopyInformation(itk_data)
            sitk.WriteImage(warp, savefile)
