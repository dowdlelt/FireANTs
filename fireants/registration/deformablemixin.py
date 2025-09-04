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

from typing import Callable, List, Optional, Union

import numpy as np
import SimpleITK as sitk
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, Adam
from tqdm import tqdm

from fireants.io.image import BatchedImages
from fireants.losses.cc import gaussian_1d, separable_filtering
from fireants.registration.abstract import AbstractRegistration
from fireants.registration.deformation.compositive import CompositiveWarp
from fireants.registration.deformation.svf import StationaryVelocity
from fireants.utils.globals import MIN_IMG_SIZE
from fireants.utils.imageutils import downsample


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
    - warp: Deformation model (for deformation restriction)
    """

    def setup_deformation_restriction(self, restrict_deformation, device, force_legacy_behavior=False):
        """Setup deformation restriction for nonlinear registration.

        Args:
            restrict_deformation (Optional[Union[List[float], tuple]]): Weights to restrict
                deformation along specific dimensions. For example, (1,1,0) restricts 3D
                deformation to first two dimensions. Must have same length as number of
                spatial dimensions.
            device: PyTorch device for tensor creation
            force_legacy_behavior (bool): If True, forces the old behavior where smoothing
                is disabled for partial restrictions instead of using anisotropic smoothing.
                Used for testing and comparison purposes.
        """
        if restrict_deformation is not None:
            if len(restrict_deformation) != self.dims:
                raise ValueError(
                    f"restrict_deformation must have length {self.dims}, got {len(restrict_deformation)}"
                )
            self.restrict_deformation = torch.tensor(
                restrict_deformation, dtype=torch.float32, device=device
            )
            # Check if restriction is partial (some dimensions restricted, others not)
            natural_partial_restriction = not (
                torch.all(self.restrict_deformation == 0)
                or torch.all(self.restrict_deformation == 1)
            )
            
            # Allow override for testing - force legacy behavior
            if force_legacy_behavior and natural_partial_restriction:
                self.has_partial_restriction = False  # Force old behavior
                self.force_legacy_behavior = True
            else:
                self.has_partial_restriction = natural_partial_restriction
                self.force_legacy_behavior = False
        else:
            self.restrict_deformation = None
            self.has_partial_restriction = False
            self.force_legacy_behavior = False

    def apply_deformation_restriction(self, warp_model):
        """Apply deformation restrictions by masking gradients for restricted dimensions.

        This method multiplies the gradients of the warp parameters by the restriction mask,
        effectively zeroing out gradients for dimensions where deformation should not occur.

        Args:
            warp_model: The deformation model (e.g., self.warp, self.fwd_warp, self.rev_warp)
        """
        if self.restrict_deformation is not None:
            # Handle different parameter names for different deformation types
            param = None
            if hasattr(warp_model, "warp") and warp_model.warp.grad is not None:
                param = warp_model.warp  # CompositiveWarp
            elif (
                hasattr(warp_model, "velocity_field")
                and warp_model.velocity_field.grad is not None
            ):
                param = warp_model.velocity_field  # StationaryVelocity

            if param is not None:
                # Warp/velocity field shape is [N, H, W, D, dims] for 3D or [N, H, W, dims] for 2D
                # We need to mask the last dimension according to restriction
                # Create proper shape for broadcasting: (..., 1, 1, ..., dims)
                mask_shape = [1] * (param.grad.ndim - 1) + [self.dims]
                mask = self.restrict_deformation.view(*mask_shape)
                param.grad.data *= mask

    def apply_warp_field_restriction(self, warp_field):
        """Apply deformation restrictions directly to warp field values.

        This method zeros out warp field components in restricted dimensions.
        Used to enforce restrictions after smoothing operations that may have
        coupled restricted and unrestricted dimensions.

        Args:
            warp_field (torch.Tensor): Warp field tensor with shape [N, H, W, [D], dims]

        Returns:
            torch.Tensor: Warp field with restricted dimensions zeroed out
        """
        if self.restrict_deformation is not None:
            # Create mask shape for broadcasting: (..., 1, 1, ..., dims)
            mask_shape = [1] * (warp_field.ndim - 1) + [self.dims]
            mask = self.restrict_deformation.view(*mask_shape)
            warp_field = warp_field * mask
        return warp_field

    def apply_anisotropic_smoothing(
        self, warp_field, gaussians, permute_vtoimg, permute_imgtov
    ):
        """Apply smoothing only in unrestricted spatial dimensions.

        This method applies separable filtering dimension by dimension, but only
        for dimensions that are not restricted. This preserves the smoothing benefits
        while respecting deformation restrictions.

        Args:
            warp_field (torch.Tensor): Warp field to smooth
            gaussians (List[torch.Tensor]): Gaussian kernels for each spatial dimension
            permute_vtoimg: Permutation to go from vector to image format
            permute_imgtov: Permutation to go from image to vector format

        Returns:
            torch.Tensor: Smoothed warp field with restrictions respected
        """
        if (self.restrict_deformation is None or not self.has_partial_restriction or 
            getattr(self, 'force_legacy_behavior', False)):
            # No restrictions, no partial restrictions, or forced legacy behavior - use normal smoothing
            from fireants.losses.cc import separable_filtering

            return separable_filtering(
                warp_field.permute(*permute_vtoimg), gaussians
            ).permute(*permute_imgtov)

        # Apply anisotropic smoothing - only smooth in spatial dimensions,
        # and only affect unrestricted deformation components
        from fireants.losses.cc import separable_filtering

        # For each deformation component (X, Y, Z), only smooth if that component is unrestricted
        smoothed_components = []
        for dim_idx in range(self.dims):
            component = warp_field[..., dim_idx : dim_idx + 1]  # [N, H, W, [D], 1]

            if self.restrict_deformation[dim_idx] != 0:
                # This dimension is unrestricted - apply full spatial smoothing
                smoothed = separable_filtering(
                    component.permute(*permute_vtoimg), gaussians
                ).permute(*permute_imgtov)
            else:
                # This dimension is restricted - keep it as zero (don't smooth)
                smoothed = component * 0  # Ensure it stays zero

            smoothed_components.append(smoothed)

        return torch.cat(smoothed_components, dim=-1)

    def should_disable_smoothing(self):
        """Check if smoothing should be disabled due to partial deformation restrictions.

        Returns:
            bool: Always False now - we use anisotropic smoothing instead
        """
        return False  # We now handle restrictions with anisotropic smoothing

    def save_as_ants_transforms(reg, filenames: Union[str, List[str]]):
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
        assert len(filenames) == reg.opt_size, (
            "Number of filenames should match the number of warps"
        )
        # get the warp field
        fixed_image: BatchedImages = reg.fixed_images
        moving_image: BatchedImages = reg.moving_images

        # get the moved coordinates and initial grid in pytorch space
        moved_coords = reg.get_warped_coordinates(
            fixed_image, moving_image
        )  # [B, H, W, [D], dim]
        init_grid = F.affine_grid(
            torch.eye(reg.dims, reg.dims + 1, device=moved_coords.device)[None],
            fixed_image.shape,
            align_corners=True,
        )
        # this is now moved displacements
        moving_t2p = moving_image.get_torch2phy()
        fixed_t2p = fixed_image.get_torch2phy()

        moved_coords = (
            torch.einsum(
                "bij, b...j->b...i", moving_t2p[:, : reg.dims, : reg.dims], moved_coords
            )
            + moving_t2p[:, : reg.dims, reg.dims]
        )
        init_grid = (
            torch.einsum(
                "bij, b...j->b...i", fixed_t2p[:, : reg.dims, : reg.dims], init_grid
            )
            + fixed_t2p[:, : reg.dims, reg.dims]
        )
        moved_coords = moved_coords - init_grid
        # save
        for i in range(reg.opt_size):
            moved_disp = moved_coords[i].detach().cpu().numpy()  # [H, W, D, 3]
            savefile = filenames[i]
            # get itk image
            if len(fixed_image.images) < i:  # this image is probably broadcasted then
                itk_data = fixed_image.images[0].itk_image
            else:
                itk_data = fixed_image.images[i].itk_image
            # copy itk data
            warp = sitk.GetImageFromArray(moved_disp)
            warp.CopyInformation(itk_data)
            sitk.WriteImage(warp, savefile)
