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


'''
Stretched Cross Correlation (SCC)
Based on AFNI's lpc/lpa cost functions, applies arctanh transformation
to emphasize higher correlation values.
'''
import torch
from fireants.losses.cc import LocalNormalizedCrossCorrelationLoss


class StretchedCrossCorrelationLoss(LocalNormalizedCrossCorrelationLoss):
    """Stretched Cross Correlation (SCC) loss.

    Extends the standard local normalized cross-correlation by applying an
    arctanh transformation to stretch higher correlation values, giving them
    more emphasis in the final loss. This is based on AFNI's lpc cost function.

    The transformation:
    1. Compute NCC as usual: ncc ∈ [-1, 1]
    2. Apply stretching: pc = arctanh(ncc) = 0.5 * log((1+ncc)/(1-ncc))
    3. Emphasize with squared term: stretched = pc * abs(pc) = pc²·sign(pc)
    4. Aggregate (weighted if intensity_weighting=True)

    This makes high positive correlations (near 1.0) have much larger impact
    on the loss than moderate correlations, encouraging better alignment.

    Inherits all parameters from LocalNormalizedCrossCorrelationLoss.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Small epsilon for numerical stability in arctanh
        self.eps = 1e-7

    def forward(self, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """
        Compute stretched cross-correlation loss.

        Args:
            target: Target image tensor
            pred: Predicted/warped image tensor

        Returns:
            Scalar loss value (negative of stretched correlation)
        """
        # Get parent class NCC computation logic by calling super().forward()
        # But we need to get the ncc values before reduction, so we'll replicate
        # the parent logic here

        # Temporarily set reduction to 'none' to get per-voxel ncc
        original_reduction = self.reduction
        self.reduction = 'none'

        # Get per-voxel NCC values using parent forward (returns -ncc, so negate back)
        ncc = -super().forward(target, pred)

        # Restore original reduction
        self.reduction = original_reduction

        # Clamp NCC to prevent arctanh(±1) = ±∞
        # Map [-1, 1] → [-1+eps, 1-eps]
        ncc_clamped = ncc.clamp(min=-1.0 + self.eps, max=1.0 - self.eps)

        # Apply arctanh transformation: pc = 0.5 * log((1+ncc)/(1-ncc))
        pc = torch.atanh(ncc_clamped)

        # Stretch by squaring with sign: pc * abs(pc) = pc² for pc>0, -pc² for pc<0
        stretched = pc * pc.abs()

        # Apply intensity weighting if enabled
        # (Parent class does this, but we've bypassed it by setting reduction='none')
        if self.intensity_weighting:
            # Recompute weights from target and pred
            # (same logic as in parent class)
            from fireants.losses.cc import separable_filtering

            if self.intensity_weight_gaussians is not None:
                target_smooth = separable_filtering(target, self.intensity_weight_gaussians)
                pred_smooth = separable_filtering(pred, self.intensity_weight_gaussians)
            else:
                target_smooth = target
                pred_smooth = pred

            weights = (target_smooth.abs() + pred_smooth.abs()) / 2.0
            weights = weights / (weights.mean() + 1e-8)
            weighted_stretched = stretched * weights
        else:
            weighted_stretched = stretched

        # Apply reduction
        if self.reduction == 'sum':
            return torch.sum(weighted_stretched).neg()  # negative because we maximize correlation
        elif self.reduction == 'none':
            return weighted_stretched.neg()
        elif self.reduction == 'mean':
            return torch.mean(weighted_stretched).neg()
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}')
