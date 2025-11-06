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
    more emphasis in the final loss. This is based on AFNI's lpc/lpa cost functions.

    Two variants:
    - lpc (use_absolute=False): minimizes weighted mean of pc*abs(pc)
      Good when you want positive correlation
    - lpa (use_absolute=True, default): minimizes 1 - abs(lpc)
      Good for similar contrast volumes, maximizes absolute correlation
      AFNI recommends lpa as first choice for similar contrast alignment

    The transformation:
    1. Compute NCC as usual: ncc ∈ [-1, 1]
    2. Apply stretching: pc = arctanh(ncc) = 0.5 * log((1+ncc)/(1-ncc))
    3. Emphasize with squared term: stretched = pc * abs(pc) = pc²·sign(pc)
    4. Aggregate to lpc = weighted_mean(stretched)
    5. If use_absolute: lpa = 1 - abs(lpc), else: return lpc

    This makes high correlations (near ±1.0) have much larger impact
    on the loss than moderate correlations, encouraging better alignment.

    Inherits all parameters from LocalNormalizedCrossCorrelationLoss.

    Args:
        use_absolute: If True (default), use lpa variant (1 - abs(lpc)).
                     Recommended for similar contrast volumes.
                     If False, use lpc variant (just lpc itself).
    """

    def __init__(self, *args, use_absolute: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        # Small epsilon for numerical stability in arctanh
        self.eps = 1e-7
        self.use_absolute = use_absolute

    def forward(self, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """
        Compute stretched cross-correlation loss.

        Args:
            target: Target image tensor
            pred: Predicted/warped image tensor

        Returns:
            Scalar loss value (lpa or lpc to minimize)
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

        # Compute lpc = weighted mean of stretched values
        if original_reduction == 'sum':
            lpc = torch.sum(weighted_stretched)
        elif original_reduction == 'mean':
            lpc = torch.mean(weighted_stretched)
        else:  # 'none' - shouldn't happen in practice for registration
            lpc = torch.mean(weighted_stretched)

        # Apply lpa transformation if requested: lpa = 1 - abs(lpc)
        # This makes the loss always positive and minimizes to 0 for perfect correlation
        if self.use_absolute:
            # lpa variant: maximize absolute correlation (works for both positive and negative)
            # Smaller lpa = better (approaches 0 when |lpc| is large)
            loss = 1.0 - torch.abs(lpc)
        else:
            # lpc variant: minimize lpc directly (assumes positive correlation desired)
            # Negative lpc = good, positive lpc = bad
            loss = -lpc

        return loss
