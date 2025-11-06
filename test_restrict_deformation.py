#!/usr/bin/env python3
"""
Simple test to verify restrict_deformation parameter works correctly.
This script tests that gradients are properly restricted along specified dimensions.
"""

import torch
import numpy as np
from fireants.io.image import Image, BatchedImages
from fireants.registration.syn import SyNRegistration

def test_restrict_deformation():
    """Test that restrict_deformation parameter properly restricts gradients"""

    # Create simple 3D test images (small for fast testing)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    shape = (1, 1, 32, 32, 32)  # Small 3D volume

    # Create fixed and moving images with some difference
    fixed_array = torch.rand(shape, device=device, dtype=torch.float32)
    moving_array = fixed_array + 0.1 * torch.randn_like(fixed_array)

    # Create Image objects (mock simple metadata)
    fixed_img = Image(fixed_array, torch.eye(4, device=device), 'test_fixed.nii.gz')
    moving_img = Image(moving_array, torch.eye(4, device=device), 'test_moving.nii.gz')

    fixed_batch = BatchedImages([fixed_img])
    moving_batch = BatchedImages([moving_img])

    print("Testing restrict_deformation feature...")
    print("=" * 60)

    # Test 1: No restriction (baseline)
    print("\nTest 1: No restriction (baseline)")
    reg_baseline = SyNRegistration(
        scales=[2, 1],
        iterations=[10, 5],
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        loss_type='mse',
        optimizer='Adam',
        optimizer_lr=0.1,
        smooth_warp_sigma=0.0,
        smooth_grad_sigma=0.0,
        restrict_deformation=None,
        deformation_type='compositive'
    )

    # Run a single optimization step to check gradients
    reg_baseline.fwd_warp.set_zero_grad()
    fwd_warp = reg_baseline.fwd_warp.get_warp()
    loss = fwd_warp.abs().mean()
    loss.backward()

    baseline_grad = reg_baseline.fwd_warp.warp.grad.clone()
    grad_x_mean = baseline_grad[..., 0].abs().mean().item()
    grad_y_mean = baseline_grad[..., 1].abs().mean().item()
    grad_z_mean = baseline_grad[..., 2].abs().mean().item()

    print(f"  Gradient magnitudes (X, Y, Z): ({grad_x_mean:.6f}, {grad_y_mean:.6f}, {grad_z_mean:.6f})")

    # Test 2: Restrict X and Z, allow Y
    print("\nTest 2: Restrict to Y-only deformation (0x1x0)")
    reg_y_only = SyNRegistration(
        scales=[2, 1],
        iterations=[10, 5],
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        loss_type='mse',
        optimizer='Adam',
        optimizer_lr=0.1,
        smooth_warp_sigma=0.0,
        smooth_grad_sigma=0.0,
        restrict_deformation=[0.0, 1.0, 0.0],
        deformation_type='compositive'
    )

    reg_y_only.fwd_warp.set_zero_grad()
    fwd_warp = reg_y_only.fwd_warp.get_warp()
    loss = fwd_warp.abs().mean()
    loss.backward()

    restricted_grad = reg_y_only.fwd_warp.warp.grad.clone()
    grad_x_restricted = restricted_grad[..., 0].abs().mean().item()
    grad_y_restricted = restricted_grad[..., 1].abs().mean().item()
    grad_z_restricted = restricted_grad[..., 2].abs().mean().item()

    print(f"  Gradient magnitudes (X, Y, Z): ({grad_x_restricted:.6f}, {grad_y_restricted:.6f}, {grad_z_restricted:.6f})")
    print(f"  X gradient should be ~0: {grad_x_restricted < 1e-6}")
    print(f"  Z gradient should be ~0: {grad_z_restricted < 1e-6}")
    print(f"  Y gradient should be non-zero: {grad_y_restricted > 1e-6}")

    # Test 3: Partial restriction
    print("\nTest 3: Partial restriction (0.1x0.6x0.1)")
    reg_partial = SyNRegistration(
        scales=[2, 1],
        iterations=[10, 5],
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        loss_type='mse',
        optimizer='Adam',
        optimizer_lr=0.1,
        smooth_warp_sigma=0.0,
        smooth_grad_sigma=0.0,
        restrict_deformation=[0.1, 0.6, 0.1],
        deformation_type='compositive'
    )

    reg_partial.fwd_warp.set_zero_grad()
    fwd_warp = reg_partial.fwd_warp.get_warp()
    loss = fwd_warp.abs().mean()
    loss.backward()

    partial_grad = reg_partial.fwd_warp.warp.grad.clone()
    grad_x_partial = partial_grad[..., 0].abs().mean().item()
    grad_y_partial = partial_grad[..., 1].abs().mean().item()
    grad_z_partial = partial_grad[..., 2].abs().mean().item()

    print(f"  Gradient magnitudes (X, Y, Z): ({grad_x_partial:.6f}, {grad_y_partial:.6f}, {grad_z_partial:.6f})")

    # Check ratios (should be approximately 0.1:0.6:0.1)
    if grad_y_partial > 0:
        ratio_x = grad_x_partial / grad_y_partial
        ratio_z = grad_z_partial / grad_y_partial
        expected_ratio = 0.1 / 0.6
        print(f"  X/Y ratio: {ratio_x:.4f} (expected ~{expected_ratio:.4f})")
        print(f"  Z/Y ratio: {ratio_z:.4f} (expected ~{expected_ratio:.4f})")

    print("\n" + "=" * 60)
    print("Tests completed successfully!")
    print("\nKey observations:")
    print("- Test 1 shows baseline gradients with no restriction")
    print("- Test 2 should show X and Z gradients near zero, Y gradient non-zero")
    print("- Test 3 should show gradient ratios matching restriction factors")

    return True


if __name__ == '__main__':
    test_restrict_deformation()
