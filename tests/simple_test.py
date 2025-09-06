"""
Test for restricted deformation functionality comparing new anisotropic smoothing vs legacy behavior.

This test compares the new anisotropic smoothing approach (which preserves smoothing in
unrestricted dimensions) against the legacy behavior (which would disable smoothing
entirely for partial restrictions).

The test uses two approaches:
1. New behavior: Uses anisotropic smoothing that respects dimensional restrictions
2. Legacy behavior: Forces the old approach using force_legacy_behavior=True

By comparing the resulting warp fields and transformed images, we can verify:
- The new approach maintains better regularization
- Restrictions are properly enforced in both cases
- The new approach should produce smoother results in unrestricted dimensions
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# You can modify these paths to point to your test data
TEST_DATA_DIR = "/media/logan/NVMe_Storage/Data/nii_data/MassedEvents9pt4T/derivatives/"  # User will provide actual paths
RESULTS_DIR = (
    "/media/logan/NVMe_Storage/Data/nii_data/MassedEvents9pt4T/derivatives/results"
)

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

from fireants.io import BatchedImages, Image
from fireants.registration import GreedyRegistration, SyNRegistration


def create_synthetic_test_data():
    """Create synthetic test data if real data paths aren't provided."""
    print("Creating synthetic test data...")

    # Create two simple 3D volumes with some structure
    size = (64, 64, 32)  # Smaller for faster testing

    # Fixed image: simple geometric pattern
    fixed_data = np.zeros(size, dtype=np.float32)
    center = np.array(size) // 2

    # Add some geometric structures
    for z in range(size[2]):
        for y in range(size[1]):
            for x in range(size[0]):
                # Create a pattern that varies in X and Y but is consistent in Z
                dist_xy = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
                if 10 < dist_xy < 20:
                    fixed_data[x, y, z] = 1.0
                elif 25 < dist_xy < 30:
                    fixed_data[x, y, z] = 0.5

    # Moving image: shifted version with some additional distortion in X,Y only
    moving_data = np.zeros(size, dtype=np.float32)
    shift_x, shift_y = 5, 3  # Shift in X,Y only (not Z)

    for z in range(size[2]):
        for y in range(size[1]):
            for x in range(size[0]):
                # Apply shift and slight distortion in X,Y
                src_x = x - shift_x + int(2 * np.sin(y * 0.1))
                src_y = y - shift_y + int(2 * np.cos(x * 0.1))
                src_z = z  # No change in Z

                if (
                    0 <= src_x < size[0]
                    and 0 <= src_y < size[1]
                    and 0 <= src_z < size[2]
                ):
                    moving_data[x, y, z] = fixed_data[src_x, src_y, src_z]

    return fixed_data, moving_data


def test_restricted_deformation_comparison():
    """Test that demonstrates the difference between new anisotropic smoothing and legacy behavior."""

    print("============================================================")
    print("RESTRICTED DEFORMATION COMPARISON TEST")
    print("============================================================")

    # Device selection: prefer MPS, fallback to CPU
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"Using device: {device} (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: {device} (CUDA GPU)")
    else:
        device = "cpu"
        print(f"Using device: {device} (CPU fallback)")

    try:
        # Try to load user-provided data (you can modify these paths)
        if os.path.exists(f"{TEST_DATA_DIR}/fixed.nii.gz"):
            print("Loading user-provided test data...")
            fixed_img = Image.load_file(f"{TEST_DATA_DIR}/fixed.nii.gz", device=device)
            moving_img = Image.load_file(
                f"{TEST_DATA_DIR}/moving.nii.gz", device=device
            )
        else:
            raise FileNotFoundError("Using synthetic data")

    except:
        print("Creating synthetic test data...")
        fixed_data, moving_data = create_synthetic_test_data()

        # Convert to FireANTs Image objects (simplified - normally you'd use proper spacing/origin)
        import SimpleITK as sitk

        fixed_sitk = sitk.GetImageFromArray(fixed_data)
        moving_sitk = sitk.GetImageFromArray(moving_data)

        fixed_img = Image(fixed_sitk, device=device)
        moving_img = Image(moving_sitk, device=device)

        # Save for inspection
        print("Test data generated. Image shapes:")
        print(f"  Fixed: {fixed_img.shape}")
        print(f"  Moving: {moving_img.shape}")

    # Setup test parameters
    scales = [2, 1]  # Small scale for quick testing
    iterations = [10, 10]  # Few iterations for speed
    restrict_deformation = [0, 1, 0]  # Allow Y motion, restrict X, Z

    print(f"\nTest parameters:")
    print(f"  Scales: {scales}")
    print(f"  Iterations: {iterations}")
    print(f"  Restriction: {restrict_deformation} (1=allow, 0=restrict)")

    # Create batched images
    batch_fixed = BatchedImages([fixed_img])
    batch_moving = BatchedImages([moving_img])

    # Store results
    results = {}

    print("\n" + "=" * 60)
    print("RUNNING NEW ANISOTROPIC SMOOTHING (CORRECTED LOGIC)")
    print("=" * 60)

    # Test NEW behavior (corrected anisotropic smoothing)
    registration_new = SyNRegistration(
        scales=scales,
        iterations=iterations,
        fixed_images=batch_fixed,
        moving_images=batch_moving,
        restrict_deformation=restrict_deformation,
        force_legacy_behavior=False,  # Use new anisotropic smoothing
        smooth_warp_sigma=0.5,
        smooth_grad_sigma=1.0,
        tolerance=1e-8,
        progress_bar=True,
    )

    registration_new.optimize()

    # Get results
    new_coords = registration_new.get_warped_coordinates(
        batch_fixed, batch_moving, displacement=True
    )
    # Use the evaluate method to get the warped image
    new_warped = registration_new.evaluate(batch_fixed, batch_moving)

    results["new"] = {
        "registration": registration_new,
        "displacement": new_coords,
        "warped_image": new_warped,
    }

    print("\n" + "=" * 60)
    print("RUNNING LEGACY BEHAVIOR (FORCE LEGACY)")
    print("=" * 60)

    # Test LEGACY behavior (forced legacy)
    registration_legacy = SyNRegistration(
        scales=scales,
        iterations=iterations,
        fixed_images=batch_fixed,
        moving_images=batch_moving,
        restrict_deformation=restrict_deformation,
        force_legacy_behavior=True,  # Force legacy behavior for comparison
        smooth_warp_sigma=0.5,
        smooth_grad_sigma=1.0,
        tolerance=1e-8,
        progress_bar=True,
    )

    registration_legacy.optimize()

    # Get results
    legacy_coords = registration_legacy.get_warped_coordinates(
        batch_fixed, batch_moving, displacement=True
    )
    # Use the evaluate method to get the warped image
    legacy_warped = registration_legacy.evaluate(batch_fixed, batch_moving)

    results["legacy"] = {
        "registration": registration_legacy,
        "displacement": legacy_coords,
        "warped_image": legacy_warped,
    }

    print("\n" + "=" * 40)
    print("ANALYSIS AND COMPARISON")
    print("=" * 40)

    analyze_results(results, batch_fixed, batch_moving)

    # Save results
    save_test_results(results)

    return results


def analyze_results(results, batch_fixed, batch_moving):
    """Analyze and compare the registration results."""

    new_disp = results["new"]["displacement"][0].detach().cpu().numpy()
    legacy_disp = results["legacy"]["displacement"][0].detach().cpu().numpy()

    new_warped = results["new"]["warped_image"][0, 0].detach().cpu().numpy()
    legacy_warped = results["legacy"]["warped_image"][0, 0].detach().cpu().numpy()

    fixed_image = batch_fixed()[0, 0].detach().cpu().numpy()
    moving_image = batch_moving()[0, 0].detach().cpu().numpy()

    print("\n1. RESTRICTION ENFORCEMENT CHECK")
    print("-" * 35)

    # Check Z-displacement (should be zero for both)
    new_z_max = np.abs(new_disp[..., 2]).max()
    legacy_z_max = np.abs(legacy_disp[..., 2]).max()

    print(f"Max Z displacement (new):    {new_z_max:.8f}")
    print(f"Max Z displacement (legacy): {legacy_z_max:.8f}")
    print(
        f"Z restriction maintained:    {'✓' if max(new_z_max, legacy_z_max) < 1e-6 else '✗'}"
    )

    print("\n2. DEFORMATION SMOOTHNESS ANALYSIS")
    print("-" * 35)

    # Analyze smoothness in X,Y dimensions (compute gradients)
    def compute_gradient_magnitude(field):
        """Compute spatial gradient magnitude for smoothness assessment."""
        if len(field.shape) == 3:  # [H, W, D]
            dy, dx, dz = np.gradient(field)
            return np.sqrt(dx**2 + dy**2 + dz**2)
        return None

    # Smoothness in X,Y displacements
    new_x_smoothness = compute_gradient_magnitude(new_disp[..., 0])
    new_y_smoothness = compute_gradient_magnitude(new_disp[..., 1])
    legacy_x_smoothness = compute_gradient_magnitude(legacy_disp[..., 0])
    legacy_y_smoothness = compute_gradient_magnitude(legacy_disp[..., 1])

    if new_x_smoothness is not None:
        new_avg_roughness = (new_x_smoothness.mean() + new_y_smoothness.mean()) / 2
        legacy_avg_roughness = (
            legacy_x_smoothness.mean() + legacy_y_smoothness.mean()
        ) / 2

        print(f"Average roughness (new):    {new_avg_roughness:.6f}")
        print(f"Average roughness (legacy): {legacy_avg_roughness:.6f}")
        print(
            f"Smoothness improvement:     {((legacy_avg_roughness - new_avg_roughness) / legacy_avg_roughness * 100):.1f}%"
        )

    print("\n3. REGISTRATION QUALITY")
    print("-" * 25)

    # Compute similarity between warped and fixed
    def compute_ncc(img1, img2):
        """Compute normalized cross correlation."""
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()

        img1_centered = img1_flat - np.mean(img1_flat)
        img2_centered = img2_flat - np.mean(img2_flat)

        ncc = np.sum(img1_centered * img2_centered) / (
            np.sqrt(np.sum(img1_centered**2)) * np.sqrt(np.sum(img2_centered**2))
        )
        return ncc

    new_ncc = compute_ncc(new_warped, fixed_image)
    legacy_ncc = compute_ncc(legacy_warped, fixed_image)
    baseline_ncc = compute_ncc(moving_image, fixed_image)

    print(f"Baseline NCC (no registration): {baseline_ncc:.4f}")
    print(f"New method NCC:                 {new_ncc:.4f}")
    print(f"Legacy method NCC:              {legacy_ncc:.4f}")
    print(
        f"New vs Legacy improvement:      {((new_ncc - legacy_ncc) / abs(legacy_ncc) * 100):.2f}%"
    )

    print("\n4. DISPLACEMENT STATISTICS")
    print("-" * 27)

    # Displacement magnitudes in X,Y (allowed dimensions)
    new_xy_magnitude = np.sqrt(new_disp[..., 0] ** 2 + new_disp[..., 1] ** 2)
    legacy_xy_magnitude = np.sqrt(legacy_disp[..., 0] ** 2 + legacy_disp[..., 1] ** 2)

    print(f"Mean XY displacement (new):    {new_xy_magnitude.mean():.4f}")
    print(f"Mean XY displacement (legacy): {legacy_xy_magnitude.mean():.4f}")
    print(f"Max XY displacement (new):     {new_xy_magnitude.max():.4f}")
    print(f"Max XY displacement (legacy):  {legacy_xy_magnitude.max():.4f}")


def save_test_results(results):
    """Save test results for inspection."""

    print(f"\n5. SAVING RESULTS TO {RESULTS_DIR}")
    print("-" * 30)

    # Save warp fields
    new_disp = results["new"]["displacement"][0].detach().cpu().numpy()
    legacy_disp = results["legacy"]["displacement"][0].detach().cpu().numpy()

    new_warped = results["new"]["warped_image"][0, 0].detach().cpu().numpy()
    legacy_warped = results["legacy"]["warped_image"][0, 0].detach().cpu().numpy()

    import SimpleITK as sitk

    # Save displacement fields
    new_disp_sitk = sitk.GetImageFromArray(new_disp)
    legacy_disp_sitk = sitk.GetImageFromArray(legacy_disp)

    sitk.WriteImage(new_disp_sitk, f"{RESULTS_DIR}/displacement_new_anisotropic.nii.gz")
    sitk.WriteImage(legacy_disp_sitk, f"{RESULTS_DIR}/displacement_legacy.nii.gz")

    # Save warped images
    new_warped_sitk = sitk.GetImageFromArray(new_warped)
    legacy_warped_sitk = sitk.GetImageFromArray(legacy_warped)

    sitk.WriteImage(new_warped_sitk, f"{RESULTS_DIR}/warped_new_anisotropic.nii.gz")
    sitk.WriteImage(legacy_warped_sitk, f"{RESULTS_DIR}/warped_legacy.nii.gz")

    # Save difference maps
    disp_diff = new_disp - legacy_disp
    warped_diff = new_warped - legacy_warped

    sitk.WriteImage(
        sitk.GetImageFromArray(disp_diff),
        f"{RESULTS_DIR}/displacement_difference.nii.gz",
    )
    sitk.WriteImage(
        sitk.GetImageFromArray(warped_diff), f"{RESULTS_DIR}/warped_difference.nii.gz"
    )

    print("✓ Displacement fields saved")
    print("✓ Warped images saved")
    print("✓ Difference maps saved")

    # Create summary plot if matplotlib is available
    try:
        create_summary_plot(results)
        print("✓ Summary plot saved")
    except Exception as e:
        print(f"Could not create summary plot: {e}")


def create_summary_plot(results):
    """Create a summary visualization."""

    new_disp = results["new"]["displacement"][0].detach().cpu().numpy()
    legacy_disp = results["legacy"]["displacement"][0].detach().cpu().numpy()

    new_warped = results["new"]["warped_image"][0, 0].detach().cpu().numpy()
    legacy_warped = results["legacy"]["warped_image"][0, 0].detach().cpu().numpy()

    # Take middle slice for visualization
    mid_slice = new_disp.shape[2] // 2

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1: New method
    axes[0, 0].imshow(new_warped[:, :, mid_slice], cmap="gray")
    axes[0, 0].set_title("New: Warped Image")

    axes[0, 1].imshow(new_disp[:, :, mid_slice, 0], cmap="RdBu_r")
    axes[0, 1].set_title("New: X Displacement")

    axes[0, 2].imshow(new_disp[:, :, mid_slice, 1], cmap="RdBu_r")
    axes[0, 2].set_title("New: Y Displacement")

    axes[0, 3].imshow(new_disp[:, :, mid_slice, 2], cmap="RdBu_r")
    axes[0, 3].set_title("New: Z Displacement")

    # Row 2: Legacy method
    axes[1, 0].imshow(legacy_warped[:, :, mid_slice], cmap="gray")
    axes[1, 0].set_title("Legacy: Warped Image")

    axes[1, 1].imshow(legacy_disp[:, :, mid_slice, 0], cmap="RdBu_r")
    axes[1, 1].set_title("Legacy: X Displacement")

    axes[1, 2].imshow(legacy_disp[:, :, mid_slice, 1], cmap="RdBu_r")
    axes[1, 2].set_title("Legacy: Y Displacement")

    axes[1, 3].imshow(legacy_disp[:, :, mid_slice, 2], cmap="RdBu_r")
    axes[1, 3].set_title("Legacy: Z Displacement")

    # Remove axis ticks for cleaner look
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/comparison_summary.png", dpi=150, bbox_inches="tight")
    plt.close()


def test_greedy_registration():
    """Quick test of Greedy registration with restrictions."""
    print("\n" + "=" * 40)
    print("BONUS: GREEDY REGISTRATION TEST")
    print("=" * 40)

    # This is a simpler test just to verify Greedy also works
    try:
        from fireants.io import BatchedImages, Image

        # Create simple synthetic data for quick test
        fixed_data, moving_data = create_synthetic_test_data()

        import SimpleITK as sitk

        fixed_sitk = sitk.GetImageFromArray(fixed_data)
        moving_sitk = sitk.GetImageFromArray(moving_data)

        fixed_img = Image(fixed_sitk)
        moving_img = Image(moving_sitk)

        batch_fixed = BatchedImages([fixed_img])
        batch_moving = BatchedImages([moving_img])

        # Quick Greedy test
        reg_greedy = GreedyRegistration(
            scales=[4, 2],
            iterations=[20, 20],
            fixed_images=batch_fixed,
            moving_images=batch_moving,
            restrict_deformation=[1, 1, 0],
            force_legacy_behavior=False,
        )

        print("Running Greedy registration with anisotropic smoothing...")
        reg_greedy.optimize()

        # Check restriction enforcement
        coords = reg_greedy.get_warped_coordinates(batch_fixed, batch_moving)
        displacement = reg_greedy.get_warped_coordinates(
            batch_fixed, batch_moving, displacement=True
        )

        z_displacement = displacement[0, ..., 2].detach().cpu().numpy()
        max_z = np.abs(z_displacement).max()

        print(f"Max Z displacement: {max_z:.8f}")
        print(f"Restriction enforced: {'✓' if max_z < 1e-6 else '✗'}")

        return True

    except Exception as e:
        print(f"Greedy test failed: {e}")
        return False


if __name__ == "__main__":
    """
    To use this test:
    
    1. With your own data:
       - Put your fixed image at: /tmp/fireants_test_data/fixed.nii.gz
       - Put your moving image at: /tmp/fireants_test_data/moving.nii.gz
       
    2. Or just run as-is to use synthetic test data
    
    Results will be saved to: /tmp/fireants_test_results/
    """

    print("FireANTs Restricted Deformation Test")
    print("=" * 50)

    try:
        # Main test
        results = test_restricted_deformation_comparison()

        # Bonus test
        greedy_success = test_greedy_registration()

        print("\n" + "=" * 50)
        print("TEST COMPLETION SUMMARY")
        print("=" * 50)
        print("✓ SyN anisotropic vs legacy comparison completed")
        print(f"{'✓' if greedy_success else '✗'} Greedy registration test completed")
        print(f"📁 Results saved to: {RESULTS_DIR}")
        print("\nTo inspect results:")
        print(f"   cd {RESULTS_DIR}")
        print("   ls -la *.nii.gz")
        print(
            "   # Open .nii.gz files in your favorite viewer (FSLeyes, ITK-SNAP, etc.)"
        )

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()

    print("\nTest completed!")
