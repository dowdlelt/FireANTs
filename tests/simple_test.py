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
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# You can modify these paths to point to your test data
TEST_DATA_DIR = "/tmp/fireants_test_data"  # User will provide actual paths
RESULTS_DIR = "/tmp/fireants_test_results"

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

from fireants.io import BatchedImages, Image
from fireants.registration import SyNRegistration, GreedyRegistration


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
                dist_xy = np.sqrt((x - center[0])**2 + (y - center[1])**2)
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
                
                if (0 <= src_x < size[0] and 0 <= src_y < size[1] and 
                    0 <= src_z < size[2]):
                    moving_data[x, y, z] = fixed_data[src_x, src_y, src_z]
    
    return fixed_data, moving_data


def test_restricted_deformation_comparison():
    """Test comparing new anisotropic smoothing vs legacy behavior."""
    
    print("=" * 60)
    print("RESTRICTED DEFORMATION COMPARISON TEST")
    print("=" * 60)
    
    # Load or create test data
    try:
        # Try to load user-provided data (you can modify these paths)
        if os.path.exists(f"{TEST_DATA_DIR}/fixed.nii.gz"):
            print("Loading user-provided test data...")
            fixed_img = Image.load_file(f"{TEST_DATA_DIR}/fixed.nii.gz")
            moving_img = Image.load_file(f"{TEST_DATA_DIR}/moving.nii.gz")
        else:
            raise FileNotFoundError("Using synthetic data")
            
    except:
        print("Creating synthetic test data...")
        fixed_data, moving_data = create_synthetic_test_data()
        
        # Convert to FireANTs Image objects (simplified - normally you'd use proper spacing/origin)
        import SimpleITK as sitk
        fixed_sitk = sitk.GetImageFromArray(fixed_data)
        moving_sitk = sitk.GetImageFromArray(moving_data)
        
        fixed_img = Image(fixed_sitk)
        moving_img = Image(moving_sitk)
        
        # Save for inspection
        sitk.WriteImage(fixed_sitk, f"{RESULTS_DIR}/synthetic_fixed.nii.gz")
        sitk.WriteImage(moving_sitk, f"{RESULTS_DIR}/synthetic_moving.nii.gz")
    
    # Create batched images
    batch_fixed = BatchedImages([fixed_img])
    batch_moving = BatchedImages([moving_img])
    
    print(f"Image dimensions: {fixed_img.shape}")
    print(f"Testing restriction pattern: [1,1,0] (X,Y allowed, Z restricted)")
    
    # Test parameters - using smaller/faster settings for testing
    test_params = {
        'scales': [4, 2, 1],
        'iterations': [50, 50, 50],  # Reduced for faster testing
        'fixed_images': batch_fixed,
        'moving_images': batch_moving,
        'cc_kernel_size': 3,
        'smooth_warp_sigma': 0.25,      # Moderate smoothing
        'smooth_grad_sigma': 0.25,      # Moderate gradient smoothing  
        'tolerance': 1e-6,
        'optimizer_lr': 0.1,
        'restrict_deformation': [1, 1, 0],  # Allow X,Y motion, restrict Z
    }
    
    results = {}
    
    # Test 1: New anisotropic smoothing behavior
    print("\n" + "-" * 40)
    print("TEST 1: New Anisotropic Smoothing")
    print("-" * 40)
    
    print("Creating SyN registration with anisotropic smoothing...")
    reg_new = SyNRegistration(**test_params, force_legacy_behavior=False)
    
    print(f"has_partial_restriction: {reg_new.has_partial_restriction}")
    print(f"force_legacy_behavior: {getattr(reg_new, 'force_legacy_behavior', 'Not set')}")
    print("Starting optimization...")
    
    reg_new.optimize(save_transformed=False)
    
    # Get final warp field and transformed image
    new_coords = reg_new.get_warped_coordinates(batch_fixed, batch_moving)
    new_warp_displacement = reg_new.get_warped_coordinates(batch_fixed, batch_moving, displacement=True)
    
    # Apply transformation to get warped image
    import torch.nn.functional as F
    new_warped = F.grid_sample(batch_moving(), new_coords, mode='bilinear', align_corners=True)
    
    results['new'] = {
        'coords': new_coords,
        'displacement': new_warp_displacement,
        'warped_image': new_warped,
        'registration': reg_new
    }
    
    # Test 2: Legacy behavior (forced)
    print("\n" + "-" * 40)  
    print("TEST 2: Legacy Behavior (Forced)")
    print("-" * 40)
    
    print("Creating SyN registration with forced legacy behavior...")
    reg_legacy = SyNRegistration(**test_params, force_legacy_behavior=True)
    
    print(f"has_partial_restriction: {reg_legacy.has_partial_restriction}")
    print(f"force_legacy_behavior: {getattr(reg_legacy, 'force_legacy_behavior', 'Not set')}")
    print("Starting optimization...")
    
    reg_legacy.optimize(save_transformed=False)
    
    # Get final warp field and transformed image
    legacy_coords = reg_legacy.get_warped_coordinates(batch_fixed, batch_moving)
    legacy_warp_displacement = reg_legacy.get_warped_coordinates(batch_fixed, batch_moving, displacement=True)
    
    # Apply transformation to get warped image
    legacy_warped = F.grid_sample(batch_moving(), legacy_coords, mode='bilinear', align_corners=True)
    
    results['legacy'] = {
        'coords': legacy_coords,
        'displacement': legacy_warp_displacement,
        'warped_image': legacy_warped,
        'registration': reg_legacy
    }
    
    # Analysis and comparison
    print("\n" + "=" * 40)
    print("ANALYSIS AND COMPARISON")
    print("=" * 40)
    
    analyze_results(results, batch_fixed, batch_moving)
    
    # Save results
    save_test_results(results)
    
    return results


def analyze_results(results, batch_fixed, batch_moving):
    """Analyze and compare the registration results."""
    
    new_disp = results['new']['displacement'][0].detach().cpu().numpy()
    legacy_disp = results['legacy']['displacement'][0].detach().cpu().numpy()
    
    new_warped = results['new']['warped_image'][0, 0].detach().cpu().numpy()
    legacy_warped = results['legacy']['warped_image'][0, 0].detach().cpu().numpy()
    
    fixed_image = batch_fixed()[0, 0].detach().cpu().numpy()
    moving_image = batch_moving()[0, 0].detach().cpu().numpy()
    
    print("\n1. RESTRICTION ENFORCEMENT CHECK")
    print("-" * 35)
    
    # Check Z-displacement (should be zero for both)
    new_z_max = np.abs(new_disp[..., 2]).max()
    legacy_z_max = np.abs(legacy_disp[..., 2]).max()
    
    print(f"Max Z displacement (new):    {new_z_max:.8f}")
    print(f"Max Z displacement (legacy): {legacy_z_max:.8f}")
    print(f"Z restriction maintained:    {'✓' if max(new_z_max, legacy_z_max) < 1e-6 else '✗'}")
    
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
        legacy_avg_roughness = (legacy_x_smoothness.mean() + legacy_y_smoothness.mean()) / 2
        
        print(f"Average roughness (new):    {new_avg_roughness:.6f}")
        print(f"Average roughness (legacy): {legacy_avg_roughness:.6f}")
        print(f"Smoothness improvement:     {((legacy_avg_roughness - new_avg_roughness) / legacy_avg_roughness * 100):.1f}%")
    
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
    print(f"New vs Legacy improvement:      {((new_ncc - legacy_ncc) / abs(legacy_ncc) * 100):.2f}%")
    
    print("\n4. DISPLACEMENT STATISTICS")
    print("-" * 27)
    
    # Displacement magnitudes in X,Y (allowed dimensions)
    new_xy_magnitude = np.sqrt(new_disp[..., 0]**2 + new_disp[..., 1]**2)
    legacy_xy_magnitude = np.sqrt(legacy_disp[..., 0]**2 + legacy_disp[..., 1]**2)
    
    print(f"Mean XY displacement (new):    {new_xy_magnitude.mean():.4f}")
    print(f"Mean XY displacement (legacy): {legacy_xy_magnitude.mean():.4f}")
    print(f"Max XY displacement (new):     {new_xy_magnitude.max():.4f}")
    print(f"Max XY displacement (legacy):  {legacy_xy_magnitude.max():.4f}")


def save_test_results(results):
    """Save test results for inspection."""
    
    print(f"\n5. SAVING RESULTS TO {RESULTS_DIR}")
    print("-" * 30)
    
    # Save warp fields
    new_disp = results['new']['displacement'][0].detach().cpu().numpy()
    legacy_disp = results['legacy']['displacement'][0].detach().cpu().numpy()
    
    new_warped = results['new']['warped_image'][0, 0].detach().cpu().numpy()
    legacy_warped = results['legacy']['warped_image'][0, 0].detach().cpu().numpy()
    
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
    
    sitk.WriteImage(sitk.GetImageFromArray(disp_diff), f"{RESULTS_DIR}/displacement_difference.nii.gz")
    sitk.WriteImage(sitk.GetImageFromArray(warped_diff), f"{RESULTS_DIR}/warped_difference.nii.gz")
    
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
    
    new_disp = results['new']['displacement'][0].detach().cpu().numpy()
    legacy_disp = results['legacy']['displacement'][0].detach().cpu().numpy()
    
    new_warped = results['new']['warped_image'][0, 0].detach().cpu().numpy()
    legacy_warped = results['legacy']['warped_image'][0, 0].detach().cpu().numpy()
    
    # Take middle slice for visualization
    mid_slice = new_disp.shape[2] // 2
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: New method
    axes[0, 0].imshow(new_warped[:, :, mid_slice], cmap='gray')
    axes[0, 0].set_title('New: Warped Image')
    
    axes[0, 1].imshow(new_disp[:, :, mid_slice, 0], cmap='RdBu_r')
    axes[0, 1].set_title('New: X Displacement')
    
    axes[0, 2].imshow(new_disp[:, :, mid_slice, 1], cmap='RdBu_r')
    axes[0, 2].set_title('New: Y Displacement')
    
    axes[0, 3].imshow(new_disp[:, :, mid_slice, 2], cmap='RdBu_r')
    axes[0, 3].set_title('New: Z Displacement')
    
    # Row 2: Legacy method
    axes[1, 0].imshow(legacy_warped[:, :, mid_slice], cmap='gray')
    axes[1, 0].set_title('Legacy: Warped Image')
    
    axes[1, 1].imshow(legacy_disp[:, :, mid_slice, 0], cmap='RdBu_r')
    axes[1, 1].set_title('Legacy: X Displacement')
    
    axes[1, 2].imshow(legacy_disp[:, :, mid_slice, 1], cmap='RdBu_r')
    axes[1, 2].set_title('Legacy: Y Displacement')
    
    axes[1, 3].imshow(legacy_disp[:, :, mid_slice, 2], cmap='RdBu_r')
    axes[1, 3].set_title('Legacy: Z Displacement')
    
    # Remove axis ticks for cleaner look
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/comparison_summary.png", dpi=150, bbox_inches='tight')
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
            force_legacy_behavior=False
        )
        
        print("Running Greedy registration with anisotropic smoothing...")
        reg_greedy.optimize()
        
        # Check restriction enforcement
        coords = reg_greedy.get_warped_coordinates(batch_fixed, batch_moving)
        displacement = reg_greedy.get_warped_coordinates(batch_fixed, batch_moving, displacement=True)
        
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
        print("   # Open .nii.gz files in your favorite viewer (FSLeyes, ITK-SNAP, etc.)")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    print("\nTest completed!")
