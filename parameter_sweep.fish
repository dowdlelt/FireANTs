#!/usr/bin/env fish
# Parameter sweep for SyN registration optimization
# Evaluates different learning rates, smooth_warp_sigma, and smooth_grad_sigma
# by measuring similarity metrics with 3dAllineate

set -g FIREANTS_CLI ~/Dropbox/Resources/code/packages/fireants/cli/fireantsRegistration
set -g FIXED_IMAGE step01_pad_ses-01_fmap-01_blipRev.nii.gz
set -g MOVING_IMAGE step01_pad_ses-01_fmap-01_blipFor.nii.gz
set -g OUTPUT_DIR param_sweep_results
set -g RESULTS_CSV $OUTPUT_DIR/sweep_results.csv

# Parameter ranges
set -g LEARNING_RATES 0.01 0.05 0.1 0.2
set -g SMOOTH_WARP_SIGMAS 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5
set -g SMOOTH_GRAD_SIGMAS 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

# Create output directory
mkdir -p $OUTPUT_DIR

# Initialize results CSV
echo "lr,smooth_warp,smooth_grad,ls,sp,mi,crM,nmi,je,hel,crA,crU,lss,lpc,lpa,lpc_plus,lpa_plus" > $RESULTS_CSV

# Function to parse 3dAllineate output
function parse_allineate_output
    set output_file $argv[1]

    # Extract metrics - they appear after "++ allcost output: init #0"
    set ls (grep "ls   =" $output_file | awk '{print $3}')
    set sp (grep "sp   =" $output_file | awk '{print $3}')
    set mi (grep "mi   =" $output_file | awk '{print $3}')
    set crM (grep "crM  =" $output_file | awk '{print $3}')
    set nmi (grep "nmi  =" $output_file | awk '{print $3}')
    set je (grep "je   =" $output_file | awk '{print $3}')
    set hel (grep "hel  =" $output_file | awk '{print $3}')
    set crA (grep "crA  =" $output_file | awk '{print $3}')
    set crU (grep "crU  =" $output_file | awk '{print $3}')
    set lss (grep "lss  =" $output_file | awk '{print $3}')
    set lpc (grep "lpc  =" $output_file | awk '{print $3}')
    set lpa (grep "lpa  =" $output_file | awk '{print $3}')
    set lpc_plus (grep "lpc+ =" $output_file | awk '{print $3}')
    set lpa_plus (grep "lpa+ =" $output_file | awk '{print $3}')

    echo "$ls,$sp,$mi,$crM,$nmi,$je,$hel,$crA,$crU,$lss,$lpc,$lpa,$lpc_plus,$lpa_plus"
end

# Main parameter sweep loop
set total_runs (math (count $LEARNING_RATES) "*" (count $SMOOTH_WARP_SIGMAS) "*" (count $SMOOTH_GRAD_SIGMAS))
set current_run 0

echo "Starting parameter sweep: $total_runs total combinations"
echo "=================================================="

for lr in $LEARNING_RATES
    for sw in $SMOOTH_WARP_SIGMAS
        for sg in $SMOOTH_GRAD_SIGMAS
            set current_run (math $current_run + 1)

            # Create descriptive output prefix
            set output_prefix $OUTPUT_DIR/syn_lr{$lr}_sw{$sw}_sg{$sg}
            set warped_output {$output_prefix}_warped.nii.gz
            set warp_output {$output_prefix}0Warp.nii.gz
            set allineate_output {$output_prefix}_allineate.txt

            echo ""
            echo "[$current_run/$total_runs] Testing lr=$lr, smooth_warp=$sw, smooth_grad=$sg"

            # Run SyN registration
            echo "  Running registration..."
            python $FIREANTS_CLI \
                --output $output_prefix \
                --transform SyN[$lr] \
                --metric CC[$FIXED_IMAGE,$MOVING_IMAGE,7] \
                --convergence [400x400x400x400,1e-6,10] \
                --shrink-factors 8x4x2x1 \
                --restrict-deformation 0x1x0 \
                --smooth_warp_sigma $sw \
                --smooth_grad_sigma $sg \
                --winsorize-image-intensities 0.005,0.995 \
                > {$output_prefix}_registration.log 2>&1

            # Check if registration succeeded
            if test $status -ne 0
                echo "  ERROR: Registration failed!"
                echo "$lr,$sw,$sg,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> $RESULTS_CSV
                continue
            end

            # Check if warped output exists
            if not test -f $warped_output
                echo "  ERROR: Warped output not found!"
                echo "$lr,$sw,$sg,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> $RESULTS_CSV
                continue
            end

            # Run 3dAllineate to evaluate similarity
            echo "  Evaluating similarity metrics..."
            3dAllineate \
                -base $FIXED_IMAGE \
                -source $warped_output \
                -allcostX \
                -automask \
                -autoweight \
                > $allineate_output 2>&1

            # Parse metrics
            set metrics (parse_allineate_output $allineate_output)

            # Save to CSV
            echo "$lr,$sw,$sg,$metrics" >> $RESULTS_CSV

            # Clean up intermediate files to save space (keep only warp and warped image)
            rm -f {$output_prefix}_registration.log
            rm -f {$output_prefix}0InverseWarp.nii.gz  # We don't need inverse for this sweep

            echo "  ✓ Completed"
        end
    end
end

echo ""
echo "=================================================="
echo "Parameter sweep complete!"
echo "Results saved to: $RESULTS_CSV"
echo ""

# Analyze results - find best parameters for each metric
echo "Analyzing results..."
set ANALYSIS_SCRIPT (dirname (status -f))/analyze_sweep_results.py
python3 $ANALYSIS_SCRIPT $RESULTS_CSV

echo ""
echo "Analysis complete!"
echo "Check '$OUTPUT_DIR' for all warped images and warps"
