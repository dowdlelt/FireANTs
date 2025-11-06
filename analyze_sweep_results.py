#!/usr/bin/env python3
"""Analyze parameter sweep results and find optimal parameters."""

import pandas as pd
import sys
from pathlib import Path

def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'param_sweep_results/sweep_results.csv'

    try:
        df = pd.read_csv(csv_file)

        # Remove error rows
        df = df[df['ls'] != 'ERROR'].copy()

        # Convert metrics to numeric
        metric_cols = ['ls', 'sp', 'mi', 'crM', 'nmi', 'je', 'hel', 'crA', 'crU', 'lss', 'lpc', 'lpa', 'lpc_plus', 'lpa_plus']
        for col in metric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        print("\n" + "="*80)
        print("BEST PARAMETERS FOR EACH METRIC")
        print("="*80)

        # Define optimization direction for each metric
        # Higher is better: crM, nmi, crA, crU, lpc, lpc_plus, lpa_plus
        # Lower is better: ls, sp, mi, je, hel, lss, lpa

        metrics_info = {
            'ls': ('Least Squares', 'min'),
            'sp': ('Spearman Correlation', 'max'),
            'mi': ('Mutual Information', 'max'),
            'crM': ('Correlation Ratio (M)', 'max'),
            'nmi': ('Normalized MI', 'max'),
            'je': ('Joint Entropy', 'min'),
            'hel': ('Hellinger', 'max'),
            'crA': ('Correlation Ratio (A)', 'max'),
            'crU': ('Correlation Ratio (U)', 'max'),
            'lss': ('Least Squares Scaled', 'min'),
            'lpc': ('Local Pearson Correlation', 'max'),
            'lpa': ('Local Pearson Abs', 'max'),
            'lpc_plus': ('Local Pearson Clipped', 'max'),
            'lpa_plus': ('Local Pearson Abs Clipped', 'max'),
        }

        for metric, (name, opt_dir) in metrics_info.items():
            if metric not in df.columns or df[metric].isna().all():
                continue

            if opt_dir == 'min':
                idx = df[metric].idxmin()
                best_val = df.loc[idx, metric]
                worst_val = df[metric].max()
            else:
                idx = df[metric].idxmax()
                best_val = df.loc[idx, metric]
                worst_val = df[metric].min()

            best_lr = df.loc[idx, 'lr']
            best_sw = df.loc[idx, 'smooth_warp']
            best_sg = df.loc[idx, 'smooth_grad']

            print(f"\n{name} ({metric}):")
            print(f"  Best:  {best_val:.6f} @ lr={best_lr}, smooth_warp={best_sw}, smooth_grad={best_sg}")
            print(f"  Range: [{worst_val:.6f}, {best_val:.6f}]")

        # Find parameter combo that appears most frequently in top results
        print("\n" + "="*80)
        print("PARAMETER COMBINATIONS APPEARING IN TOP 5 FOR MOST METRICS")
        print("="*80)

        top_combos = {}
        for metric in metric_cols:
            if metric not in df.columns or df[metric].isna().all():
                continue

            opt_dir = metrics_info.get(metric, (None, 'max'))[1]
            if opt_dir == 'min':
                top_5 = df.nsmallest(5, metric)
            else:
                top_5 = df.nlargest(5, metric)

            for _, row in top_5.iterrows():
                combo = (row['lr'], row['smooth_warp'], row['smooth_grad'])
                if combo not in top_combos:
                    top_combos[combo] = 0
                top_combos[combo] += 1

        sorted_combos = sorted(top_combos.items(), key=lambda x: x[1], reverse=True)

        print("\nTop parameter combinations (by frequency in top 5):")
        for i, ((lr, sw, sg), count) in enumerate(sorted_combos[:10]):
            print(f"  {i+1}. lr={lr}, smooth_warp={sw}, smooth_grad={sg} (appears in {count} metrics)")

        print("\n" + "="*80)

    except Exception as e:
        print(f"Error analyzing results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
