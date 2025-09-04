#!/usr/bin/env python3
"""
Analyze and visualize pseudo-labeling experiment results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_results(csv_file: str) -> pd.DataFrame:
    """Load experiment results from CSV."""
    df = pd.read_csv(csv_file)
    return df

def print_summary_table(df: pd.DataFrame):
    """Print a formatted summary table."""
    print("\n" + "="*100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*100)
    
    # Sort by F1 score descending
    df_sorted = df.sort_values('f1', ascending=False)
    print(df_sorted.to_string(index=False, float_format='%.4f'))
    
    # Best results
    if len(df) > 1:
        best = df_sorted.iloc[0]
        print(f"\n🏆 Best F1 Score: {best['f1']:.4f}")
        print(f"   Run ID: {best['run_id']}")
        print(f"   Config: K={best['K']}, N={best['N']}, variant={best['variant']}")
        
        # Compare with baseline if available
        baseline = df[df['run_id'] == 'baseline']
        if not baseline.empty:
            baseline_f1 = baseline['f1'].iloc[0]
            improvement = best['f1'] - baseline_f1
            print(f"\n📈 Improvement over baseline:")
            print(f"   Baseline F1: {baseline_f1:.4f}")
            print(f"   Best F1: {best['f1']:.4f}")
            print(f"   Improvement: {improvement:+.4f} ({improvement/baseline_f1*100:+.1f}%)")

def create_visualizations(df: pd.DataFrame, output_dir: str = "plots"):
    """Create visualization plots."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Filter out baseline for certain plots
    df_exp = df[df['run_id'] != 'baseline'].copy()
    
    if df_exp.empty:
        print("No experiment data to plot (only baseline found)")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. F1 Score by K and N (faceted by variant)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('F1 Score by Number of Pseudo-Images (K) and Agreement Level (N)', fontsize=14)
    
    for i, variant in enumerate(['no-remove', 'remove-unknown']):
        df_variant = df_exp[df_exp['variant'] == variant]
        
        # Create pivot table for heatmap
        pivot = df_variant.pivot(index='N', columns='K', values='f1')
        
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', 
                   ax=axes[i], cbar_kws={'label': 'F1 Score'})
        axes[i].set_title(f'Variant: {variant}')
        axes[i].set_xlabel('Number of Pseudo-Images (K)')
        axes[i].set_ylabel('Agreement Level (N)')
    
    plt.tight_layout()
    plt.savefig(output_path / 'f1_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bar plot comparing variants
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group by K and N, compare variants
    df_pivot = df_exp.pivot_table(index=['K', 'N'], columns='variant', values='f1')
    df_pivot.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_title('F1 Score Comparison: Original vs Gray-Painted Images')
    ax.set_xlabel('(K, N) Configuration')
    ax.set_ylabel('F1 Score')
    ax.legend(title='Variant')
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / 'variant_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Line plot showing effect of K for each N
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Effect of Number of Pseudo-Images (K) on F1 Score', fontsize=14)
    
    for i, variant in enumerate(['no-remove', 'remove-unknown']):
        df_variant = df_exp[df_exp['variant'] == variant]
        
        for n in df_variant['N'].unique():
            df_n = df_variant[df_variant['N'] == n]
            axes[i].plot(df_n['K'], df_n['f1'], marker='o', linewidth=2, label=f'N={n}')
        
        axes[i].set_title(f'Variant: {variant}')
        axes[i].set_xlabel('Number of Pseudo-Images (K)')
        axes[i].set_ylabel('F1 Score')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'k_effect.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Overall comparison with baseline
    if 'baseline' in df['run_id'].values:
        baseline_f1 = df[df['run_id'] == 'baseline']['f1'].iloc[0]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a comparison plot
        exp_f1s = df_exp['f1'].values
        exp_labels = [f"K{row['K']}_N{row['N']}_{row['variant']}" for _, row in df_exp.iterrows()]
        
        colors = ['green' if f1 > baseline_f1 else 'red' for f1 in exp_f1s]
        bars = ax.bar(range(len(exp_f1s)), exp_f1s, color=colors, alpha=0.7)
        
        # Add baseline line
        ax.axhline(y=baseline_f1, color='blue', linestyle='--', linewidth=2, label=f'Baseline (F1={baseline_f1:.3f})')
        
        ax.set_title('All Experiments vs Baseline')
        ax.set_xlabel('Experiment Configuration')
        ax.set_ylabel('F1 Score')
        ax.set_xticks(range(len(exp_labels)))
        ax.set_xticklabels(exp_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, f1 in zip(bars, exp_f1s):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{f1:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path / 'baseline_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\n📊 Plots saved to {output_path}/")
    print("   - f1_heatmap.png: F1 scores by K and N for each variant")
    print("   - variant_comparison.png: Comparison between original and gray-painted images")
    print("   - k_effect.png: Effect of number of pseudo-images")
    if 'baseline' in df['run_id'].values:
        print("   - baseline_comparison.png: All experiments vs baseline")

def analyze_trends(df: pd.DataFrame):
    """Analyze trends in the results."""
    print("\n" + "="*60)
    print("TREND ANALYSIS")
    print("="*60)
    
    df_exp = df[df['run_id'] != 'baseline'].copy()
    
    if df_exp.empty:
        print("No experiment data to analyze")
        return
    
    # Effect of K (number of pseudo-images)
    print("\n📈 Effect of Number of Pseudo-Images (K):")
    k_effect = df_exp.groupby(['K', 'variant'])['f1'].mean().unstack()
    print(k_effect.round(4))
    
    # Effect of N (agreement level)
    print("\n🤝 Effect of Agreement Level (N):")
    n_effect = df_exp.groupby(['N', 'variant'])['f1'].mean().unstack()
    print(n_effect.round(4))
    
    # Best configurations
    print("\n🏆 Top 5 Configurations:")
    top5 = df_exp.nlargest(5, 'f1')[['run_id', 'K', 'N', 'variant', 'f1']]
    print(top5.to_string(index=False, float_format='%.4f'))
    
    # Variant comparison
    print("\n🔄 Variant Comparison (Average F1):")
    variant_avg = df_exp.groupby('variant')['f1'].agg(['mean', 'std', 'min', 'max'])
    print(variant_avg.round(4))

def main():
    parser = argparse.ArgumentParser(description="Analyze pseudo-labeling experiment results")
    parser.add_argument("--csv", default="experiment_results.csv", help="Results CSV file")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not Path(args.csv).exists():
        print(f"❌ Results file not found: {args.csv}")
        print("Run the experiments first with: python run_experiments.py")
        return
    
    # Load and analyze results
    df = load_results(args.csv)
    
    print(f"📊 Loaded {len(df)} experiment results from {args.csv}")
    
    # Print summary
    print_summary_table(df)
    
    # Analyze trends
    analyze_trends(df)
    
    # Create visualizations
    if not args.no_plots:
        try:
            create_visualizations(df, args.output_dir)
        except ImportError:
            print("\n⚠️  Matplotlib/Seaborn not available. Skipping plots.")
            print("   Install with: pip install matplotlib seaborn")

if __name__ == "__main__":
    main()
