import numpy as np
import matplotlib.pyplot as plt
import os
from src.config import METRICS_OUTPUT_DIR, LANDMARKS

def calculate_precision_at_k(query_result, query_landmark, k):
    if k <= 0:
        return 0.0
    
    top_k_results = query_result[:k]
    correct_count = 0
    
    for result in top_k_results:
        basename = os.path.splitext(os.path.basename(result['path']))[0]
        if basename.startswith(query_landmark + '-'):
            correct_count += 1
    
    return correct_count / k

def evaluate_retrieval(results, query_landmark_groups, top_k_values=[20, 40, 60]):
    precision_results = {landmark: {k: [] for k in top_k_values} for landmark in LANDMARKS}
    
    for landmark, query_paths in query_landmark_groups.items():
        if landmark not in precision_results:
            continue
        
        for query_path in query_paths:
            if query_path not in results:
                continue
            
            query_result = results[query_path]
            
            for k in top_k_values:
                precision = calculate_precision_at_k(query_result, landmark, k)
                precision_results[landmark][k].append(precision)
    
    for landmark in precision_results:
        for k in top_k_values:
            if precision_results[landmark][k]:
                precision_results[landmark][k] = np.mean(precision_results[landmark][k])
            else:
                precision_results[landmark][k] = 0.0
    
    return precision_results

def plot_precision_curve(precision_results, landmark, output_dir=METRICS_OUTPUT_DIR):
    k_values = [20, 40, 60]
    precisions = [precision_results[landmark][k] for k in k_values]
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, precisions, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.title(f'P@K Curve for {landmark}', fontsize=14)
    plt.xlabel('K', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.xticks(k_values)
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    for i, (k, p) in enumerate(zip(k_values, precisions)):
        plt.text(k, p + 0.02, f'{p:.3f}', ha='center', fontsize=10)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{landmark}_precision_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_precision_curves(precision_results):
    for landmark in LANDMARKS:
        plot_precision_curve(precision_results, landmark)
        print(f"Saved precision curve for {landmark}")

def print_metrics_summary(precision_results):
    print("\n=== Retrieval Metrics Summary ===")
    print(f"{'Landmark':<8} {'P@20':<8} {'P@40':<8} {'P@60':<8}")
    print("-" * 40)
    
    for landmark in LANDMARKS:
        p20 = precision_results[landmark][20]
        p40 = precision_results[landmark][40]
        p60 = precision_results[landmark][60]
        print(f"{landmark:<8} {p20:<8.3f} {p40:<8.3f} {p60:<8.3f}")
    
    avg_p20 = np.mean([precision_results[l][20] for l in LANDMARKS])
    avg_p40 = np.mean([precision_results[l][40] for l in LANDMARKS])
    avg_p60 = np.mean([precision_results[l][60] for l in LANDMARKS])
    print("-" * 40)
    print(f"{'Average':<8} {avg_p20:<8.3f} {avg_p40:<8.3f} {avg_p60:<8.3f}")