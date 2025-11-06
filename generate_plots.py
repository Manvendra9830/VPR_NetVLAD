
# generate_plots.py
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_recall_comparison(alexnet_recalls, vgg16_recalls, output_dir='.'):
    """Generates and saves comparison plots for model recalls."""
    
    labels = [f'R@{n}' for n in [1, 5, 10, 20]]
    x = np.arange(len(labels)) # the label locations
    width = 0.35 # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, alexnet_recalls, width, label='AlexNet (Trained)')
    rects2 = ax.bar(x + width/2, vgg16_recalls, width, label='VGG16 (Pre-trained)')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Recall Rate')
    ax.set_title('Model Comparison: Recall @ N')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Attach a text label above each bar in *rects*, displaying its height.
    ax.bar_label(rects1, padding=3, fmt='%.4f')
    ax.bar_label(rects2, padding=3, fmt='%.4f')

    ax.set_ylim(0, 1.1) # Set y-axis limit slightly above 1 for labels

    fig.tight_layout()
    
    output_filename = os.path.join(output_dir, 'recall_comparison.png')
    plt.savefig(output_filename)
    print(f"✅ Comparison plot saved to {output_filename}")

if __name__ == r'__main__':
    parser = argparse.ArgumentParser(description='Plotting Script for Recall Comparison')
    parser.add_argument('--alexnet_recalls', type=float, nargs='+', required=True, 
                        help='Space-separated list of Recall@1, 5, 10, 20 values for AlexNet')
    parser.add_argument('--vgg16_recalls', type=float, nargs='+', required=True, 
                        help='Space-separated list of Recall@1, 5, 10, 20 values for VGG16')
    parser.add_argument('--output_dir', type=str, default='.', 
                        help='Directory to save the comparison plot (default: current directory)')
    
    args = parser.parse_args()
    
    if len(args.alexnet_recalls) != 4 or len(args.vgg16_recalls) != 4:
        raise ValueError("Please provide exactly 4 recall values (R@1, 5, 10, 20) for each model.")
        
    plot_recall_comparison(args.alexnet_recalls, args.vgg16_recalls, args.output_dir)
