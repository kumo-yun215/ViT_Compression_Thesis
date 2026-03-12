import os
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt


def plot_svd_and_energy(model_path, model_name='vit_tiny_patch16_224', num_classes=200):
    """
    Perform Singular Value Decomposition (SVD) on selected weight matrices
    to analyze their rank and plot the cumulative energy.
    """
    # 1. check and load
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return

    print(f"[INFO] Loading model: {model_name}")
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    try:
        state_dict = torch.load(model_path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict, strict=False)
        print("[INFO] Weights loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
        return

    # 2. choose target layer
    # For vit_tiny，there are 12 blocks (from 0 to 11)
    # pick Block 0 and Block 11
    target_layers = {
        'Block 0 - MSA (qkv)': 'blocks.0.attn.qkv.weight',
        'Block 0 - MLP (fc1)': 'blocks.0.mlp.fc1.weight',
        'Block 11 - MSA (qkv)': 'blocks.11.attn.qkv.weight',
        'Block 11 - MLP (fc1)': 'blocks.11.mlp.fc1.weight'
    }

    results = {}

    # 3. SVD
    for label, layer_name in target_layers.items():
        weight = model.state_dict()[layer_name].float()
        if weight.dim() != 2:
            weight = weight.view(weight.size(0), -1)

        print(f"[INFO] Performing SVD on {label} (Shape: {weight.shape})...")

        _, S, _ = torch.linalg.svd(weight, full_matrices=False)
        S = S.numpy()

        # Cumulative Energy
        energy = S ** 2
        total_energy = np.sum(energy)
        cumulative_energy = np.cumsum(energy) / total_energy

        results[label] = {
            'singular_values': S,
            'cumulative_energy': cumulative_energy
        }

    # 4. plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    linestyles = ['-', '-', '--', '--']

    for i, (label, data) in enumerate(results.items()):
        S = data['singular_values']
        cum_energy = data['cumulative_energy']
        x_axis = np.arange(1, len(S) + 1)

        # Figure 1: Normalized Singular Values
        ax1.plot(x_axis, S / S[0], label=label, color=colors[i], linestyle=linestyles[i], linewidth=2)

        # Figure 2: Cumulative Energy
        ax2.plot(x_axis, cum_energy * 100, label=label, color=colors[i], linestyle=linestyles[i], linewidth=2)

    # Figure 1
    ax1.set_title('Singular Value Decay', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Singular Value Index (Rank)', fontsize=12)
    ax1.set_ylabel('Normalized Magnitude ($S_i / S_1$)', fontsize=12)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=10)

    # Figure 2
    ax2.set_title('Cumulative Energy by Rank', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Singular Values Retained', fontsize=12)
    ax2.set_ylabel('Cumulative Energy (%)', fontsize=12)
    ax2.axhline(y=90, color='gray', linestyle=':', label='90% Energy')
    ax2.axhline(y=99, color='black', linestyle=':', label='99% Energy')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.suptitle('Rank Analysis of ViT Weight Matrices (SVD)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # save
    save_path = os.path.join(os.path.dirname(model_path) if os.path.dirname(model_path) else '.',
                             'vit_rank_analysis_svd.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Plot saved to: {save_path}")

    plt.show()


if __name__ == '__main__':
    model_path = r'D:\2024-2026KTH\Vit_Compression_Thesis\checkpoints\vit_tiny_patch16_224_tiny_imagenet_best.PTH'

    plot_svd_and_energy(
        model_path=model_path,
        model_name='vit_tiny_patch16_224',
        num_classes=200
    )