import torch
import matplotlib.pyplot as plt
import numpy as np
import timm
import os


def plot_blockwise_distributions(model_path, model_name='vit_tiny_patch16_224', num_classes=200, x_range=(-0.3, 0.3)):
    # 1. Load model
    if not os.path.exists(model_path):
        print(f"[ERROR] Path not found: {model_path}")
        return

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state_dict = torch.load(model_path, map_location='cpu')
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict, strict=False)

    # 2. Get parameters from blocks
    num_blocks = len(model.blocks)
    msa_patterns = ['attn.qkv', 'attn.proj']
    mlp_patterns = ['mlp.fc1', 'mlp.fc2']

    fig, axes = plt.subplots(num_blocks, 2, figsize=(12, 3 * num_blocks), squeeze=False)

    for i in range(num_blocks):
        block = model.blocks[i]
        block_msa = []
        block_mlp = []

        # Go through all parameters
        for name, param in block.named_parameters():
            vals = param.detach().cpu().numpy().flatten()
            if any(p in name for p in msa_patterns):
                block_msa.extend(vals)
            elif any(p in name for p in mlp_patterns):
                block_mlp.extend(vals)

        block_msa = np.array(block_msa)
        block_mlp = np.array(block_mlp)

        # --- Plot: MSA (left column) ---
        ax_msa = axes[i, 0]
        ax_msa.hist(block_msa, bins=80, range=x_range, color='#1f77b4', alpha=0.7, edgecolor='black', linewidth=0.3)
        ax_msa.set_title(f'Block {i} - MSA', fontsize=10, fontweight='bold')
        ax_msa.set_xlim(x_range)
        ax_msa.grid(True, alpha=0.2)
        ax_msa.annotate(f"Mean: {np.mean(block_msa):.4f}\nStd: {np.std(block_msa):.4f}",
                        xy=(0.05, 0.7), xycoords='axes fraction', fontsize=8,
                        bbox=dict(boxstyle="round", fc="w", alpha=0.5))

        # --- Plot: MLP (right column) ---
        ax_mlp = axes[i, 1]
        ax_mlp.hist(block_mlp, bins=80, range=x_range, color='#ff7f0e', alpha=0.7, edgecolor='black', linewidth=0.3)
        ax_mlp.set_title(f'Block {i} - MLP', fontsize=10, fontweight='bold')
        ax_mlp.set_xlim(x_range)
        ax_mlp.grid(True, alpha=0.2)
        ax_mlp.annotate(f"Mean: {np.mean(block_mlp):.4f}\nStd: {np.std(block_mlp):.4f}",
                        xy=(0.05, 0.7), xycoords='axes fraction', fontsize=8,
                        bbox=dict(boxstyle="round", fc="w", alpha=0.5))

    plt.suptitle(f'Parameter Distribution per Block: {model_name}', fontsize=16, y=1.02, fontweight='bold')
    plt.tight_layout()

    # Save
    save_name = f'blockwise_dist_{x_range[0]}_{x_range[1]}.png'
    save_path = os.path.join(os.path.dirname(model_path), save_name)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"[INFO] Block-wise plot saved to: {save_path}")
    plt.show()


if __name__ == '__main__':
    PATH = r'D:\2024-2026KTH\Vit_Compression_Thesis\checkpoints\vit_tiny_patch16_224_tiny_imagenet_best.PTH'

    plot_blockwise_distributions(
        model_path=PATH,
        model_name='vit_tiny_patch16_224',
        x_range=(-0.2, 0.2)
    )