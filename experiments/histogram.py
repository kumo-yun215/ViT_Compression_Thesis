import torch
import matplotlib.pyplot as plt
import numpy as np
import timm
import os


def quick_plot_msa_mlp(model_path, model_name='vit_tiny_patch16_224', num_classes=200, x_range=(-1, 1)):
    """Plot MSA and MLP parameter histograms with focused x-axis range."""

    # 1. check model
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return None, None

    print(f"[INFO] Creating model architecture: {model_name}")
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    print(f"[INFO] Loading weights from: {model_path}")
    try:
        state_dict = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None, None

    # deal with different checkpoint formats
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    try:
        model.load_state_dict(state_dict, strict=False)
        print("[INFO] Weights loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load state_dict: {e}")
        return None, None

    # 2. collect parameters
    msa, mlp = [], []
    msa_patterns = ['attn.qkv', 'attn.proj', 'attention.qkv', 'attention.proj']
    mlp_patterns = ['mlp.fc1', 'mlp.fc2', 'fc1', 'fc2']

    for name, param in model.named_parameters():
        vals = param.detach().cpu().numpy().flatten()
        if any(p in name for p in msa_patterns):
            msa.extend(vals)
        elif any(p in name for p in mlp_patterns):
            if 'head' not in name:
                mlp.extend(vals)

    msa = np.array(msa)
    mlp = np.array(mlp)

    print(f"[INFO] Collected {len(msa):,} MSA parameters")
    print(f"[INFO] Collected {len(mlp):,} MLP parameters")

    # print parameters ratio in x_range
    msa_in_range = np.sum((msa >= x_range[0]) & (msa <= x_range[1])) / len(msa) * 100
    mlp_in_range = np.sum((mlp >= x_range[0]) & (mlp <= x_range[1])) / len(mlp) * 100
    print(f"[INFO] {msa_in_range:.1f}% of MSA parameters are in range {x_range}")
    print(f"[INFO] {mlp_in_range:.1f}% of MLP parameters are in range {x_range}")

    # 3. plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # MSA histogram
    n1, bins1, patches1 = ax1.hist(msa, bins=100, color='#1f77b4', alpha=0.8,
                                   edgecolor='black', linewidth=0.5,
                                   range=x_range)
    ax1.set_xlim(x_range)
    ax1.set_title(f'MSA Parameters (n={len(msa):,})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Parameter Value', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Zero')
    ax1.axvline(x=np.mean(msa), color='green', linestyle='--', alpha=0.7, linewidth=2,
                label=f'Mean: {np.mean(msa):.4f}')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # MLP histogram
    n2, bins2, patches2 = ax2.hist(mlp, bins=100, color='#ff7f0e', alpha=0.8,
                                   edgecolor='black', linewidth=0.5,
                                   range=x_range)
    ax2.set_xlim(x_range)
    ax2.set_title(f'MLP Parameters (n={len(mlp):,})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Parameter Value', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Zero')
    ax2.axvline(x=np.mean(mlp), color='green', linestyle='--', alpha=0.7, linewidth=2,
                label=f'Mean: {np.mean(mlp):.4f}')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'ViT Parameter Distribution',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    # save
    save_path = os.path.join(os.path.dirname(model_path) if os.path.dirname(model_path) else '.',
                             f'msa_mlp_histogram_xrange_{x_range[0]}_{x_range[1]}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Plot saved to: {save_path}")

    plt.show()
    return msa, mlp


# ==================== main ====================
if __name__ == '__main__':
    model_path = r'D:\2024-2026KTH\Vit_Compression_Thesis\checkpoints\vit_tiny_patch16_224_tiny_imagenet_best.PTH'

    print(f"Current path: {os.getcwd()}")
    print(f"Target path: {model_path}")
    print(f"File existence: {os.path.exists(model_path)}")

    result = quick_plot_msa_mlp(
        model_path=model_path,
        model_name='vit_tiny_patch16_224',
        num_classes=200,
        x_range=(-0.3, 0.3)
    )

    if result[0] is not None and result[1] is not None:
        msa_params, mlp_params = result

        print("\n" + "=" * 60)
        print("PARAMETER STATISTICS (within -1 to 1 range)")
        print("=" * 60)

        print(f"\nMSA PARAMETERS:")
        print(f"  Count: {len(msa_params):,}")
        print(f"  Mean: {np.mean(msa_params):.6f}")
        print(f"  Std: {np.std(msa_params):.6f}")
        print(f"  Min: {np.min(msa_params):.6f}")
        print(f"  Max: {np.max(msa_params):.6f}")
        print(f"  Zero-centered: {np.sum(msa_params == 0):,} parameters are exactly 0")

        print(f"\nMLP PARAMETERS:")
        print(f"  Count: {len(mlp_params):,}")
        print(f"  Mean: {np.mean(mlp_params):.6f}")
        print(f"  Std: {np.std(mlp_params):.6f}")
        print(f"  Min: {np.min(mlp_params):.6f}")
        print(f"  Max: {np.max(mlp_params):.6f}")
        print(f"  Zero-centered: {np.sum(mlp_params == 0):,} parameters are exactly 0")
        print("=" * 60)
    else:
        print("[ERROR] Failed to load model or extract parameters")