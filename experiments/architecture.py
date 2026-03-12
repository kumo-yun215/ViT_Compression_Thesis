import torch
import timm
from torchinfo import summary

def show_model_summary(model, input_size=(1, 3, 224, 224)):
    """
    :param model: PyTorch
    :param input_size: input tensor shape (batch_size, channels, height, width)
    """
    stats = summary(model,
                    input_size=input_size,
                    col_names=["input_size", "output_size", "num_params", "mult_adds"],
                    depth=3)
    return stats

# 1. load model from timm
# pretrained=False
model_vit = timm.create_model('vit_tiny_patch16_224', pretrained=False)

# 2. plot architecture
print(show_model_summary(model_vit))