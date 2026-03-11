import os
import requests
import zipfile
from tqdm import tqdm
import io


def download_and_prepare_tiny_imagenet(data_dir='./data'):
    """
    Downloads Tiny-ImageNet-200, extracts it, and reformats the validation set
    to be compatible with torchvision.datasets.ImageFolder.
    """
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    dataset_dir = os.path.join(data_dir, 'tiny-imagenet-200')

    # 1. Check if dataset already exists
    if os.path.exists(dataset_dir):
        print(f"[INFO] Tiny-ImageNet found at {dataset_dir}. Skipping download.")
        return dataset_dir

    print(f"[INFO] Downloading Tiny-ImageNet from {url}...")
    os.makedirs(data_dir, exist_ok=True)

    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    content = io.BytesIO()
    for data in response.iter_content(block_size):
        progress_bar.update(len(data))
        content.write(data)
    progress_bar.close()

    # Unzip
    print("[INFO] Extracting zip file...")
    with zipfile.ZipFile(content) as z:
        z.extractall(data_dir)

    # 2. Reformat Validation Set
    # The original structure: val/images/img.jpg (flat), val/val_annotations.txt
    # Required structure: val/class_id/img.jpg
    print("[INFO] Reformatting validation set structure...")
    val_dir = os.path.join(dataset_dir, 'val')
    img_dir = os.path.join(val_dir, 'images')
    annot_file = os.path.join(val_dir, 'val_annotations.txt')

    # Read annotations to map image -> class
    with open(annot_file, 'r') as f:
        lines = f.readlines()

    val_img_to_class = {}
    for line in lines:
        parts = line.strip().split('\t')
        val_img_to_class[parts[0]] = parts[1]

    # Move images into class subfolders
    for img_file, class_id in val_img_to_class.items():
        src_path = os.path.join(img_dir, img_file)
        dst_folder = os.path.join(val_dir, class_id)

        # Create class folder if not exists
        os.makedirs(dst_folder, exist_ok=True)

        # Move file
        dst_path = os.path.join(dst_folder, img_file)
        if os.path.exists(src_path):
            os.rename(src_path, dst_path)

    # Clean up empty images folder
    if os.path.exists(img_dir) and not os.listdir(img_dir):
        os.rmdir(img_dir)

    print("[INFO] Tiny-ImageNet preparation complete.")
    return dataset_dir