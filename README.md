
# Visual Place Recognition using NetVLAD

This project implements and evaluates the NetVLAD architecture for Visual Place Recognition (VPR). It provides pipelines for evaluating a pre-trained VGG16 model and training an AlexNet model from scratch on the Pittsburgh 30k benchmark.

*This work builds upon the PyTorch implementation by Nanne, available [here](https://github.com/Nanne/pytorch-NetVlad).*

## VGG16 Pre-Trained Model: Achieved Results

Using the Google Colab pipeline described below, the following "off-the-shelf" performance was achieved for the pre-trained VGG16 model on the `val` split of the pitts30k dataset.

| Metric    | Result |
| :-------- | :----- |
| Recall@1  | 0.6978 |
| Recall@5  | 0.8509 |
| Recall@10 | 0.8939 |
| Recall@20 | 0.9280 |

---

## About The Project

The goal of Visual Place Recognition is to identify a camera's geographical location by matching its current view against a database of geo-tagged images. This project uses NetVLAD, a powerful aggregation layer, to create a single, robust "global descriptor" vector for each image.

### Dataset: Pittsburgh 30k

This project is configured to use the **Pittsburgh 30k (pitts30k)** dataset.

-   **Training Data (`train` split):** Used during the `cluster` mode to generate the visual vocabulary (centroids).
-   **Evaluation Data (`val` split):** Performance is measured on the `val` split, which is geographically separate from the training data to ensure a fair test of the model's generalization ability. The `val` set is composed of a **database** of known places and a set of **query** images to be localized.

---

## VGG16 Pipeline (Recommended for Google Colab)

This pipeline is the fastest way to get a strong baseline result. It skips training and directly evaluates the high-performance, pre-trained VGG16 model.

### 1. Initial Setup on Google Drive

1.  Create a folder in your Google Drive (e.g., `My Drive/WSAI/Internship_project/`).
2.  Upload the project source code as `pytorch-NetVlad.zip` to this folder.
3.  Upload the pre-trained VGG16 model as `vgg16_netvlad_checkpoint.zip` to this folder.

### 2. Colab Execution

Create a new Google Colab notebook, connect to a GPU runtime, and run the following cells in order.

**Cell 1: Mount Drive & Unzip Files**
```python
from google.colab import drive
import os

print("⚙️ Mounting Google Drive...")
drive.mount('/content/drive')

# --- Define Key Paths ---
project_zip_path = "/content/drive/My Drive/WSAI/Internship_project/pytorch-NetVlad.zip"
checkpoint_zip_path = "/content/drive/My Drive/WSAI/Internship_project/vgg16_netvlad_checkpoint.zip"

print("\n⚙️ Unzipping project files...")
!unzip -o "{project_zip_path}" -d /content/
%cd /content/pytorch-NetVlad/

print("\n⚙️ Unzipping VGG16 checkpoint...")
checkpoint_dest = "/content/pytorch-NetVlad/pretrained_models/vgg16_netvlad_checkpoint/checkpoints/"
os.makedirs(checkpoint_dest, exist_ok=True)
!unzip -o "{checkpoint_zip_path}" -d "{checkpoint_dest}"

print("\n✅ Project and checkpoint are ready.")
```

**Cell 2: Install Dependencies & Apply Patches**
```python
import os
print("⚙️ Installing dependencies and patching code...")

# Install libraries
!pip install -q torch torchvision faiss-cpu tensorboardX h5py matplotlib

# Create the 'checkpoints' directory to prevent a known bug
os.makedirs("checkpoints", exist_ok=True)

# Patch pittsburgh.py to fix the dataset path
!sed -i 's|root_dir = .*|root_dir = "/content/pytorch-NetVlad/"|' pittsburgh.py

# Patch main.py to fix the GPU runtime error
!sed -i "s/model = nn.DataParallel(model)/model = nn.DataParallel(model).to(device)/" main.py

print("✅ Environment is fully prepared and patched.")
```

**Cell 3: Generate VGG16 Centroids**
```python
print(f"\n{'='*20} Starting: VGG16 CENTROID GENERATION {'='*20}")
!python main.py --mode=cluster --arch=vgg16 --num_clusters=64
print("✅ VGG16 Centroid Generation Complete.")
```

**Cell 4: Run VGG16 Evaluation**
```python
print(f"\n{'='*20} Starting: VGG16 TESTING {'='*20}")
# The --threads=0 flag is crucial for memory efficiency on Colab
!python main.py --mode=test --arch=vgg16 --split=val --threads=0
print("\n✅ Testing complete.")
```

---

## AlexNet Pipeline (For Future Work)

This pipeline trains the smaller AlexNet model from scratch. It is more time-consuming but useful for understanding the full training process.

```bash
# 1. Generate AlexNet Centroids
python main.py --mode=cluster --arch=alexnet --num_clusters=64

# 2. Train AlexNet
python main.py --mode=train --arch=alexnet --num_clusters=64 --nEpochs=5 --batchSize=8

# 3. Test AlexNet
# NOTE: Replace 'path/to/alexnet/run' with the run path from the training output.
python main.py --mode=test --arch=alexnet --split=val --resume path/to/alexnet/run/
```
