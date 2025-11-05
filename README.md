# Visual Place Recognition using NetVLAD and CNN Backbones

This project implements and evaluates the NetVLAD architecture for the task of Visual Place Recognition (VPR). It provides a complete pipeline for training and testing different CNN backbones on the standard Pittsburgh 30k benchmark dataset.

*This work builds upon the PyTorch implementation by Nanne, available [here](https://github.com/Nanne/pytorch-NetVlad).*

## My Contributions and Project Focus

While based on an existing implementation, this project includes several key contributions and modifications to enable modern training and evaluation:

-   **Code Portability:** Patched the source code to remove hardcoded paths and ensure it can run on any machine.
-   **GPU Compatibility:** Corrected device placement issues to ensure the model properly utilizes an available GPU during training.
-   **Reproducible Pipelines:** Developed and documented clear, end-to-end pipelines for training `AlexNet` from scratch and evaluating a pre-trained `VGG16` model.
-   **Result Visualization:** Added a dedicated script (`generate_plots.py`) to compare the performance of different models.
-   **Dependency Management:** Modernized the dependency list and instructions for a smoother setup process.

## About The Project

The goal of Visual Place Recognition is to identify a camera's geographical location by matching its current view against a database of geo-tagged images. This project uses NetVLAD, a powerful aggregation layer, to create a single, robust "global descriptor" vector for each image.

### Dataset: Pittsburgh 30k

This project is configured to use the **Pittsburgh 30k (pitts30k)** dataset.

-   **Training Data:** The model is trained on the `train` split of pitts30k.
-   **Evaluation Data:** Performance is measured on the `val` split, which is geographically separate from the training data to ensure a fair test of the model's generalization ability. The `val` set is composed of a **database** of known places and a set of **query** images to be localized.

---

## Getting Started

Follow these instructions to set up the project and reproduce the results.

### 1. Prerequisites

A Python environment with the following libraries is required. Using a virtual environment is highly recommended.

-   PyTorch
-   Faiss (`faiss-gpu` is recommended)
-   NumPy
-   scikit-learn
-   h5py
-   TensorBoardX
-   Matplotlib
-   gdown (for downloading the pre-trained model)

You can install all dependencies with pip:
```bash
pip install torch torchvision faiss-gpu tensorboardX h5py matplotlib scikit-learn gdown
```

### 2. Data and Pre-trained Model Setup

This project requires the Pittsburgh image dataset, the pitts30k specification files, and the pre-trained VGG16 model checkpoint.

1.  **Download Images:** The Pittsburgh 250k image database can be downloaded from [here](https://data.deepai.org/pittsburgh.zip) (~85 GB).
2.  **Download Specifications:** The `.mat` files for the pitts30k train/val/test splits are available [here](https://www.di.ens.fr/willow/research/netvlad/data/netvlad_v100_datasets.tar.gz).

After downloading and unzipping, place the image folders (`000`-`010`, `queries_real`) and the `datasets` folder in the root of this project directory.

### 3. Code Preparation

Before running, execute the following commands from the project root to apply necessary patches and download the VGG16 checkpoint.

```bash
# Fix the hardcoded data path to use the current directory
sed -i 's|root_dir = .*|root_dir = "./"|' pittsburgh.py

# Ensure the model is correctly moved to the GPU for training/testing
sed -i "s/model = nn.DataParallel(model)/model = nn.DataParallel(model).to(device)/" main.py

# Create the directory for saving model checkpoints
mkdir -p checkpoints

# Create the directory structure for the pre-trained model
mkdir -p pretrained_models/vgg16_netvlad_checkpoint/checkpoints/

# Download and place the pre-trained VGG16 model from Google Drive
# The gdown package is required for this command.
gdown '1qd3eyDE9zGTmV6CEyHUlU2qD8MRw_N5l' -O vgg16_netvlad_checkpoint.zip
unzip vgg16_netvlad_checkpoint.zip
# This will likely extract a file like 'checkpoint.pth.tar'. Move it.
mv checkpoint.pth.tar pretrained_models/vgg16_netvlad_checkpoint/checkpoints/
```

---

## Usage: Running the Pipelines

This repository is configured to train AlexNet and evaluate the pre-trained VGG16.

### AlexNet Pipeline (Train from Scratch)

```bash
# 1. Generate AlexNet Centroids
python main.py --mode=cluster --arch=alexnet --num_clusters=64

# 2. Train AlexNet
python main.py --mode=train --arch=alexnet --num_clusters=64 --nEpochs=5 --batchSize=8

# 3. Test AlexNet
# NOTE: Replace 'path/to/alexnet/run' with the run path from the training output.
python main.py --mode=test --arch=alexnet --split=val --resume path/to/alexnet/run/
```

### VGG16 Pipeline (Evaluate Pre-trained Model)

This pipeline is much faster as it skips training and directly evaluates the high-performance, pre-trained VGG16 model.

```bash
# 1. Test the pre-trained VGG16 model
python main.py --mode=test --arch=vgg16 --split=val --resume pretrained_models/vgg16_netvlad_checkpoint/
```

### Visualizing Results

Use the included `generate_plots.py` script to create a bar chart comparing the performance of the two models. After running the test commands for both pipelines, insert the resulting Recall@N scores into the command below.

```bash
# --- EXAMPLE PLOTTING COMMAND ---
# Replace the recall values with the actual numbers from your test outputs.

python generate_plots.py \
    --alexnet_recalls <R@1> <R@5> <R@10> <R@20> \
    --vgg16_recalls <R@1> <R@5> <R@10> <R@20>
```