# NetVLAD for Visual Place Recognition

An implementation of NetVLAD in PyTorch, with code for training and evaluating models on the Pittsburgh dataset. This repository documents the reproduction of baseline results and explores model optimization techniques like pruning and quantization.

*Original implementation by Nanne, available [here](https://github.com/Nanne/pytorch-NetVlad).*

## 📊 Updated Results

### VGG16 Model Results

| Model Type | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
|---|---|---|---|---|
| Pre-trained | 0.8537 | 0.9473 | 0.9696 | 0.9842 |
| Unstructured Pruned (10%) | 0.8482 | 0.9470 | 0.9694 | 0.9842 |
| Unstructured Pruned (20%) | 0.8513 | 0.9489 | 0.9711 | 0.9836 |
| Unstructured Pruned (30%) | 0.8265 | 0.9387 | 0.9658 | 0.9791 |
| Structured Pruned | 0.3916 | 0.6066 | 0.6994 | 0.7844 |
| Quantization (Static) | 0.8484 | 0.9447 | 0.9675 | 0.9829 |
| Quantization (QAT) | 0.8395 | 0.9389 | 0.9631 | 0.9795 |

### AlexNet Model Results

| Model Type | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
|---|---|---|---|---|
| Pre-trained | 0.6711 | 0.8479 | 0.8972 | 0.9351 |
| Unstructured Pruned (10%) | 0.6818 | 0.8545 | 0.9044 | 0.9407 |
| Unstructured Pruned (20%) | 0.6853 | 0.8569 | 0.9033 | 0.9368 |
| Unstructured Pruned (30%) | 0.6450 | 0.8270 | 0.8841 | 0.9226 |
| Structured Pruned | 0.1435 | 0.2939 | 0.3987 | 0.5230 |
| Quantization (Static) | 0.6711 | 0.8492 | 0.8927 | 0.9338 |
| Quantization (QAT) | 0.6702 | 0.8469 | 0.8947 | 0.9339 |

---

## 1. Setup and Directory Structure

### A. Install Dependencies
This project requires Python and several packages. First, clone the repository, then install the dependencies:
```bash
git clone https://github.com/Manvendra9830/VPR_NetVLAD.git
cd VPR_NetVLAD
pip install -r requirements.txt
```

### B. Download Data and Pretrained Models
To run this project, you need the dataset files and pretrained models. After downloading and extracting everything, your project folder must have the following structure:

```
VPR_NetVLAD/
│
├── main.py
├── requirements.txt
├── ... (other .py files)
│
├── data/
│   └── centroids/
│       └── alexnet_pittsburgh_64_desc_cen.hdf5   <-- Created automatically
│
├── datasets/
│   ├── pitts30k_train.mat                      <-- From Download #1
│   └── ... (other .mat files)
│
├── 000/                                        <-- From Download #2
│   └── ... (image files)
├── ... (up to 010/)
│
├── queries_real/                               <-- From Download #2
│   └── ... (image files)
│
└── vgg16_netvlad_checkpoint/                     <-- From Download #3
    └── vgg16_netvlad_checkpoint/
        └── checkpoints/
            └── checkpoint.pth.tar
```

**Download Links:**
1.  **Dataset Definitions (`.mat` files):** [Download Link](http://www.di.ens.fr/willow/research/netvlad/data/netvlad_v100_datasets.tar.gz)
    *   **Action:** Extract and place the `datasets` and `groundtruth` folders in the project root.
2.  **Image Files (`.jpg` files):** [Download Link](https://www.kaggle.com/datasets/duongoku/pittsburgh250k)
    *   **Action:** Extract and place all image folders (`000`-`010`, `queries_real`) in the project root.
3.  **Pretrained VGG16 Model:** [Download Link](https://drive.google.com/file/d/17lKHxuxJxtODONW8aciNNT4r__u9UBRJ/view?usp=sharing)
    *   **Action:** Extract and place the `vgg16_netvlad_checkpoint` folder in the project root.

---

## 2. Workflow and Commands

The general workflow is: **Cluster -> Train -> Test**. Optimizations are then applied to the trained models. All commands should be run from the project root.

### AlexNet Workflow

**Phase 1: Train and Test Baseline AlexNet**
```bash
# Step 1: Generate Clusters (Run Once)
python main.py --mode=cluster --arch=alexnet --nGPU 1

# Step 2: Train the AlexNet Model (Run Once)
# This saves the best model to alexnet_netvlad_checkpoint/alexnet_netvlad_checkpoint/checkpoints/
python main.py --mode=train --arch=alexnet --nGPU 1 --optim ADAM --lr 0.0001

# Step 3: Test the Baseline Model
python main.py --mode=test --arch=alexnet --split=test --nGPU 1 --resume alexnet_netvlad_checkpoint/alexnet_netvlad_checkpoint/ --ckpt best
```

**Phase 2: Pruning & Quantization for AlexNet**
(These commands reuse the checkpoint from Step 2)
```bash
# Unstructured Pruning @ 10%
python main.py --mode=prune --arch=alexnet --pruning_type=unstructured --pruning_amount=0.1 --nGPU 1 --resume alexnet_netvlad_checkpoint/alexnet_netvlad_checkpoint/

# Structured Pruning @ 20%
python main.py --mode=prune --arch=alexnet --pruning_type=structured --pruning_amount=0.2 --nGPU 1 --resume alexnet_netvlad_checkpoint/alexnet_netvlad_checkpoint/

# Post-Training Quantization (PTQ) - Runs on CPU
python main.py --mode=quantize --arch=alexnet --quantization_type=static --resume alexnet_netvlad_checkpoint/alexnet_netvlad_checkpoint/

# Quantization-Aware Training (QAT) - Runs on CPU
python main.py --mode=quantize --arch=alexnet --quantization_type=qat --resume alexnet_netvlad_checkpoint/alexnet_netvlad_checkpoint/
```

### VGG16 Workflow (Using Pretrained Model)

**Phase 1: Test Baseline VGG16**
```bash
# Step 1: Generate Clusters (Run Once)
python main.py --mode=cluster --arch=vgg16 --nGPU 1 --cacheBatchSize=4

# Step 2: Test the Pretrained Model (Baseline)
python main.py --mode=test --arch=vgg16 --split=test --nGPU 1 --resume vgg16_netvlad_checkpoint/vgg16_netvlad_checkpoint/
```

**Phase 2: Pruning & Quantization for VGG16**
(These commands reuse the pretrained checkpoint)
```bash
# Unstructured Pruning @ 10%
python main.py --mode=prune --arch=vgg16 --pruning_type=unstructured --pruning_amount=0.1 --nGPU 1 --resume vgg16_netvlad_checkpoint/vgg16_netvlad_checkpoint/

# Structured Pruning @ 20%
python main.py --mode=prune --arch=vgg16 --pruning_type=structured --pruning_amount=0.2 --nGPU 1 --resume vgg16_netvlad_checkpoint/vgg16_netvlad_checkpoint/

# Post-Training Quantization (PTQ) - Runs on CPU
python main.py --mode=quantize --arch=vgg16 --quantization_type=static --resume vgg16_netvlad_checkpoint/vgg16_netvlad_checkpoint/

# Quantization-Aware Training (QAT) - Runs on CPU
python main.py --mode=quantize --arch=vgg16 --quantization_type=qat --resume vgg16_netvlad_checkpoint/vgg16_netvlad_checkpoint/
```