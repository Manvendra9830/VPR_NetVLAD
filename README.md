
# NetVLAD for Visual Place Recognition

An implementation of NetVlad in PyTorch, with code for training and evaluating models on the Pittsburgh dataset. This repository documents the reproduction of baseline results and serves as a foundation for future work in model optimization and compression.

*Original implementation by Nanne, available [here](https://github.com/Nanne/pytorch-NetVlad).*

## Project Roadmap & Results

This project follows a multi-stage plan, from establishing baseline performance to exploring model optimization techniques.

### Stage 1: Baseline Performance Evaluation

The first step is to evaluate the performance of standard, pre-trained models to establish a baseline.

| Model Architecture | R@1 | R@5 | R@10 | R@20 |
| :--- | :--- | :--- | :--- | :--- |
| **VGG16 (Pre-trained)** | **0.6978** | **0.8509** | **0.8939** | **0.9280** |
| AlexNet (Trained) | (TBD) | (TBD) | (TBD) | (TBD) |

*TBD: To Be Determined in future experiments.*

### Stage 2: Model Optimization (Future Work)

This stage will involve applying compression techniques to the trained models and evaluating the trade-off between performance and model size.

| Model & Technique | R@1 | R@5 | R@10 | R@20 | Model Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Pruned Model | (TBD) | (TBD) | (TBD) | (TBD) | (TBD) |
| Quantized Model | (TBD) | (TBD) | (TBD) | (TBD) | (TBD) |

---

## Setup

### Dependencies
- PyTorch
- Faiss (`faiss-cpu` is recommended for compatibility, `faiss-gpu` for performance)
- NumPy, SciPy, scikit-learn
- h5py, TensorBoardX, Matplotlib

Install all dependencies using pip:
```bash
pip install torch torchvision faiss-cpu scipy numpy sklearn h5py tensorboardX matplotlib
```

### Data

This project requires the **Pittsburgh 250k** image database and the **pitts30k** dataset specifications.

1.  **Download Images:** The Pittsburgh 250k image database can be downloaded from a source like [Kaggle](https://www.kaggle.com/datasets/pittsburgh/pittsburgh-vpr-dataset)(~85 GB).
2.  **Download Specifications:** The `.mat` files for the pitts30k train/val/test splits are available [here](https://www.di.ens.fr/willow/research/netvlad/data/netvlad_v100_datasets.tar.gz).

After downloading and unzipping, place the image folders (`000`-`010`, `queries_real`) and the `datasets` folder in the root of this project directory.

---

## General Workflow

The project has three main modes of operation, which should generally be run in this order:

### 1. `cluster` Mode

This step is **required** before training or testing any model. It builds a "visual vocabulary" (centroids) from the training data. This must be run once for each CNN architecture.

```bash
# For VGG16
python main.py --mode=cluster --arch=vgg16 --num_clusters=64

# For AlexNet (future work)
python main.py --mode=cluster --arch=alexnet --num_clusters=64
```

### 2. `train` Mode

This step trains a model from scratch. It is time-consuming and requires a powerful GPU. In this project, this will be used for the AlexNet model.

```bash
# Example for training AlexNet
python main.py --mode=train --arch=alexnet --num_clusters=64 --nEpochs=5
```

### 3. `test` Mode

This step evaluates a trained model. It can be used to test a model you trained yourself (using `--resume`) or to evaluate the "off-the-shelf" performance of a pre-trained model (by omitting `--resume`).

```bash
# To test a model you trained (e.g., AlexNet)
# Replace <path_to_your_run> with the directory saved during training
python main.py --mode=test --split=val --resume=<path_to_your_run>/

# To test the off-the-shelf pre-trained VGG16 model (after clustering)
python main.py --mode=test --arch=vgg16 --split=val
```

**Note on Reproducing VGG16 Results on Google Colab:** For a complete, step-by-step guide to reproduce the VGG16 results, please use the notebook file `vgg16_pretrained_Netvlad_implemenation.ipynb`. This notebook is designed for the Colab environment and requires the pre-trained weights, which can be downloaded from the link provided [here](https://drive.google.com/file/d/17lKHxuxJxtODONW8aciNNT4r__u9UBRJ/view?usp=sharing)
