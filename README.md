# NetVLAD for Visual Place Recognition

An implementation of NetVLAD in PyTorch, with code for training and evaluating models on the Pittsburgh dataset. This repository documents the reproduction of baseline results and explores model optimization techniques like pruning and quantization.

*Original implementation by Nanne, available [here](https://github.com/Nanne/pytorch-NetVlad).*

## Project Results

This table summarizes the performance of different models and optimization techniques.

| Model Architecture | R@1 | R@5 | R@10 | R@20 |
| :--- | :--- | :--- | :--- | :--- |
| **VGG16 (Pre-trained)** | **0.6978** | **0.8509** | **0.8939** | **0.9280** |
| AlexNet (Trained) | (TBD) | (TBD) | (TBD) | (TBD) |
| Pruned Model | (TBD) | (TBD) | (TBD) | (TBD) |
| Quantized Model | (TBD) | (TBD) | (TBD) | (TBD) |

*TBD: To Be Determined in future experiments.*

---

## 1. Setup

This project requires Python and several packages. First, ensure you have all the dependencies by creating a `requirements.txt` file and running the following command:

```bash
pip install -r requirements.txt
```

---

## 2. Required Data & Pretrained Models

To run this project, you need the dataset files and, for some workflows, a pretrained model. Think of it like a cookbook: you need the "recipes" (`.mat` files), the "ingredients" (image files), and sometimes a "pre-made dish" (pretrained model weights).

### Step-by-Step Download Guide

**A. Download the "Recipes" (`.mat` files)**
These small files define the dataset structure (which images are for training/testing) and the ground truth "answer key".

*   **Link:** [http://www.di.ens.fr/willow/research/netvlad/data/netvlad_v100_datasets.tar.gz](http://www.di.ens.fr/willow/research/netvlad/data/netvlad_v100_datasets.tar.gz)
*   **Action:** Download and extract this file. It will create `datasets` and `groundtruth` folders. Place both of these folders in your main project directory.

**B. Download the "Ingredients" (Image files)**
This is a very large download (~18 GB) containing all the actual `.jpg` images.

*   **Link:** [https://www.kaggle.com/datasets/duongoku/pittsburgh250k](https://www.kaggle.com/datasets/duongoku/pittsburgh250k)
*   **Action:** Download and extract this file. It will create several image folders (e.g., `queries_real`, `000`, `001`...). Place all of these folders in your main project directory.

**C. Download the Pretrained VGG16 Model (Optional, but Recommended)**
This allows you to test the VGG16 model without training it from scratch.

*   **Link:** [https://drive.google.com/file/d/17lKHxuxJxtODONW8aciNNT4r__u9UBRJ/view?usp=sharing](https://drive.google.com/file/d/17lKHxuxJxtODONW8aciNNT4r__u9UBRJ/view?usp=sharing)
*   **Action:** Download the `vgg16_netvlad_checkpoint.zip` file. Extract it and place the resulting `vgg16_netvlad_checkpoint` folder in your main project directory.

After these steps, your project directory should look like this:
```
pytorch-NetVlad/
├── main.py
├── requirements.txt
├── datasets/         <-- From Download A
├── groundtruth/      <-- From Download A
├── queries_real/     <-- From Download B
├── 000/              <-- From Download B
├── ...               <-- etc.
└── vgg16_netvlad_checkpoint/  <-- From Download C
```

---

## 3. Workflow and Usage

The general workflow is: **Cluster -> Train -> Test**. You can then optionally **Prune** or **Quantize** a trained model.

### AlexNet Workflow

**Step 1: Generate Clusters**
This is a mandatory first step to initialize the NetVLAD layer.

```bash
python main.py --mode=cluster --arch=alexnet --dataPath=. --cachePath=./cache
```

**Step 2: Train the Model**
Train the AlexNet model from scratch.

```bash
python main.py --mode=train --arch=alexnet --dataPath=. --runsPath=./runs --cachePath=./cache
```

**Step 3: Test the Trained Model**
Evaluate the model you just trained. Replace `<path_to_your_run>` with the timestamped folder created in the `runs/` directory during training.

```bash
python main.py --mode=test --arch=alexnet --split=val --resume=<path_to_your_run>/
```

### VGG16 Workflow

**Step 1: Generate Clusters**
Even for the pretrained model, you must generate clusters first.

```bash
python main.py --mode=cluster --arch=vgg16 --dataPath=. --cachePath=./cache
```

**Step 2: Test the Pretrained Model**
Evaluate the performance of the pretrained VGG16 model you downloaded.

```bash
python main.py --mode=test --arch=vgg16 --split=val --resume=vgg16_netvlad_checkpoint/vgg16_netvlad_checkpoint/
```

---

### Optimization Workflows

After you have a trained model (like the pretrained VGG16 or an AlexNet you trained), you can apply these optimizations.

**Pruning**
Make the model smaller by removing weights.

```bash
# Example: Apply 10% unstructured pruning to the pretrained VGG16 model
python main.py --mode=prune --arch=vgg16 --pruning_type=unstructured --pruning_amount=0.1 --resume=vgg16_netvlad_checkpoint/vgg16_netvlad_checkpoint/
```

**Quantization**
Make the model more efficient by reducing the precision of its weights.

```bash
# Example: Apply static quantization to the pretrained VGG16 model
python main.py --mode=quantize --arch=vgg16 --quantization_type=static --dataPath=. --resume=vgg16_netvlad_checkpoint/vgg16_netvlad_checkpoint/
```