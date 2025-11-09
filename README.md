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

This project requires Python and several packages. First, ensure you have all the dependencies by running the following command from your project root:

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

*   **Link:** [https://data.deepai.org/pittsburgh.zip](https://data.deepai.org/pittsburgh.zip)
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

#### Phase 1: Get the Baseline AlexNet Result

**Step 1: Generate Clusters (Run Once)**
This is the only time you need to run this. It creates the `alexnet_pittsburgh_64_desc_cen.hdf5` file in `data/centroids/`.

```bash
python main.py --mode=cluster --arch=alexnet --nGPU 1
```

**Step 2: Train the AlexNet Model (Run Once)**
This uses the centroids from Step 1 and saves your trained model to `alexnet_netvlad_checkpoint/alexnet_netvlad_checkpoint/checkpoints/`. This is the most time-consuming step.

```bash
python main.py --mode=train --arch=alexnet --nGPU 1 --optim ADAM --lr 0.0001
```

**Step 3: Test the Baseline Model**
This uses the checkpoint from Step 2 to get your baseline R@N scores.

```bash
python main.py --mode=test --arch=alexnet --split=test --nGPU 1 --resume=alexnet_netvlad_checkpoint/alexnet_netvlad_checkpoint/
```

#### Phase 2: Apply Optimizations to the Trained AlexNet Model

For all the following steps, you will **reuse the trained checkpoint** from Phase 1. The `--resume` path will always be the same.

**Step 4: Get Structured Pruning Result**
This applies 20% structured pruning to the trained model and evaluates it.

```bash
python main.py --mode=prune --arch=alexnet --pruning_type=structured --pruning_amount=0.2 --nGPU 1 --resume=alexnet_netvlad_checkpoint/alexnet_netvlad_checkpoint/
```

**Step 5: Get Unstructured Pruning Result**
This applies 20% unstructured pruning to the trained model and evaluates it.

```bash
python main.py --mode=prune --arch=alexnet --pruning_type=unstructured --pruning_amount=0.2 --nGPU 1 --resume=alexnet_netvlad_checkpoint/alexnet_netvlad_checkpoint/
```

**Step 6: Get Static Quantization Result**
This applies post-training static quantization to the trained model and evaluates it.

```bash
python main.py --mode=quantize --arch=alexnet --quantization_type=static --nGPU 1 --resume=alexnet_netvlad_checkpoint/alexnet_netvlad_checkpoint/
```

**Step 7: Get Quantization-Aware Training (QAT) Result**
This applies QAT to the trained model, fine-tunes it for a few epochs, and then evaluates it.

```bash
python main.py --mode=quantize --arch=alexnet --quantization_type=qat --nGPU 1 --resume=alexnet_netvlad_checkpoint/alexnet_netvlad_checkpoint/
```

### VGG16 Workflow

#### Workflow A: Using the Pretrained VGG16 Model (Recommended for Baseline)

This workflow uses the `vgg16_netvlad_checkpoint` you already have.

**Step 1: Generate Clusters (Run Once)**
This is a one-time setup. It creates the `vgg16_pittsburgh_64_desc_cen.hdf5` file in `data/centroids/`.

```bash
python main.py --mode=cluster --arch=vgg16 --nGPU 1 --cacheBatchSize=4
```
*(Note: Using `--cacheBatchSize=4` to avoid GPU memory issues.)*

**Step 2: Test the Pretrained VGG16 Model (Baseline Result)**
This uses the provided pretrained checkpoint to get your baseline R@N scores.

```bash
python main.py --mode=test --arch=vgg16 --split=test --nGPU 1 --resume=vgg16_netvlad_checkpoint/vgg16_netvlad_checkpoint/
```

**Step 3: Prune the Pretrained VGG16 Model (Example: Structured Pruning)**
This applies 20% structured pruning to the pretrained model and evaluates it.

```bash
python main.py --mode=prune --arch=vgg16 --pruning_type=structured --pruning_amount=0.2 --nGPU 1 --resume=vgg16_netvlad_checkpoint/vgg16_netvlad_checkpoint/
```

**Step 4: Quantize the Pretrained VGG16 Model (Example: Static Quantization)**
This applies post-training static quantization to the pretrained model and evaluates it.

```bash
python main.py --mode=quantize --arch=vgg16 --quantization_type=static --nGPU 1 --resume=vgg16_netvlad_checkpoint/vgg16_netvlad_checkpoint/
```

**Step 5: Quantize the Pretrained VGG16 Model (Example: Quantization-Aware Training)**
This applies QAT to the pretrained model, fine-tunes it for a few epochs, and then evaluates it.

```bash
python main.py --mode=quantize --arch=vgg16 --quantization_type=qat --nGPU 1 --resume=vgg16_netvlad_checkpoint/vgg16_netvlad_checkpoint/
```

#### Workflow B: Training VGG16 from Scratch

This workflow trains a VGG16 model without using any pre-existing weights (other than the ImageNet pretraining for the encoder, which is standard).

**Step 1: Generate Clusters (Run Once)**
This is the same as above. It creates the `vgg16_pittsburgh_64_desc_cen.hdf5` file in `data/centroids/`.

```bash
python main.py --mode=cluster --arch=vgg16 --nGPU 1 --cacheBatchSize=4
```

**Step 2: Train the VGG16 Model from Scratch**
This will train the VGG16 model and save its checkpoints to `vgg16_netvlad_checkpoint/vgg16_netvlad_checkpoint/checkpoints/`.

```bash
python main.py --mode=train --arch=vgg16 --nGPU 1 --optim ADAM --lr 0.0001 --cacheBatchSize=4
```
*(Note: Using `--cacheBatchSize=4` to avoid GPU memory issues during training.)*

**Step 3: Test Your Newly Trained VGG16 Model**
This evaluates the model you just trained.

```bash
python main.py --mode=test --arch=vgg16 --split=test --nGPU 1 --resume=vgg16_netvlad_checkpoint/vgg16_netvlad_checkpoint/
```

**Step 4: Prune Your Trained VGG16 Model (Example: Unstructured Pruning)**
This applies 10% unstructured pruning to your newly trained model.

```bash
python main.py --mode=prune --arch=vgg16 --pruning_type=unstructured --pruning_amount=0.1 --nGPU 1 --resume=vgg16_netvlad_checkpoint/vgg16_netvlad_checkpoint/
```

**Step 5: Quantize Your Trained VGG16 Model (Example: Static Quantization)**
This applies static quantization to your newly trained model.

```bash
python main.py --mode=quantize --arch=vgg16 --quantization_type=static --nGPU 1 --resume=vgg16_netvlad_checkpoint/vgg16_netvlad_checkpoint/
```
