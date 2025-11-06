# Understanding and Using pytorch-NetVlad for Visual Place Recognition

This guide provides a comprehensive walkthrough of the NetVLAD project, explaining the core concepts, the purpose of the files, and the step-by-step workflow for execution.

## 1. The Goal: What Are We Doing?

The main goal of this project is **Visual Place Recognition (VPR)**. Imagine you have a massive photo album of a city (the "database" images). If you take a new photo (the "query" image), the goal is for a computer to automatically find the exact same place in your album.

This project trains a model that can look at a query image and find the best match from a database of thousands of images by converting each image into a special, compact description called a **global descriptor**.

## 2. The Core Concepts: How It Works

The process happens in three main stages:

### Stage A: Building a "Visual Vocabulary" (Clustering)

A computer can't understand pixels directly. First, we need to break down images into "visual words."

1.  **Local Features**: For every image in our training dataset, we use a standard pre-trained neural network (like AlexNet or VGG16) to extract hundreds of **local feature vectors**. Each vector is a list of numbers describing a small patch of the image (e.g., a corner, a texture, a color pattern).
2.  **Creating Centroids**: We then take a huge sample of these local features from all the training images and use a clustering algorithm (k-means) to group them into a small, fixed number of clusters (e.g., 64 clusters).
3.  **The "Visual Vocabulary"**: The center of each cluster is called a **centroid**. This set of 64 centroids acts as our "visual vocabulary." Each centroid represents a common visual element found in the dataset (like "brick texture," "window frame," "tree foliage," etc.).

This step is performed by running the code in `cluster` mode.

### Stage B: Describing an Image (The NetVLAD Global Descriptor)

Once we have our vocabulary, the NetVLAD layer describes a whole image using it.

For any new image, it performs these steps:
1.  It extracts all the local feature vectors from the image.
2.  For each local feature, it finds the closest "visual word" (centroid) from our 64-word vocabulary.
3.  It calculates the difference between the feature and its closest centroid (the "residual").
4.  Finally, it intelligently aggregates all these residuals into a **single vector**: the **global descriptor**. This single vector is a compact, powerful representation of the entire image.

### Stage C: Learning to See (Training with Triplets)

The model learns to create *good* global descriptors using a method called **Triplet Loss**.

During training, the model is given three images at a time:
1.  **Query**: The image we want to find a match for.
2.  **Positive**: An image that is a known match for the query (taken from the same place).
3.  **Negative**: An image that is a known non-match (taken from a different place).

The training goal is to update the model's weights to:
*   **PULL** the Query and Positive descriptors closer together in vector space.
*   **PUSH** the Query and Negative descriptors further apart.

By doing this thousands of times, the model gets better and better at producing descriptors that are highly effective for recognizing places.

---

## 3. Instructions for Setup and Execution

This guide provides all the steps required to set up the environment, download the necessary data, run the AlexNet training and evaluation, and evaluate the VGG16 pre-trained baseline.

**Estimated GPU Time:** Training AlexNet for 5 epochs on a modern GPU (like an NVIDIA T4 or V100) is expected to take approximately **5-7 hours**. VGG16 evaluation is much faster (under 30 minutes).

### 3.1. Initial Environment Setup

These commands will prepare your system by cloning the repository, downloading the large dataset, installing Python libraries, and applying essential code fixes.

```bash
# 1. Clone this GitHub repository
git clone https://github.com/Manvendra9830/VPR_NetVLAD.git

# 2. Navigate into the project directory
cd VPR_NetVLAD

# 3. Download and extract the Pittsburgh 30k dataset (~85 GB download, >150 GB unzipped)
# This is a very large file and will take a significant amount of time.
# Ensure you have sufficient disk space (at least 200 GB free).
wget -c https://data.deepai.org/pittsburgh.zip
unzip pittsburgh.zip

# 4. Install required Python libraries
# It is highly recommended to use a Python virtual environment.
pip install torch torchvision faiss-gpu tensorboardX h5py matplotlib

# 5. Apply necessary code fixes and prepare directories
sed -i 's|root_dir = .*|root_dir = "./"|' pittsburgh.py
sed -i "s/model = nn.DataParallel(model)/model = nn.DataParallel(model).to(device)/" main.py
mkdir -p checkpoints
```

### 3.2. AlexNet Pipeline (Train from Scratch)

This section outlines the steps to run the AlexNet architecture, including generating centroids and training the model from scratch.

```bash
echo "\n======> [AlexNet Pipeline] Starting Process..."

# 1. Generate AlexNet Centroids (Visual Vocabulary)
echo "  Starting: AlexNet Centroid Generation..."
python main.py --mode=cluster --arch=alexnet --num_clusters=64
echo "  ✅ AlexNet Centroid Generation Complete."

# 2. Train AlexNet for 5 Epochs
# Note the path to the saved checkpoint that will be printed in the output.
echo "  Starting: AlexNet Training (5 Epochs)..."
python main.py --mode=train --arch=alexnet --num_clusters=64 --nEpochs=5 --batchSize=8 --threads=2
echo "  ✅ AlexNet Training Complete."

# 3. Test AlexNet
# Replace 'path/to/alexnet/run/' with the actual run path from the training step.
echo "  Starting: AlexNet Testing..."
python main.py --mode=test --arch=alexnet --split=val --resume path/to/alexnet/run/
echo "  ✅ AlexNet Testing Complete."
```

### 3.3. VGG16 Pipeline (Evaluate Pre-trained Baseline)

This section details how to download and evaluate the high-performing VGG16-NetVLAD model.

```bash
echo "\n======> [VGG16 Pipeline] Starting Process..."

# 1. Download the official pre-trained VGG16-NetVLAD model
echo "  Starting: Downloading VGG16 pre-trained model..."
mkdir -p pretrained_models/vgg16_netvlad_checkpoint/checkpoints/
wget https://matlab.p-cheng.org/Downloads/brp/pitts30k_vgg16_netvlad.pth.tar -O pretrained_models/vgg16_netvlad_checkpoint/checkpoints/checkpoint.pth.tar
echo "  ✅ VGG16 Pre-trained Model Download Complete."

# 2. Test the pre-trained VGG16 model
echo "  Starting: VGG16 Testing..."
python main.py --mode=test --arch=vgg16 --split=val --resume pretrained_models/vgg16_netvlad_checkpoint/
echo "  ✅ VGG16 Testing Complete."
```

### 3.4. Visualize Results (Comparison Plots)

After completing the testing for both models, use the `generate_plots.py` script to create a comparison chart. You will need to manually input the Recall@N values from the terminal output.

```bash
# --- EXAMPLE PLOTTING COMMAND ---

# Replace <...> with the actual recall values from the testing steps.
python generate_plots.py \
    --alexnet_recalls <ALEXNET_R@1> <ALEXNET_R@5> <ALEXNET_R@10> <ALEXNET_R@20> \
    --vgg16_recalls <VGG16_R@1> <VGG16_R@5> <VGG16_R@10> <VGG16_R@20>
```

---

## 4. Future Work: Optimization and Compression

After achieving baseline results, the next phase of this project will involve optimizing and compressing the trained models. Techniques like pruning or quantization will be explored. This typically requires a brief period of **fine-tuning** (re-training for a few epochs) to recover any accuracy lost during compression, resulting in faster and more memory-efficient models.
