# Understanding and Using pytorch-NetVlad

This guide provides a beginner-friendly walkthrough of the NetVLAD project for visual place recognition. It explains the core concepts, the purpose of the files, and the step-by-step workflow.

## 1. The Goal: What Are We Doing?

The main goal of this project is **Visual Place Recognition**. Imagine you have a massive photo album of a city (the "database" images). If you take a new photo (the "query" image), can a computer automatically find the exact same place in your album?

This project trains a model that can look at a query image and find the best match from a database of thousands of images.

## 2. The Core Concepts: How It Works

To achieve this, the model needs to learn how to convert an image into a special, compact description called a **global descriptor**. The core idea is that if two images are of the same place, their global descriptors should be very similar.

The process happens in three main stages:

### Stage A: Building a "Visual Vocabulary" (Clustering)

A computer can't understand pixels directly. First, we need to break down the images into "visual words."

1.  **Local Features**: For every image in our training dataset, we use a standard pre-trained neural network (like VGG16) to extract hundreds of **local feature vectors**. Each vector is a list of numbers describing a small patch of the image (e.g., a corner, a texture, a color pattern).
2.  **Creating Centroids**: We then take a huge sample of these local features from all the training images (millions of them!) and use a clustering algorithm (k-means) to group them into a small, fixed number of clusters (e.g., 64 clusters).
3.  **The "Visual Vocabulary"**: The center of each cluster is called a **centroid**. This set of 64 centroids acts as our "visual vocabulary." Each centroid represents a common visual element found in the dataset (like "brick texture," "window frame," "tree foliage," etc.).

**Key Idea**: We don't have one centroid per image. We have one set of shared centroids for the entire dataset. This step is performed by running the code in `cluster` mode.

### Stage B: Describing an Image (The NetVLAD Global Descriptor)

Once we have our vocabulary, we need a way to describe a whole image using it. This is the job of the NetVLAD layer itself.

For any new image, it performs these steps:
1.  It extracts all the local feature vectors from the image.
2.  For each local feature, it finds the closest "visual word" (centroid) from our 64-word vocabulary.
3.  It calculates the difference between the feature and its closest centroid. This is called the "residual" and it captures the unique details of that feature.
4.  Finally, it intelligently aggregates all these residuals into a **single vector**: the **global descriptor**. This single vector is a compact, powerful representation of the entire image.

### Stage C: Learning to See (Training with Triplets)

How does the model learn to create *good* global descriptors? It learns by example, using a method called Triplet Loss.

During training, the model is given three images at a time:
1.  **Query**: The image we want to find a match for.
2.  **Positive**: An image that is a known match for the query (taken from the same place).
3.  **Negative**: An image that is a known non-match (taken from a different place).

The model calculates the global descriptor for all three. The training goal is to update the model's internal weights to:
*   **PULL** the Query and Positive descriptors closer together.
*   **PUSH** the Query and Negative descriptors further apart.

By doing this thousands of times, the model's feature extractor and NetVLAD layer get better and better at producing global descriptors that are highly effective for recognizing places.

## 3. The Step-by-Step Workflow

Here is the practical workflow for running the project:

### Step 1: Create the Centroids (Visual Vocabulary)

This step only needs to be done once for a given dataset and model configuration.

*   **Command**:
    ```bash
    python main.py --mode=cluster --arch=vgg16 --num_clusters=64
    ```
*   **What It Does**:
    *   Loads training images (from folders `000`, `001`, etc.).
    *   Uses the specified architecture (`vgg16`) to extract local features.
    *   Clusters them into 64 groups to find the centroids.
*   **Output**: A `.hdf5` file is created in the `data/centroids/` directory. This file contains your visual vocabulary.

### Step 2: Train the Model

This is where the model learns to recognize places.

*   **Command**:
    ```bash
    python main.py --mode=train --arch=vgg16 --pooling=netvlad
    ```
*   **What It Does**:
    *   Loads the training images and the centroids file from Step 1.
    *   Uses the "triplet loss" method to train the feature extractor and the NetVLAD layer.
    *   This process is iterative and can take a long time.
*   **Output**: A `runs/` directory is created, containing a subfolder for this specific training run. Inside that, in a `checkpoints/` folder, the trained model is saved as a `.pth.tar` file (e.g., `model_best.pth.tar`). This is the "brain" of your model.

### Step 3: Test the Model's Performance

Now, you can see how well your trained model performs.

*   **Command**:
    ```bash
    python main.py --mode=test --split=val --resume=<path_to_your_run_folder>
    ```
    (Replace `<path_to_your_run_folder>` with the actual path from the `runs/` directory).
*   **What It Does**:
    *   Loads the trained "brain" (the checkpoint file) from Step 2.
    *   Calculates the global descriptor for a set of test query images.
    *   Calculates the global descriptors for all the database images to create a searchable catalog.
    *   For each query, it searches the catalog for the most similar descriptors.
*   **Output**: The script prints **Recall@N** scores. A score like `Recall@1: 0.85` means the model found the perfect match as its #1 result for 85% of the queries.

## 4. File & Folder Glossary

*   **`main.py`**: The main script that orchestrates everything. It handles clustering, training, and testing based on the `--mode` you provide.
*   **`pittsburgh.py`**: The data loader. It knows how to read the image files and, crucially, the `.mat` files in the `datasets` folder to get the GPS coordinates for finding positive and negative pairs.
*   **`netvlad.py`**: The implementation of the core NetVLAD layer itself.
*   **`data/centroids/`**: Stores the output of the clustering step—the "visual vocabulary" for the model.
*   **`cache/`**: A temporary folder used during training to store pre-calculated features, which speeds up the process. It's safe to delete its contents.
*   **`runs/`**: The most important output folder. It stores everything related to a training session, including the final trained model checkpoint.
*   **`datasets/`**: Contains the `.mat` files with the ground-truth GPS data for the images.
*   **`__pycache__/`**: A standard Python folder that stores compiled code to make scripts start faster. You can ignore it.

