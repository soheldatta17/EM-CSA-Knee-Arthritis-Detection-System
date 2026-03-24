# EM-CSA Knee Arthritis Detection System
## A Complete Guide from Setup to Advanced Understanding

---

## Table of Contents

1. What This Project Does
2. Prerequisites and Environment
3. Understanding the Dataset
4. Step-by-Step Setup on Google Colab
5. Core Terminology: Basic Level
6. Core Terminology: Intermediate Level
7. Core Terminology: Advanced Level — The EM-CSA Mechanism
8. The Training Pipeline, Step by Step
9. Evaluation and Results
10. Visualization Modules

---

## 1. What This Project Does

This project trains a deep learning model to look at knee X-ray images and classify the severity of arthritis (also called osteoarthritis) into one of five grades. The grading system used is the Kellgren-Lawrence scale, where:

- Grade 0 means no arthritis is present
- Grade 1 means doubtful narrowing of the joint space
- Grade 2 means definite osteophytes (bone spurs) and possible narrowing
- Grade 3 means moderate multiple osteophytes and definite narrowing
- Grade 4 means large osteophytes, marked narrowing, and severe deformity

The model does not just predict a grade. It also shows you, visually, exactly which part of the X-ray it looked at to make the decision. This is done through a custom attention mechanism called EM-CSA (Edge-Enhanced Multi-Scale Coordinate-Spatial Attention), which is the novel contribution of this codebase.

---

## 2. Prerequisites and Environment

### Where to Run This

This code is designed to run on Google Colab. Colab is a free, browser-based environment provided by Google that gives you access to a GPU without any local installation. You open it at colab.research.google.com.

You should not attempt to run this code on your personal computer without significant modifications, because it relies on `google.colab.files` for uploading files, and it expects a GPU for training.

### What You Need Before Starting

- A Google account to access Colab
- Your dataset in a `.zip` file (described in the next section)
- No prior installation of Python or any library is needed. Colab has all required libraries pre-installed.

### Libraries Used

The code imports the following. You do not need to install these in Colab, but you should understand what each one does:

| Library | Purpose |
|---|---|
| `tensorflow` / `keras` | Building, training, and running the neural network |
| `numpy` | Numerical operations on arrays and matrices |
| `matplotlib` | Plotting graphs and displaying images |
| `sklearn` | Generating the confusion matrix and classification report |
| `opencv (cv2)` | Image processing and blending operations |
| `PIL (Pillow)` | Image resizing with high-quality interpolation |
| `os`, `shutil`, `zipfile` | File and folder management |

---

## 3. Understanding the Dataset

### What the Dataset Should Look Like

The dataset is a collection of knee X-ray images organized into folders. Each folder is named after the arthritis grade it represents: `0`, `1`, `2`, `3`, and `4`.

Your zip file should contain either:

- A folder named `Training` that directly contains the five numbered subfolders, or
- The five numbered subfolders (`0`, `1`, `2`, `3`, `4`) at the root of the zip file

If the second structure is found, the code automatically creates the `Training` folder and moves everything into it.

### How the Data Is Split

Once loaded, the dataset is divided into three portions:

- Training set (81%) — used to teach the model
- Validation set (9%) — used to check the model during training without influencing the weights
- Test set (10%) — used only at the very end to report final performance

All images are resized to 256 by 256 pixels and converted to grayscale (one color channel, not three).

---

## 4. Step-by-Step Setup on Google Colab

Follow these steps exactly, in order.

### Step 1: Open Google Colab

Go to colab.research.google.com. Click "New Notebook" or upload the `.py` file by going to File > Upload Notebook.

### Step 2: Change the Runtime to GPU

Click on "Runtime" in the top menu. Then click "Change runtime type". Set "Hardware accelerator" to GPU. Click Save. This step is critical. Without a GPU, training 300 epochs will take many hours instead of minutes.

### Step 3: Upload the Code File

If you have the file `em_csa_knee.py`, go to the Files panel on the left sidebar of Colab, click the upload icon, and upload the file. Then run it cell by cell.

Alternatively, copy and paste the entire code into a single Colab cell or multiple cells separated by the section headers.

### Step 4: Run the Setup Function

The first function to run is `setup_dataset()`. When you run the cell containing this function call, Colab will display a file picker. Select your zip file from your local computer. The code will:

1. Upload the file to Colab's temporary storage
2. Unzip the contents
3. Check if the `Training` folder exists
4. If not, create it and move the class folders into it

### Step 5: Verify the Dataset Structure

After setup, run the following manually in a cell to confirm the structure looks correct:

```python
import os
for root, dirs, files in os.walk("Training"):
    level = root.replace("Training", '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    if level == 1:
        print(f'{indent}  ({len(files)} images)')
```

You should see five subfolders (0, 1, 2, 3, 4), each containing image files.

### Step 6: Run the Main Pipeline

Call `main()`. This single function runs the entire pipeline from data loading through training and evaluation. It takes no arguments.

```python
main()
```

Training will take time. At 300 epochs with a batch size of 16, expect anywhere from 30 minutes to several hours depending on dataset size.

### Step 7: Run the Visualization Dashboard (Optional but Recommended)

After training is complete and `best_model.h5` has been saved, run the second `main()` function defined at the bottom of the file (the one inside the "VISUALIZATION SUITE" section). It will ask you to upload a single X-ray image and will produce a detailed 12-panel visualization showing how the model reasoned about it.

---

## 5. Core Terminology: Basic Level

This section explains terms you will see in the output and code, starting from the most fundamental concepts.

### X-ray Image as a Tensor

An image in this code is not stored as a picture file once loaded. It is stored as a tensor, which is a multi-dimensional array of numbers. A grayscale image of size 256x256 becomes a tensor of shape (256, 256, 1), meaning 256 rows, 256 columns, and 1 color channel. The pixel values range from 0 (black) to 255 (white).

### Batch

Instead of showing the model one image at a time, images are grouped into batches. This code uses a batch size of 16, meaning the model processes 16 images at once before updating its internal parameters. Batching improves training efficiency and stability.

### Epoch

One epoch means the model has seen every image in the training set exactly once. This code trains for 300 epochs, so the model passes through the entire dataset 300 times.

### Label / Class

Each image has a label, which is an integer from 0 to 4 representing the arthritis grade. The model is trained to predict this label from the image. There are 5 classes total.

### Loss

Loss is a single number that measures how wrong the model's predictions are. A lower loss means better predictions. The specific loss function used here is sparse categorical crossentropy, which is the standard choice when you have multiple classes represented as integers (0, 1, 2, 3, 4).

### Accuracy

Accuracy is the percentage of images the model classifies correctly. An accuracy of 0.80 means the model gets 80 out of every 100 images right.

### Optimizer

The optimizer is the algorithm that updates the model's internal weights based on the loss. This code uses RMSProp (Root Mean Square Propagation), which is well-suited for image classification tasks. It adjusts each weight differently based on how that weight has been changing.

---

## 6. Core Terminology: Intermediate Level

### Convolutional Layer (Conv2D)

A convolutional layer is the primary building block of this model. It slides a small filter (also called a kernel) across the image and computes a weighted sum at each position. This operation detects local patterns such as edges, textures, and shapes. The number of filters determines how many different patterns the layer can detect simultaneously.

In this model, convolutional layers are stacked progressively with increasing filter counts: 8, 16, 32, 64, and finally 128. Each layer detects increasingly complex patterns.

### MaxPooling

After a convolutional layer, a MaxPooling layer reduces the spatial dimensions of the feature map by half. It does this by taking the maximum value in each small region. MaxPooling serves two purposes: it reduces computation, and it makes the model less sensitive to the exact position of a feature in the image.

### ReLU Activation

ReLU stands for Rectified Linear Unit. It is a simple mathematical function applied element-wise after convolution: if a value is negative, it becomes zero; otherwise it stays as-is. This introduces non-linearity, which allows the model to learn complex patterns beyond simple linear relationships.

### Sigmoid Activation

Sigmoid squashes any input value into a range between 0 and 1. In this code, sigmoid is used to create attention maps, where values close to 1 mean "pay attention here" and values close to 0 mean "ignore this region."

### Softmax Activation

Softmax is used in the final output layer. It converts the five raw output scores into a probability distribution that sums to 1.0. For example, an output of [0.05, 0.10, 0.60, 0.20, 0.05] means the model is 60% confident the image belongs to class 2.

### Dropout

Dropout is a regularization technique that randomly deactivates a percentage of neurons during each training step. This code uses a dropout rate of 0.2 in the dense layers (meaning 20% of neurons are turned off randomly). This forces the model not to rely too heavily on any single neuron, which reduces overfitting.

### Overfitting

Overfitting occurs when the model performs very well on the training data but poorly on new, unseen images. It means the model has memorized the training examples rather than learning generalizable patterns. Dropout and data augmentation both help prevent this.

### Data Augmentation

Data augmentation artificially increases the diversity of the training data by applying random transformations to images before they are fed to the model. This model applies three augmentations:

- Random horizontal flip: mirrors the image left-to-right
- Random rotation up to 10%: rotates the image slightly
- Random zoom up to 20%: zooms in or out slightly

These transformations make the model more robust to variations in how an X-ray might be positioned or framed.

### Rescaling

Before images enter the network, pixel values are divided by 255 using a Rescaling layer. This converts values from the range [0, 255] to [0, 1]. Neural networks train more stably with small, normalized input values.

### Flatten

After the convolutional and attention layers, the 3D feature map (height x width x channels) is collapsed into a single long 1D vector by the Flatten layer. This vector is then passed to the Dense layers for final classification.

### Dense Layer

A Dense (or fully connected) layer connects every neuron from the previous layer to every neuron in itself. It is the classical building block of neural networks. This model has Dense layers with 1024 and 512 neurons in the classification head.

### Model Checkpoint

During training, the callback `ModelCheckpoint` monitors validation accuracy after each epoch. Whenever a new best validation accuracy is achieved, it saves the model weights to the file `best_model.h5`. This ensures you always retain the best version of the model, even if later epochs cause performance to degrade.

### Validation Accuracy vs. Training Accuracy

- Training accuracy is measured on images the model is actively learning from.
- Validation accuracy is measured on images the model has never trained on.

A large gap between the two, where training accuracy is much higher, is a sign of overfitting.

### Confusion Matrix

A confusion matrix is a table that shows, for each actual class, how often the model predicted each class. The diagonal entries represent correct predictions. Off-diagonal entries represent errors. It reveals specific patterns, such as whether the model frequently confuses grade 2 with grade 3.

### Classification Report

Printed after the confusion matrix, this report gives three metrics per class:

- Precision: of all the images the model labeled as class X, what fraction actually were class X
- Recall: of all the images that truly belong to class X, what fraction did the model correctly identify
- F1-score: the harmonic mean of precision and recall, a single balanced metric

---

## 7. Core Terminology: Advanced Level — The EM-CSA Mechanism

This is the central novel contribution of the codebase. EM-CSA stands for Edge-Enhanced Multi-Scale Coordinate-Spatial Attention. It is a custom attention block inserted into the CNN after the fifth convolutional layer.

Attention mechanisms in neural networks give the model a way to focus on certain regions or features more than others, rather than treating all parts of the feature map equally.

The EM-CSA block proceeds in four sequential phases, described in the code as "Outline, Zoom, Align, Spotlight."

---

### Phase 1: Edge Attention — "Outline"

**Function: `edge_attention_block`**

The first phase explicitly extracts bone boundary information using Sobel filters.

A Sobel filter is a fixed mathematical kernel (a small 3x3 matrix of numbers) that detects intensity gradients in an image. Specifically:

- The horizontal Sobel kernel detects vertical edges (where brightness changes left to right)
- The vertical Sobel kernel detects horizontal edges (where brightness changes top to bottom)

The edge magnitude at each pixel is computed as the square root of the sum of squared responses from both kernels. This produces a map that is bright where edges exist (such as the contour of the femur and tibia bones) and dark where the image is uniform.

This edge map is passed through a sigmoid activation to produce values between 0 and 1, creating an edge attention mask. The mask is then multiplied with the original feature map, and the result is added back to the original. This means edge regions are emphasized while flat regions are preserved but not amplified.

This phase answers the question: "Where are the structural boundaries in this X-ray?"

---

### Phase 2: Multi-Scale Feature Extraction — "Zoom"

**Within `em_csa_block`, after edge attention**

The edge-enhanced features are passed through two parallel convolutional branches simultaneously:

- One branch uses a 3x3 kernel, which detects fine-grained, local features such as small bone irregularities
- The other branch uses a 7x7 kernel, which detects broader, more contextual patterns such as the overall shape of the joint space

These two branches capture different scales of information about the same image. The outputs are kept separate until after the coordinate attention phase, then fused together.

---

### Phase 3: Coordinate Attention — "Align"

**Function: `coordinate_attention_part`**

Standard global average pooling (commonly used in simpler attention mechanisms) compresses the entire spatial feature map into a single vector, losing all position information. Coordinate attention addresses this limitation.

Instead of pooling globally, it pools along each spatial axis independently:

- It pools across the entire width of the feature map (one value per row) to capture vertical position information
- It pools across the entire height of the feature map (one value per column) to capture horizontal position information

These two pools are concatenated along the spatial axis, then compressed using a 1x1 convolution with hard sigmoid activation and batch normalization. The compressed representation is then split back into height and width components. Each component is expanded back to full channel depth using another 1x1 convolution.

The result is two attention maps: one that knows where to focus vertically (useful for locating the joint gap along the image height) and one that knows where to focus horizontally (useful for distinguishing the medial compartment from the lateral compartment).

The final output of this phase is the input feature map multiplied by both attention maps simultaneously. Regions at the intersection of horizontally and vertically important positions are amplified the most.

This phase is applied independently to both the 3x3 and the 7x7 branches from Phase 2.

---

### Phase 3.5: Adaptive Scale Fusion

**Within `em_csa_block`, after coordinate attention**

The two attention-weighted feature maps from the 3x3 and 7x7 branches are fused using learned weights rather than simple averaging.

The two maps are added together and passed through global average pooling to produce a compact descriptor vector. This vector is passed through a small Dense layer (a bottleneck with `filters // ratio` neurons and ReLU activation) and then through two separate Dense layers with softmax activation, one for each scale branch.

The softmax activations produce two sets of weights that together sum to 1.0 across the channel dimension. These weights determine how much each scale branch (fine-grained vs. coarse) should contribute to the final fused representation. The model learns these weights during training, adapting to which scale is more informative for each channel.

This adaptive selection is more powerful than simply adding or averaging the two branches.

---

### Phase 4: Spatial Attention — "Spotlight"

**Function: `spatial_attention_part`**

The final phase focuses on which spatial locations within the feature map are most important, regardless of which channels those locations belong to.

The fused feature map from Phase 3.5 is processed in two ways:

- Channel-wise average pooling: for each pixel position, compute the average value across all channels
- Channel-wise max pooling: for each pixel position, compute the maximum value across all channels

These two single-channel maps are concatenated along the channel dimension, producing a two-channel map. This is passed through a single convolutional layer with a 7x7 kernel (to capture broad spatial context) and a sigmoid activation, producing a final spatial attention map.

Each pixel in the attention map has a value between 0 and 1. The feature map is multiplied by this attention map, amplifying regions the model has learned to associate with arthritis indicators and suppressing irrelevant background regions.

---

### Why the "ratio" Parameter Exists

Both the coordinate attention and the adaptive fusion phases use a parameter called `ratio` (default value 16 in this code). This is a compression ratio that controls the bottleneck size in the intermediate reduction steps. A ratio of 16 means the channel dimension is reduced to `channels / 16` in the bottleneck before being expanded again. This keeps the number of additional parameters introduced by the attention mechanism small relative to the main network.

---

### Grad-CAM: Visualizing the Final Decision

**Functions: `get_gradcam`, `get_smooth_gradcam`**

Gradient-weighted Class Activation Mapping (Grad-CAM) is a post-hoc visualization technique that answers: "For a given prediction, which pixels in the original image were most responsible?"

It works as follows:

1. A sub-model is created that outputs both the last convolutional layer's feature maps and the final prediction simultaneously.
2. The gradient of the predicted class score is computed with respect to the last convolutional layer's output using TensorFlow's `GradientTape`, which records operations to enable automatic differentiation.
3. These gradients are averaged across the spatial dimensions to produce importance weights per channel.
4. The feature maps are weighted by these importances and summed, producing a single heatmap.
5. Only positive values are retained (using `tf.maximum`), since negative values would indicate features that suppress the predicted class.
6. The heatmap is upscaled to the original image size using Lanczos or Bicubic interpolation for smoothness, then overlaid on the original X-ray with a color scale (jet colormap).

Hot colors (red, yellow) indicate regions that most strongly drove the prediction. This allows a clinician to verify whether the model is looking at the correct anatomical region.

---

### Sobel Filter: Mathematical Detail

The Sobel kernels are defined as fixed constants (not learned). The horizontal kernel is:

```
-1  0  1
-2  0  2
-1  0  1
```

The vertical kernel is:

```
-1  -2  -1
 0   0   0
 1   2   1
```

These are applied as depthwise convolutions, meaning each input channel is convolved independently with its own copy of the kernel rather than mixing information across channels. This preserves the per-channel edge information cleanly. The small constant `1e-6` added before taking the square root prevents numerical errors when both gradients are zero (which would produce a zero denominator in derivative operations).

---

## 8. The Training Pipeline, Step by Step

When `main()` is called, it executes the following sequence:

1. `setup_dataset()` — prompts for zip file upload and organizes directory structure
2. `load_dataset()` — loads images from the `Training` directory, shuffles them with a fixed seed for reproducibility, and splits into train, validation, and test sets
3. `plot_class_distribution()` — counts images per class and displays a bar chart; useful for identifying class imbalance
4. `visualize_samples()` — displays 20 sample images in a grid so you can visually verify the data loaded correctly
5. `build_augmented_model()` — constructs the full model graph including the data augmentation pipeline and the EM-CSA block
6. `model.summary()` — prints a table of every layer, its output shape, and its parameter count
7. `plot_model_architecture()` — saves and displays a visual graph of the model (requires Graphviz to be installed; Colab usually has it)
8. `train_model()` — compiles the model with RMSProp and sparse categorical crossentropy, then calls `model.fit()` for 300 epochs
9. `plot_training_history()` — plots accuracy and loss curves for both train and validation sets across all epochs
10. Load `best_model.h5` — reloads the best checkpoint saved during training
11. `evaluate_model()` — runs the model on the test set and prints final loss and accuracy
12. `plot_confusion_matrix_custom()` — generates predictions for the entire test set and displays the confusion matrix plus the full classification report
13. `predict_samples()` — shows 5 individual predictions with the X-ray image and the actual versus predicted grade

---

## 9. Evaluation and Results

After training, these are the key numbers to look at:

- Best validation accuracy (printed in the Training Summary): the highest accuracy achieved on the validation set across all 300 epochs
- Test accuracy: the accuracy on the held-out test set using the best checkpoint. This is the most honest measure of real-world performance
- Per-class F1 scores in the classification report: identifies which grades the model handles well and which it struggles with (grades 0 and 4 are typically easier because they are the extremes; grades 1, 2, and 3 are harder because they are adjacent)
- Confusion matrix: reveals systematic errors, such as whether the model tends to underestimate or overestimate severity

---

## 10. Visualization Modules

The file contains two separate visualization systems, both defined toward the bottom of the file.

### Visualization Module A: EM-CSA Internals

**Function: `visualize_em_csa_internals`**

This produces an 8-panel figure showing:

1. The original X-ray image
2. The Sobel edge detection output
3. The spatial attention map (mean activation across channels of the last convolutional layer)
4. The Grad-CAM overlay with the predicted class
5. Through 8. The four most activated individual filters in the last convolutional layer

This is useful for understanding what the attention mechanism has learned to look at.

### Visualization Module B: Full Dashboard

**Function: `visualize_em_csa_dashboard`**

This produces a more comprehensive 12-panel figure organized in a 3-row, 4-column grid:

- Row 1: Original X-ray, Sobel edge map, horizontal coordinate attention profile (line plot), vertical coordinate attention profile (line plot)
- Row 2: Spatial attention overlay (viridis colormap), Grad-CAM overlay with predicted class label
- Row 3: Top four most activated feature channels individually

---

## NOTE: Quick Reference: All Functions

| Function | What It Does |
|---|---|
| `setup_dataset()` | Uploads, unzips, and organizes the dataset |
| `apply_sobel(x)` | Applies Sobel edge detection to a tensor |
| `split_features(x, h, w)` | Splits a tensor into height and width halves |
| `global_mean_pool(x)` | Computes channel-wise mean across spatial dimensions |
| `global_max_pool(x)` | Computes channel-wise max across spatial dimensions |
| `edge_attention_block(input)` | Phase 1 of EM-CSA: edge-based feature emphasis |
| `coordinate_attention_part(input, ratio)` | Phase 3 of EM-CSA: axis-aware spatial attention |
| `spatial_attention_part(input)` | Phase 4 of EM-CSA: pixel-level spotlight |
| `em_csa_block(input, filters, ratio)` | Full EM-CSA block combining all four phases |
| `load_dataset(data_dir, ...)` | Loads and splits the dataset |
| `visualize_samples(dataset, ...)` | Displays sample images in a grid |
| `plot_class_distribution(dataset, ...)` | Plots bar chart of class counts |
| `plot_confusion_matrix_custom(model, ...)` | Prints confusion matrix and classification report |
| `plot_model_architecture(model, ...)` | Saves and displays the model graph |
| `build_augmented_model(...)` | Constructs the full CNN with EM-CSA |
| `train_model(model, ...)` | Trains with checkpointing and validation |
| `plot_training_history(history)` | Plots accuracy and loss curves |
| `evaluate_model(model, ...)` | Reports test loss and accuracy |
| `predict_samples(model, ...)` | Shows individual predictions with images |
| `smooth_resize(heatmap, ...)` | Upscales heatmap with Lanczos interpolation |
| `get_gradcam(model, ...)` | Computes Grad-CAM heatmap |
| `get_smooth_gradcam(model, ...)` | Computes Grad-CAM with Bicubic smoothing |
| `generate_overlay(img_path, ...)` | Blends heatmap onto original image |
| `visualize_em_csa_internals(model, ...)` | 8-panel internal attention visualization |
| `visualize_em_csa_dashboard(model, ...)` | 12-panel full visualization dashboard |

---

# EM-CSA Lung Cancer Detection System
## A Complete Step-by-Step Guide — IQ-OTH/NCCD Dataset

---

## Table of Contents

1. What This Project Does
2. Prerequisites and Environment
3. Understanding the Dataset
4. Step-by-Step Setup and Execution on Google Colab
5. How the Data Pipeline Works
6. Core Terminology: Basic Level
7. Core Terminology: Intermediate Level
8. Core Terminology: Advanced Level — The EM-CSA Mechanism
9. What Happens Inside `main()`: Step by Step
10. Evaluation and Reading the Results
11. Visualization: The 4-Panel Attention Dashboard
12. Key Differences from the Knee Arthritis Version

---

## 1. What This Project Does

This project trains a deep learning model to analyze lung CT scan slices and classify them into one of three categories:

- **Benign** — a non-cancerous abnormality is present, but it is not malignant
- **Malignant** — cancerous tissue is detected
- **Normal** — no abnormality is present

The dataset used is the **IQ-OTH/NCCD** (Iraqi Oncology Teaching Hospital / National Center for Cancer Diseases) lung cancer dataset, which contains CT scan slices collected from real patients.

The model does not just output a classification label. After training, it also generates visual explanations for its decisions — showing exactly which regions of the CT scan influenced the prediction. This is done through the same **EM-CSA (Edge-Enhanced Multi-Scale Coordinate-Spatial Attention)** mechanism used in the knee arthritis codebase, adapted here for pulmonary imaging.

A key practical feature: the code includes a **class balancing step** (oversampling) to handle the unequal distribution of Benign, Malignant, and Normal samples commonly found in medical datasets. Without this, the model would naturally learn to favor the majority class.

---

## 2. Prerequisites and Environment

### Where to Run

Run this code exclusively on **Google Colab** (colab.research.google.com). The code uses `google.colab.files` for uploading your dataset and test images, and it requires a GPU for training in a reasonable timeframe. No local Python installation or setup is required.

### What You Need Before Starting

- A Google account
- The IQ-OTH/NCCD dataset downloaded as a `.zip` file (available on Kaggle)
- No libraries to install — Colab has all dependencies pre-loaded

### Libraries Used

| Library | Purpose |
|---|---|
| `tensorflow` / `keras` | Model construction, training, and inference |
| `numpy` | Array and matrix operations |
| `matplotlib` | Plotting training curves, distributions, confusion matrix, visualizations |
| `sklearn` | `train_test_split`, `confusion_matrix`, `classification_report` |
| `opencv (cv2)` | Resizing attention maps to original image dimensions for overlay |
| `collections.Counter` | Counting class frequencies for oversampling logic |

---

## 3. Understanding the Dataset

### Source

The IQ-OTH/NCCD dataset contains 2D CT scan slices taken from lung cancer patients and healthy controls. Each slice is a grayscale image stored as `.jpg` or `.png`.

### Expected Folder Structure

Your zip file should contain three subfolders, one per class. The exact folder names become your class labels automatically. A typical structure looks like:

```
dataset.zip
└── Data/
    ├── Benign/
    │   ├── image001.jpg
    │   ├── image002.jpg
    │   └── ...
    ├── Malignant/
    │   ├── image001.jpg
    │   └── ...
    └── Normal/
        ├── image001.jpg
        └── ...
```

The code scans the unzipped contents to find the first directory that contains more than one subfolder and treats that as the root data directory. This means nested structures are handled automatically — you do not need to manually reorganize the zip.

### How Images Are Processed

All images are resized to **256 × 256 pixels** and loaded in **grayscale** (one color channel). Pixel values remain in the range [0, 255] during loading. Normalization to [0, 1] happens inside the model itself via the `Rescaling` layer.

### How the Data Is Split

The full dataset is split in two stages using **stratified splitting**, which preserves the original class ratios in each subset:

- First split: 85% for training+validation, 15% held out as the test set
- Second split: from the 85%, roughly 85% becomes training and 15% becomes validation

Final approximate proportions: **72% training, 13% validation, 15% test**.

---

## 4. Step-by-Step Setup and Execution on Google Colab

Follow these steps exactly, in order, without skipping any.

---

### Step 1: Open Google Colab

Go to [colab.research.google.com](https://colab.research.google.com). Sign in with your Google account. Click **New Notebook** to create a fresh notebook.

---

### Step 2: Enable GPU Runtime

Click **Runtime** in the top menu bar. Select **Change runtime type**. Under "Hardware accelerator", select **GPU** (T4 GPU is the free default). Click **Save**.

> **Why this matters:** Training for 50 epochs with a CNN on 256×256 images takes roughly 10–30 minutes on a GPU. On a CPU it could take several hours.

---

### Step 3: Upload and Paste the Code

Copy the entire contents of `Lungs.ipynb` into one or more cells in your Colab notebook. You can also upload the `.ipynb` file directly by going to **File > Upload Notebook**.

If you are copy-pasting, paste everything into a single large code cell. All the function definitions must be present in memory before `main()` is called.

---

### Step 4: Install Graphviz (Optional but Recommended)

The `plot_model_architecture` function requires Graphviz. Run this in a separate cell before executing the main code:

```python
!apt-get install -y graphviz
!pip install pydot
```

If you skip this step, you will see a warning message but training will still proceed normally.

---

### Step 5: Run the Cell

Run the code cell containing `main()`. As soon as you call `main()`:

**A file upload dialog will appear.** This is Colab prompting you to upload your zip file. Click the "Choose Files" button and select your IQ-OTH/NCCD dataset zip from your local computer. The upload will begin immediately.

> **Important:** Do not close the browser tab or refresh during the upload. Large zip files (the IQ-OTH/NCCD dataset is approximately 200–400MB) can take several minutes depending on your internet speed.

---

### Step 6: Wait for Unzipping and Dataset Detection

After the upload completes, the code will:

1. Unzip the contents into a folder called `lung_images/`
2. Walk the directory tree to find the folder that contains the three class subfolders
3. Print a message like `Using Data Directory: lung_images/Data` or similar

---

### Step 7: Watch the Class Detection Output

The code will print something like:

```
Classes: {'Benign': 0, 'Malignant': 1, 'Normal': 2}
```

This tells you the class names were detected correctly and assigned integer indices. The alphabetical ordering is used, so Benign=0, Malignant=1, Normal=2 is the standard output for this dataset. Keep note of this mapping — it is what the confusion matrix and classification report will use.

---

### Step 8: Observe the Oversampling Output

The training set is balanced by oversampling minority classes. You will see:

```
Balancing Training Data (Oversampling)...
```

After this, all three classes in the training set will have equal counts (equal to the count of whichever class had the most images). The validation and test sets are **not** oversampled — they retain the original distribution to give honest performance estimates.

---

### Step 9: Inspect the Sample Visualization

A 4×5 grid of 20 CT scan samples will be displayed. Each image is shown with its class label as the title. Visually verify that:

- Images look like recognizable CT scan slices (lung cross-sections)
- All three class labels appear in the grid
- Images are not all black, corrupted, or obviously wrong

If the images look unusual (all white, random noise, etc.), there may be an issue with the image decoding. JPEG images are expected; PNG images also work. Files with `.dcm` (DICOM format) extension will not be loaded by this code.

---

### Step 10: Inspect the Class Distribution Chart

A bar chart will display the class counts in the balanced training set. All three bars should be equal height after oversampling. The exact count on top of each bar represents how many images of that class the model will train on per epoch.

---

### Step 11: Review the Model Summary

The model architecture summary will be printed. Key things to look for:

- The total parameter count (typically in the range of 5–15 million for this architecture)
- The presence of `em_csa_cnn_lung` as the model name
- Layers named `attention_edge_out`, `attention_spatial_map`, and `last_conv_layer` — these names are critical for visualization later

If you installed Graphviz in Step 4, a visual graph of the model layers will also be displayed as an image.

---

### Step 12: Watch Training Progress

Training runs for **50 epochs** with a batch size of 16. After each epoch you will see a line like:

```
Epoch 12/50 — loss: 0.4821 — accuracy: 0.8213 — val_loss: 0.5102 — val_accuracy: 0.7944
```

Whenever `val_accuracy` reaches a new best, you will see:

```
Epoch 00012: val_accuracy improved from 0.78901 to 0.79440, saving model to best_lung_model.h5
```

This means the best version of the model is being saved automatically. Do not interrupt training at this point.

> **Note on epochs:** 50 epochs is significantly fewer than the knee arthritis version's 300 epochs. This is because the lung dataset converges faster and overtraining is more of a risk with the oversampled data.

---

### Step 13: Training Curves Are Plotted

After training finishes, two side-by-side plots appear:

- **Accuracy curve:** Training accuracy (blue) and validation accuracy (red) across all 50 epochs
- **Loss curve:** Training loss (blue) and validation loss (red) across all 50 epochs

Healthy training shows both curves declining together and eventually leveling off. If training accuracy keeps rising while validation accuracy plateaus or drops, that is overfitting.

---

### Step 14: Best Model Is Loaded and Evaluated

The code loads `best_lung_model.h5` (the checkpoint with the highest validation accuracy during training) and runs it on the held-out test set. You will see:

```
Test Accuracy: 84.37%
```

This is the most honest measure of real-world performance. The number cannot be inflated by training choices because the test images were never seen during training or validation.

---

### Step 15: Confusion Matrix and Classification Report

A confusion matrix is displayed as a color-coded grid. Then the classification report is printed. Read these together to understand where the model makes errors — see Section 10 for how to interpret these.

---

### Step 16: Attention Visualization on 3 Test Samples

Three random images are selected from the test set. For each one, a **4-panel attention figure** is generated showing the original CT scan, edge attention overlay, spatial attention overlay, and Grad-CAM heatmap. See Section 11 for a detailed explanation of each panel.

---

### Step 17: Custom Image Upload (Interactive)

The final step prompts you to upload your own CT scan image:

```
Please upload a CT Scan image for analysis...
```

A file picker dialog appears. Upload any grayscale CT lung scan image (`.jpg` or `.png`). The model will:

1. Display the image with its predicted class and confidence percentage
2. Generate the full 4-panel attention visualization for that image

This is the interactive inference mode — useful for testing the model on new scans outside the original dataset.

---

## 5. How the Data Pipeline Works

Understanding the data pipeline is important if you need to modify the code for a different dataset.

**Step 1 — Path collection:** The code walks the data directory and collects file paths and integer labels into two Python lists. No images are loaded into memory at this stage.

**Step 2 — Stratified split:** `sklearn.model_selection.train_test_split` with `stratify=labels` divides the paths and labels into train+val and test. This ensures each split has roughly the same class proportions as the full dataset.

**Step 3 — Oversampling:** For the training split only, each class is resampled up to the size of the largest class using `numpy.random.choice` with `replace=True`. This means minority class images are repeated. The resulting list is shuffled.

**Step 4 — tf.data.Dataset creation:** `create_dataset_from_paths` takes the path lists and creates a `tf.data.Dataset`. Images are loaded lazily (on demand) using `tf.io.read_file` and decoded with `tf.image.decode_jpeg`. They are resized to 256×256 inside the pipeline. This approach is memory-efficient: images are not all loaded at once.

**Step 5 — Batching:** The datasets are batched just before being passed to `model.fit()`. Batching is done separately from dataset creation so that unbatched datasets can be used for visualization and individual sample iteration.

---

## 6. Core Terminology: Basic Level

### CT Scan Slice

Unlike the X-ray images in the knee arthritis project, this dataset uses CT (Computed Tomography) scan slices. A CT scan is a series of cross-sectional images taken at different depths through the body. Each slice is a 2D image showing the internal structure at that depth. Lung cancer detection systems typically work on individual 2D slices rather than the full 3D volume.

### Three Classes Instead of Five

The knee arthritis project had 5 ordered grades (0 through 4). This project has 3 unordered categories: Benign, Malignant, and Normal. There is no inherent ordering — Benign is not "between" Normal and Malignant. This is a **multi-class classification** problem, not an ordinal regression problem.

### Class Imbalance

Medical datasets almost always have more Normal samples than disease samples, because most people in a given scan cohort are healthy. If you train a model on unbalanced data without correction, it learns to predict "Normal" too often simply because that minimizes the loss. Oversampling addresses this by repeating minority class examples until all classes are equally represented in training.

### Oversampling

Oversampling is the process of duplicating samples from underrepresented classes. In this code, if Malignant has 200 training images and Normal has 600, the Malignant images are randomly sampled (with replacement, meaning repeats are allowed) until there are 600 of them. This does not add new information — the same images appear multiple times — but it rebalances the loss contribution across classes.

### Stratified Split

When splitting data into train, validation, and test sets, stratification ensures the class proportions stay consistent. For example, if the original dataset is 20% Malignant, then the train set, validation set, and test set will each also be approximately 20% Malignant. This is important for getting reliable performance estimates.

### Epoch (50 here, not 300)

One epoch means the model has processed every image in the training set once. This code uses 50 epochs — fewer than the knee arthritis version — because the lung dataset converges faster and more epochs would risk memorizing the oversampled duplicates.

### Batch Size (16)

During each training step, 16 images are processed simultaneously. The model's weights are updated after each batch. This is the same batch size as the knee arthritis project.

### Adam Optimizer

This version uses **Adam** (Adaptive Moment Estimation) rather than RMSProp. Adam combines the benefits of two other optimizers (AdaGrad and RMSProp) and is generally considered more robust for medical imaging tasks with noisy gradients.

---

## 7. Core Terminology: Intermediate Level

### Same CNN Backbone

The convolutional backbone is identical to the knee arthritis version: five stacked blocks of Conv2D + MaxPooling with filter counts 8 → 16 → 32 → 64 → 128. The final convolutional layer is named `last_conv_layer` specifically so Grad-CAM can target it.

### Why Grayscale for CT Scans

CT scans, like X-rays, are naturally grayscale (intensity represents tissue density). Loading images with `channels=1` is correct here. Color CT scans exist but are pseudo-colored for visualization purposes — the underlying data is single-channel. Using grayscale reduces the number of parameters in the first convolutional layer.

### decode_jpeg vs. image_dataset_from_directory

The knee arthritis version used the high-level `image_dataset_from_directory` function. This version uses a lower-level approach: `tf.io.read_file` followed by `tf.image.decode_jpeg`. This gives more control over the loading process and works better with the manual oversampling logic that requires direct access to file path lists.

### Hard Sigmoid

Inside the coordinate attention block, a `hard_sigmoid` activation is used (instead of standard sigmoid). Hard sigmoid is a piecewise linear approximation: `clip((x + 3) / 6, 0, 1)`. It is computationally cheaper than the exponential operation inside true sigmoid and produces similar results in attention mechanisms.

### Sparse Categorical Crossentropy

This is the same loss function as the knee arthritis project. Labels are integers (0, 1, 2) rather than one-hot vectors, so "sparse" categorical crossentropy is appropriate. It computes the negative log probability assigned to the correct class.

### Model Checkpoint (saved as `best_lung_model.h5`)

The `ModelCheckpoint` callback monitors validation accuracy after each epoch. Only when validation accuracy improves is the model saved. This means the final saved model represents the best generalization achieved during training, not necessarily the model from the final epoch.

### Classification Report Metrics for 3 Classes

The classification report prints precision, recall, and F1-score for each of the three classes individually, plus a macro average (unweighted mean across classes) and a weighted average (weighted by class sample size in the test set). For a balanced evaluation of a medical classifier, the **macro F1** is the most important single number, because it treats all three classes equally.

---

## 8. Core Terminology: Advanced Level — The EM-CSA Mechanism

The EM-CSA block in this code is identical in structure to the knee arthritis version. The four-phase "Outline, Zoom, Align, Spotlight" sequence is applied in exactly the same way. What changes is the **context**: instead of looking for bone spurs and joint space narrowing in X-rays, the same mechanism is now identifying tumor boundaries, ground-glass opacities, and nodule textures in CT scan slices.

### Phase 1: Edge Attention — "Outline"

**Function: `edge_attention_block`**

The Sobel filter finds intensity gradients. In a lung CT, these correspond to the edges of the lung lobes, the boundary between the lung parenchyma and any lesion, and the wall of the trachea and major airways. The edge attention mask amplifies these boundaries in the feature map before subsequent processing.

The output of this phase is accessible by name (`attention_edge_out`) and is extracted for the visualization panel.

### Phase 2: Multi-Scale Extraction — "Zoom"

Two parallel convolutional branches process the edge-enhanced features:

- The **3×3 branch** captures fine-grained details: small nodules, subtle density changes, micro-texture
- The **7×7 branch** captures broader context: the overall shape and extent of a lesion, the position of a mass relative to the lung boundary

For pulmonary imaging this is especially valuable because malignant lesions can range from tiny sub-centimeter nodules to large masses that occupy significant portions of the lung field.

### Phase 3: Coordinate Attention — "Align"

Height-wise and width-wise pooling produce spatial attention maps that encode position information. In CT scans, this helps the model learn that certain regions — the central airways, the pleural surface, specific lobar positions — are more diagnostically significant. The coordinate attention learns to weight these regions appropriately.

### Phase 3.5: Adaptive Scale Fusion

The 3×3 and 7×7 branches are fused using learned softmax weights. If a particular feature channel is more informative at fine scale (e.g., for detecting small nodule texture), the fusion weights will favor the 3×3 branch for that channel. If a channel is more informative at coarse scale (e.g., for detecting a large mass), the 7×7 branch receives more weight. This is learned automatically during training.

### Phase 4: Spatial Attention — "Spotlight"

Channel-wise average and max pooling are concatenated and passed through a 7×7 convolution with sigmoid activation. The resulting map highlights specific pixel-level regions in the fused feature map. This is the most direct answer to "where in this CT slice should the model look?" and is extracted for visualization as `attention_spatial_map`.

### Grad-CAM in This Context

Grad-CAM targets `last_conv_layer` (the 128-filter conv layer just before the EM-CSA block). The heatmap shows which regions most strongly activated the predicted class score. For Malignant predictions, hot regions should ideally overlap with the actual tumor location. This provides a sanity check: if the Grad-CAM highlights the background or irrelevant tissue rather than a visible lesion, the model may be using spurious correlations.

---

## 9. What Happens Inside `main()`: Step by Step

When you call `main()`, the following sequence executes automatically:

1. **Upload & unzip** — Prompts for zip upload, unzips to `lung_images/`, walks the directory to find the data root
2. **Parse paths and labels** — Walks class folders, collects `.jpg`/`.png` file paths and integer labels
3. **Train/val/test split** — Stratified split: 15% test, then 15% of remainder as validation
4. **Oversample training set** — Balances all three classes to the count of the majority class
5. **Create tf.data.Dataset objects** — Three lazy-loading datasets: `train_ds`, `val_ds`, `test_ds`
6. **`visualize_samples`** — Displays 20 CT scan samples from the balanced training set
7. **`plot_class_distribution`** — Bar chart confirming equal class counts after oversampling
8. **`build_em_csa_model`** — Constructs the full model graph
9. **`model.summary()`** — Prints layer-by-layer parameter table
10. **`plot_model_architecture`** — Visual graph of model (requires Graphviz)
11. **`model.compile`** — Sets Adam optimizer, sparse categorical crossentropy loss, accuracy metric
12. **`model.fit`** — Trains for 50 epochs with ModelCheckpoint saving `best_lung_model.h5`
13. **`plot_training_history`** — Accuracy and loss curves for train and validation
14. **Load `best_lung_model.h5`** — Reloads the best checkpoint
15. **`best_model.evaluate`** — Prints test loss and test accuracy
16. **`plot_confusion_matrix_custom`** — Confusion matrix + classification report on test set
17. **Attention visualization loop** — Calls `visualize_internal_attention_and_gradcam` on 3 random test samples
18. **`predict_custom_image`** — Prompts for user image upload and runs full visualization

---

## 10. Evaluation and Reading the Results

### Test Accuracy

The headline number. Because the test set was never used during training or hyperparameter selection, this is the best estimate of how the model would perform on truly new scans. For the IQ-OTH/NCCD dataset, published results typically range from 80% to 95% depending on the architecture and preprocessing choices.

### Confusion Matrix

A 3×3 grid where rows represent the true class and columns represent the predicted class. The three diagonal cells show correct predictions. Pay particular attention to:

- **Malignant misclassified as Benign:** This is the most clinically dangerous error type (a false negative for cancer)
- **Normal misclassified as Malignant:** This is a false positive — causes unnecessary follow-up but is less dangerous than missing cancer
- **Benign vs. Malignant confusion:** These two are the hardest to distinguish visually; expect this cell to have the most off-diagonal errors

### Classification Report

Three key metrics per class:

- **Precision:** Of all the scans the model called Malignant, what fraction actually were? A low precision means many false alarms.
- **Recall:** Of all the actually Malignant scans, what fraction did the model find? A low recall means the model is missing cancers.
- **F1-score:** Harmonic mean of precision and recall. Use this as the single summary metric per class.

For a cancer detection system, **recall for the Malignant class** is the most safety-critical metric. Missing a malignant scan is more dangerous than raising a false alarm.

### Macro vs. Weighted Average

- **Macro average:** Treats all three classes equally. Better for evaluating performance on rare classes.
- **Weighted average:** Weights each class by how many test samples it has. Better for reflecting real-world class frequencies.

---

## 11. Visualization: The 4-Panel Attention Dashboard

After evaluation, `visualize_internal_attention_and_gradcam` is called for 3 test samples and 1 custom uploaded image. Each call produces a 1×4 figure:

**Panel 1 — Input CT Scan**
The original 256×256 grayscale CT slice, displayed with the predicted class label and confidence percentage at the top. For example: `Input: Malignant / Conf: 91.3%`

**Panel 2 — EM-CSA: Edge (Outline)**
The original CT image with the edge attention mask from `attention_edge_out` overlaid in jet colormap (blue→red) at 50% transparency. Bright colors indicate where Sobel-detected boundaries were amplified. In a lung CT, you should see bright regions along the lung wall, any visible lesion boundary, and the mediastinal structures.

**Panel 3 — EM-CSA: Spatial (Spotlight)**
The spatial attention map from `attention_spatial_map` overlaid in inferno colormap (black→yellow) at 60% transparency. This shows the final pixel-level focus of the attention mechanism. For Malignant predictions, bright spots should cluster over the region of abnormal tissue.

**Panel 4 — Grad-CAM (Final Decision)**
A gradient-weighted activation map overlaid on the original image using the jet colormap. Red/yellow regions are the most influential areas for the final classification decision. This is the most direct answer to "what made the model predict this class?"

Together, the four panels tell the diagnostic story: what edges were found (Panel 2), where the model focused (Panel 3), and what ultimately drove the decision (Panel 4).

---

## 12. Key Differences from the Knee Arthritis Version

| Aspect | Knee Arthritis | Lung Cancer (This Project) |
|---|---|---|
| Dataset | Custom knee X-ray dataset | IQ-OTH/NCCD CT scan dataset |
| Number of classes | 5 (ordered grades 0–4) | 3 (Benign, Malignant, Normal) |
| Image modality | X-ray | CT scan slice |
| Epochs | 300 | 50 |
| Optimizer | RMSProp | Adam |
| Data loading method | `image_dataset_from_directory` | Manual path collection + `tf.data` |
| Class balancing | Not present | Oversampling to majority class count |
| Splitting method | Single split | Two-stage stratified split |
| Model saved as | `best_model.h5` | `best_lung_model.h5` |
| Visualization panels | 4-panel (same structure) | 4-panel (same structure) |
| EM-CSA block | Identical | Identical |

The EM-CSA attention mechanism, the CNN backbone structure, and the visualization logic are fully shared between the two projects. The differences are entirely in the data pipeline, training configuration, and output classes.

---

## NOTE: Quick Reference: All Functions

| Function | What It Does |
|---|---|
| `apply_sobel(x)` | Sobel edge detection on a tensor (registered for serialization) |
| `split_features(x, h, w)` | Splits tensor along axis=1 into height and width halves |
| `global_mean_pool(x)` | Channel-wise mean, keepdims=True |
| `global_max_pool(x)` | Channel-wise max, keepdims=True |
| `edge_attention_block(input_tensor)` | Phase 1 (Outline): edge-based feature emphasis |
| `coordinate_attention_part(input_tensor, ratio)` | Phase 3 (Align): axis-aware spatial attention |
| `spatial_attention_part(input_tensor)` | Phase 4 (Spotlight): pixel-level spotlight |
| `em_csa_block(input_tensor, filters, ratio)` | Full EM-CSA block: all four phases + fusion |
| `build_em_csa_model(input_shape, num_classes, use_dense_layers)` | Constructs the complete CNN with EM-CSA |
| `make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)` | Computes Grad-CAM heatmap |
| `visualize_internal_attention_and_gradcam(model, img_path, image_size, class_names)` | 4-panel attention visualization |
| `create_dataset_from_paths(file_paths, labels, image_size, num_classes)` | Builds lazy-loading `tf.data.Dataset` |
| `visualize_samples(dataset, class_names, num_samples)` | Displays sample CT scan images in a grid |
| `plot_class_distribution(dataset, class_names, title)` | Bar chart of class counts |
| `plot_confusion_matrix_custom(model, dataset, class_names)` | Confusion matrix + classification report |
| `plot_training_history(history, save_path)` | Accuracy and loss curves |
| `plot_model_architecture(model, save_path)` | Visual graph of model layers |
| `predict_custom_image(model, class_names, image_size)` | Upload and predict a single custom CT scan |
| `main()` | Runs the complete pipeline end-to-end |

---

*This README covers the complete pipeline from environment setup through advanced architectural understanding. Reading it section by section, from basic to advanced, is the recommended approach for any new user of this codebase.*
