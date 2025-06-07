# Image Segmentation with U-Net

This project demonstrates a full pipeline for binary image segmentation using a custom-built U-Net model in PyTorch. It includes a manual image annotation tool, dataset preprocessing and augmentation, U-Net model architecture, training with mixed precision, and result visualization.

---

## ✍️ Annotation Tool

The annotation tool allows you to manually create binary masks for your images using a brush-based interface.

### Features:
- Left-click and drag to annotate regions of interest.
- Press:
  - `s` to save the binary mask.
  - `n` to proceed to the next image.
  - `q` to quit the annotation tool.

### How to Use:
1. Place your images in the `images/` folder.
2. Run the annotation script.
3. Annotated masks will be saved in the `mask/` folder.

---

## 🧠 Model Architecture

A simple yet effective [U-Net](https://arxiv.org/abs/1505.04597) is implemented using PyTorch for binary segmentation. The network consists of:
- Encoder (Downsampling path) with convolution and max-pooling layers.
- Bottleneck with deep feature representation.
- Decoder (Upsampling path) with transposed convolutions and skip connections.

---

## 🧪 Dataset Preparation

- Images and masks are resized to 256x256.
- Data augmentation is performed using `Albumentations`:
  - Horizontal Flip
  - Random Brightness/Contrast
  - Shift/Scale/Rotate
  - Normalization

Masks are treated as single-channel grayscale images (binary masks).

---

## 📦 Custom Dataset

A `SegmentationDataset` class loads the image-mask pairs and applies transformations. Both training and validation datasets are supported via `torch.utils.data.DataLoader`.

---

## ⚙️ Training

- **Loss Function**: Binary Cross Entropy with Logits (`BCEWithLogitsLoss`)
- **Optimizer**: Adam
- **Mixed Precision**: Enabled via `torch.cuda.amp`
- **Metrics**: Intersection over Union (IoU) via `sklearn.metrics.jaccard_score`

Training runs for a specified number of epochs (default: 5).

---

## 🧪 Evaluation

The model is evaluated on the validation set using IoU and loss curves. Segmentation outputs can be visualized using matplotlib.

---

## 🔍 Inference

To run inference:
1. Load an image.
2. Resize and normalize it.
3. Run it through the trained U-Net model.
4. Apply sigmoid + thresholding to obtain the binary mask.
5. Visualize the output.

---

## 🛠️ Dependencies

- Python 3.7+
- OpenCV (`cv2`)
- PyTorch
- Torchvision
- Albumentations
- PIL
- NumPy
- Matplotlib
- scikit-learn

You can install the dependencies with:

```bash
pip install -r requirements.txt
