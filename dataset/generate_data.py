import os
import numpy as np
import cv2
from tqdm import tqdm
import random

# --- Configuration ---
num_images = 10
img_size = (256, 256)
output_image_folder = "test_images/image"
output_mask_folder = "test_images/masks"

# --- Create folders ---
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

# --- Generation loop ---
for i in tqdm(range(num_images), desc="Generating synthetic data"):
    # Blank canvas
    image = np.zeros(img_size, dtype=np.uint8)
    mask = np.zeros(img_size, dtype=np.uint8)

    # Add 1–3 random ellipses (simulate organs/tumors)
    for _ in range(random.randint(1, 3)):
        center = (random.randint(50, 200), random.randint(50, 200))
        axes = (random.randint(10, 40), random.randint(10, 40))
        angle = random.randint(0, 180)

        color_intensity = random.randint(80, 150)
        cv2.ellipse(image, center, axes, angle, 0, 360, color_intensity, -1)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 1, -1)

    # Add Gaussian noise
    noise = np.random.normal(0, 10, img_size).astype(np.uint8)
    noisy_image = cv2.add(image, noise)

    # Smooth it
    final_image = cv2.GaussianBlur(noisy_image, (5, 5), 0)

    # Save image and mask
    img_path = os.path.join(output_image_folder, f"image_{i:03d}.png")
    mask_path = os.path.join(output_mask_folder, f"mask_{i:03d}.png")
    cv2.imwrite(img_path, final_image)
    cv2.imwrite(mask_path, mask * 255)  # Make mask visible (0 or 255)

print(f"\n✅ Generated {num_images} synthetic images and masks in:")
print(f"   {output_image_folder}/")
print(f"   {output_mask_folder}/")
