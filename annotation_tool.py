import cv2
import os
import numpy as np

# --- Config ---
image_folder = 'images'          # Folder with input images
mask_folder = 'mask'             # Folder to save binary masks
brush_size = 5                   # Brush size for annotation

# --- Setup ---
os.makedirs(mask_folder, exist_ok=True)
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()

drawing = False
mask = None
overlay = None

def draw(event, x, y, flags, param):
    global drawing, mask, overlay

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(mask, (x, y), brush_size, 255, -1)
            cv2.circle(overlay, (x, y), brush_size, (0, 0, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# --- Annotation Loop ---
for file in image_files:
    img_path = os.path.join(image_folder, file)
    image = cv2.imread(img_path)
    if image is None:
        continue

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    overlay = image.copy()

    cv2.namedWindow('Annotator')
    cv2.setMouseCallback('Annotator', draw)

    print(f"\nAnnotating: {file}")
    while True:
        display = cv2.addWeighted(image, 1.0, overlay, 0.5, 0)
        cv2.imshow('Annotator', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            mask_path = os.path.join(mask_folder, os.path.splitext(file)[0] + ".png")
            cv2.imwrite(mask_path, mask)
            print(f"Saved: {mask_path}")
        elif key == ord('n'):
            break
        elif key == ord('q'):
            exit()

cv2.destroyAllWindows()
