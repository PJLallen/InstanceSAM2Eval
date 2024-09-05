import os
import numpy as np
import torch
import cv2
import sys

# Paths to the folders
image_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE4/im'
box_prompt_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE4/gttext'
output_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM_Results/ViT-HH-GT-Box-Prompt/DIS-TE4'

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

# Model setup
# sam_checkpoint = "sam_vit_b_01ec64.pth"
# model_type = "vit_b"

# sam_checkpoint = "sam_vit_l_0b3195.pth"
# model_type = "vit_l"

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"


device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process each image in the image folder
for image_filename in os.listdir(image_folder):
    if image_filename.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
        image_path = os.path.join(image_folder, image_filename)

        # Load and process the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        # Load the corresponding box prompt
        box_prompt_filename = os.path.splitext(image_filename)[0] + '.txt'
        box_prompt_path = os.path.join(box_prompt_folder, box_prompt_filename)

        if os.path.exists(box_prompt_path):
            with open(box_prompt_path, 'r') as file:
                # Read the box prompt and convert it to a numpy array
                input_box = np.array([list(map(int, file.readline().strip().split(',')))])
        else:
            print(f"Box prompt for {image_filename} not found, skipping.")
            continue

        # Define input labels (1 for foreground, 0 for background)
        input_label = np.array([1])

        # Predict the mask
        masks, scores, logits = predictor.predict(
            point_labels=input_label,
            box=input_box,
            multimask_output=False,
        )

        # 只有一个mask，直接保存,为png格式
        mask = masks[0]
        mask = (mask * 255).astype(np.uint8)
        mask_filename = os.path.splitext(image_filename)[0] + '.png'
        mask_path = os.path.join(output_folder, mask_filename)
        cv2.imwrite(mask_path, mask)




