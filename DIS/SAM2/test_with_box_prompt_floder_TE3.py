import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import os


# Use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



# Paths to the folders
image_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE3/im'
box_prompt_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE3/gttext'

# output_folder =  "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny_promptBox/DIS-TE3"
# output_folder =  "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_B+_promptBox/DIS-TE3"
output_folder =  "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_L_promptBox/DIS-TE3"

# SAM-2
# sam2_checkpoint = "/home/pjl307/ZZJ/segment-anything-2-main/checkpoints/sam2_hiera_tiny.pt"
# model_cfg = "sam2_hiera_t.yaml"
# sam2_checkpoint = "/home/pjl307/ZZJ/segment-anything-2-main/checkpoints/sam2_hiera_base_plus.pt"
# model_cfg = "sam2_hiera_b+.yaml"
sam2_checkpoint = "/home/pjl307/ZZJ/segment-anything-2-main/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)



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




