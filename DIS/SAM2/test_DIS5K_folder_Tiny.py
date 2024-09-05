import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.95]])
        img[m] = color_mask
    ax.imshow(img)


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def write_masks_to_folder(masks, save_path):
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(save_path, filename), mask * 255)


def process_images_in_folder(input_folder, output_folder):
    # sam2_checkpoint = "/home/pjl307/ZZJ/segment-anything-2-main/checkpoints/sam2_hiera_large.pt"
    # model_cfg = "sam2_hiera_l.yaml"

    sam2_checkpoint = "/home/pjl307/ZZJ/segment-anything-2-main/checkpoints/sam2_hiera_tiny.pt"
    model_cfg = "sam2_hiera_t.yaml"



    sam2 = build_sam2(model_cfg, sam2_checkpoint, device='cuda', apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path).resize((1024, 1024))
            image = np.array(image.convert("RGB"))

            masks = mask_generator.generate(image)
            image_output_folder = os.path.join(output_folder, os.path.splitext(filename)[0])
            if not os.path.exists(image_output_folder):
                os.makedirs(image_output_folder)

            write_masks_to_folder(masks, image_output_folder)

# Set the paths to your input and output folders
input_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-VD/im"
output_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-VD"
# Process all images in the input folder
process_images_in_folder(input_folder, output_folder)


# Set the paths to your input and output folders
input_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE1/im"
output_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-TE1"
# Process all images in the input folder
process_images_in_folder(input_folder, output_folder)

# Set the paths to your input and output folders
input_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE2/im"
output_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-TE2"
# Process all images in the input folder
process_images_in_folder(input_folder, output_folder)

# Set the paths to your input and output folders
input_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE3/im"
output_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-TE3"
# Process all images in the input folder
process_images_in_folder(input_folder, output_folder)

# Set the paths to your input and output folders
input_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE4/im"
output_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-TE4"
# Process all images in the input folder
process_images_in_folder(input_folder, output_folder)

