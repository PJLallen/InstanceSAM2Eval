import argparse
import sys

sys.path.append("..")

from torch import nn

import os
import time
import json

import matplotlib
import numpy as np
import pycocotools.mask as mask_util
import torch

from PIL import Image
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2

from sam2.sam2_image_predictor import SAM2ImagePredictor

def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 model evaluation")
    parser.add_argument('--annotation_path', type=str, required=True, help='Path to the annotation JSON file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image directory')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the results')
    parser.add_argument('--sam2_checkpoint', type=str, required=True, help='Path to SAM2 checkpoint')
    parser.add_argument('--model_cfg', type=str, required=True, help='Path to model config YAML')
    return parser.parse_args()

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# def show_anns(anns, borders=True):
#     if len(anns) == 0:
#         return
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#     ax = plt.gca()
#     ax.set_autoscale_on(False)

#     img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
#     img[:,:,3] = 0
#     for ann in sorted_anns:
#         m = ann['segmentation']
#         color_mask = np.concatenate([np.random.random(3), [0.5]])
#         img[m] = color_mask 
#         if borders:
#             import cv2
#             contours, _ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
#             contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
#             cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1) 

#     ax.imshow(img)
def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        # mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
        mask_image = cv2.drawContours(mask_image , contours, -1, (1,1,1,1), thickness=7) 
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    


if __name__ == '__main__':
    args = parse_args()
    print(args)
    matplotlib.use('Agg')
    dataset_results, all_time = [], 0
    
    sam2 = build_sam2(args.model_cfg, args.sam2_checkpoint, device='cuda')
    predictor: SAM2ImagePredictor = SAM2ImagePredictor(sam2)

    cocoGT = COCO(args.annotation_path)
    categories = cocoGT.dataset['categories']
    classes = dict([(ann["id"], ann["name"]) for ann in categories])
    print("categories:", classes)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    for img_id in tqdm(cocoGT.getImgIds()):
        start = time.perf_counter()
        img_dict = cocoGT.loadImgs(img_id)[0]
        file_name, height, width = img_dict["file_name"], img_dict["height"], img_dict["width"]
        image = np.array(Image.open(os.path.join(args.image_path, file_name)).convert("RGB"))
        predictor.set_image(image)
        end = time.perf_counter()
        all_time += end - start # image encoder time ends

        # --------------------- Use SAM2 to predict masks ----------------
        ann_ids = cocoGT.getAnnIds(imgIds=img_dict['id'], iscrowd=None)
        anns = cocoGT.loadAnns(ann_ids)

        input_boxes = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            input_box = np.array([x, y, x + w, y + h])
            input_boxes.append(input_box)
        input_boxes = np.array(input_boxes)
        start = time.perf_counter() # prompt encoder and mask decoder time begins
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )


        plt.figure(figsize=(20,20))
        plt.imshow(image)
        for mask in masks:
            # print(mask.shape)
            if mask.ndim > 2:
                mask = mask.squeeze(0)
            show_mask(mask, plt.gca(), random_color=True)
        for box in input_boxes:
            show_box(box, plt.gca())
        plt.axis('off')
        save_img_path = os.path.join(args.save_path, file_name)
        directory = os.path.dirname(save_img_path)
        # 如果目录不存在，则创建
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_img_path)
                
        end = time.perf_counter()
        all_time += end - start

        if len(masks) == 0:
            print(f"No masks for {file_name}")
            continue