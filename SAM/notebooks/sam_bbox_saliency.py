import json
import os
import time

import matplotlib
import numpy as np
import pycocotools.mask as mask_util
import torch
import argparse

from PIL import Image
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

# ------------------------------ Setting ------------------------------
# ANNOTATION_PATH = r"/root/segment-anything-2/data/USIS10K/multi_class_annotations/multi_class_test_annotations.json"
# IMAGE_PATH = r"/root/segment-anything-2/data/USIS10K/test"
# sam_checkpoint = "/root/segment-anything-2/checkpoints/sam_vit_h_4b8939.pth"
# model_type = "vit_h"
# device = "cuda"
# ------------------------------ Setting ------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--annotation_path', type=str, required=True, default="/research/d1/rshr/ttzhang/d2/dataset/json/nc4k_test.json" ,help='Path to the annotation file')
    parser.add_argument('--image_path', type=str, required=True, default='/research/d1/rshr/ttzhang/d2/dataset/NC4K/test/image', help='Path to the image directory')
    parser.add_argument('--model_type', type=str, default='vit_h', help='Type of model to use')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation')
    parser.add_argument('--sam_checkpoint', type=str, required=True, default='../checkpoints/sam_vit_h.pth',help='Path to the SAM checkpoint file')
    parser.add_argument('--save_json', type=str, required=True, default='sam_auto_nc4k_huge.json', help='Filename to save JSON results')

    args = parser.parse_args()
    matplotlib.use('Agg')
    dataset_results, all_time = [], 0
    # ------------------------ Load SAM2 model ------------------------

    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    # ------------------------ Load COCO dataset -----------------------

    cocoGT = COCO(args.annotation_path)
    categories = cocoGT.dataset['categories']
    classes = dict([(ann["id"], ann["name"]) for ann in categories])
    print("categories:", classes)

    # ------------------------ Evaluate SAM model ----------------------

    for img_id in tqdm(cocoGT.getImgIds()):
        # --------------------- Load and embed image --------------------
        start = time.perf_counter()  # image encoder time begin
        img_dict = cocoGT.loadImgs(img_id)[0]
        file_name, height, width = img_dict["file_name"], img_dict["height"], img_dict["width"]
        image = np.array(Image.open(os.path.join(args.image_path, file_name)).convert("RGB"))
        predictor.set_image(image)
        end = time.perf_counter()
        all_time += end - start  # image encoder time ends
        # print("file_name:", file_name, "Image", image.shape, "height:", height, "width:", width)

        # --------------------- Use SAM to predict masks -----------------
        ann_ids = cocoGT.getAnnIds(imgIds=img_dict['id'], iscrowd=None)
        anns = cocoGT.loadAnns(ann_ids)

        for ann in anns:
            x, y, w, h = ann['bbox']
            input_box = np.array([x, y, x + w, y + h])
            start = time.perf_counter()  # prompt encoder and mask decoder time begins
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            end = time.perf_counter()
            all_time += end - start  # prompt encoder and mask decoder time ends
            masks, scores = masks[0], scores.tolist()[0]
            rle = mask_util.encode(np.array(masks[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
            rle['counts'] = rle["counts"].decode("utf-8")
            dataset_results.append({
                'image_id': ann['image_id'], 'category_id': ann['category_id'],
                'segmentation': rle, "score": float(scores)
            })

    # ------------------------- Save the results -------------------------
    with open(args.save_json, "w") as f:
        json.dump(dataset_results, f)

    # ------------------------- Evaluate the results ---------------------
    cocoDt = cocoGT.loadRes(args.save_json)
    cocoEval = COCOeval(cocoGT, cocoDt, "segm")
    cocoEval.evaluate()
    cocoEval.accumulate()
    print("Average time per image (FPS) is:", len(cocoGT.getImgIds()) / all_time)
    cocoEval.summarize()