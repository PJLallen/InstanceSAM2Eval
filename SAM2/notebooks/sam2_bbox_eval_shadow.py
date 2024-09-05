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
import argparse

from sam2.build_sam import build_sam2
from torchvision.ops.boxes import box_area
from scipy.optimize import linear_sum_assignment
from sam2.sam2_image_predictor import SAM2ImagePredictor

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs
    # (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ANNOTATION_PATH = r"/root/segment-anything-2/data/USIS10K/foreground_annotations/foreground_test_annotations.json"
# IMAGE_PATH = r"/root/segment-anything-2/data/USIS10K/test"
# ANNOTATION_PATH = r"/research/d1/rshr/ttzhang/d2/dataset/json/test2026.json"
# args.image_path = r"/research/d1/rshr/ttzhang/d2/dataset/COD10K/Test_Image_CAM"

# sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
# model_cfg = "sam2_hiera_l.yaml"
# sam2_checkpoint = "../checkpoints/sam2_hiera_base_plus.pt"
# model_cfg = "sam2_hiera_b+.yaml"
# sam2_checkpoint = "../checkpoints/sam2_hiera_small.pt"
# model_cfg = "sam2_hiera_s.yaml"





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--annotation_path', type=str, required=True, default="/research/d1/rshr/ttzhang/d2/dataset/json/nc4k_test.json" ,help='Path to the annotation file')
    parser.add_argument('--image_path', type=str, required=True, default='/research/d1/rshr/ttzhang/d2/dataset/NC4K/test/image', help='Path to the image directory')
    parser.add_argument('--model_cfg', type=str, default='vit_h', help='Type of model to use')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation')
    parser.add_argument('--sam2_checkpoint', type=str, required=True, default='../checkpoints/sam_vit_h.pth',help='Path to the SAM checkpoint file')
    parser.add_argument('--save_json', type=str, required=True, default='sam_auto_nc4k_huge.json', help='Filename to save JSON results')

    args = parser.parse_args()
    matplotlib.use('Agg')
    dataset_results, all_time = [], 0
    # ------------------------ Load SAM2 model ------------------------

    sam2_model = build_sam2(args.model_cfg, args.sam2_checkpoint, device="cuda")
    predictor: SAM2ImagePredictor = SAM2ImagePredictor(sam2_model)
    # mask_generator: SAM2AutomaticMaskGenerator = SAM2AutomaticMaskGenerator(
    # model=sam2,
    # points_per_side=64,
    # points_per_batch=128,
    # pred_iou_thresh=0.5,
    # stability_score_thresh=0.92,
    # stability_score_offset=0.7,
    # crop_n_layers=1,
    # box_nms_thresh=0.7,
    # crop_n_points_downscale_factor=2,
    # min_mask_region_area=25.0,
    # use_m2m=True,
    #     )

    # ------------------------ Load COCO dataset -----------------------

    cocoGT = COCO(args.annotation_path)
    categories = cocoGT.dataset['categories']
    classes = dict([(ann["id"], ann["name"]) for ann in categories])
    print("categories:", classes)
    # assert len(classes) == 1, "Only one class (foreground) is supported."

    # ------------------------ Evaluate SAM2 model ----------------------

    for img_id in tqdm(cocoGT.getImgIds()):
        # --------------------- Load and embed image --------------------
        start = time.perf_counter() # image encoder time begin
        img_dict = cocoGT.loadImgs(img_id)[0]
        file_name, height, width = img_dict["file_name"], img_dict["height"], img_dict["width"]
        image = np.array(Image.open(os.path.join(args.image_path, file_name)).convert("RGB"))
        predictor.set_image(image)
        end = time.perf_counter()
        all_time += end - start # image encoder time ends

        # --------------------- Use SAM2 to predict masks ----------------
        ann_ids = cocoGT.getAnnIds(imgIds=img_dict['id'], iscrowd=None)
        anns = cocoGT.loadAnns(ann_ids)

        for ann in anns:
            x, y, w, h = ann['bbox']
            input_box = np.array([x, y, x + w, y + h])
            start = time.perf_counter() # prompt encoder and mask decoder time begins
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            end = time.perf_counter()
            all_time += end - start # prompt encoder and mask decoder time ends
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