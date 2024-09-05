import json
import os
import time

import matplotlib
import numpy as np
import pycocotools.mask as mask_util
import torch

from PIL import Image
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs
    # (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# -------------------------- Setting --------------------------
ANNOTATION_PATH = r"/research/d1/rshr/ttzhang/d2/dataset/json/nc4k_test.json"
IMAGE_PATH = r"/research/d1/rshr/ttzhang/d2/dataset/NC4K/test/image"
sam2_checkpoint = "../checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"
 # ------------------------- Setting --------------------------

if __name__ == '__main__':
    matplotlib.use('Agg')
    dataset_results, all_time = [], 0
    # ------------------------ Load SAM2 model ------------------------

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor: SAM2ImagePredictor = SAM2ImagePredictor(sam2_model)

    # ------------------------ Load COCO dataset -----------------------

    cocoGT = COCO(ANNOTATION_PATH)
    categories = cocoGT.dataset['categories']
    classes = dict([(ann["id"], ann["name"]) for ann in categories])
    print("categories:", classes)

    # ------------------------ Evaluate SAM2 model ----------------------

    for img_id in tqdm(cocoGT.getImgIds()):
        # --------------------- Load and embed image --------------------
        start = time.perf_counter() # image encoder time begin
        img_dict = cocoGT.loadImgs(img_id)[0]
        file_name, height, width = img_dict["file_name"], img_dict["height"], img_dict["width"]
        image = np.array(Image.open(os.path.join(IMAGE_PATH, file_name)).convert("RGB"))
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


    # # ------------------------- Save the results -------------------------
    with open("SAM2_bbox_nc4k_tiny.json", "w") as f:
        json.dump(dataset_results, f)

    # # ------------------------- Evaluate the results ---------------------
    cocoDt = cocoGT.loadRes("SAM2_bbox_nc4k_tiny.json")
    cocoEval = COCOeval(cocoGT, cocoDt, "segm")
    cocoEval.evaluate()
    cocoEval.accumulate()
    print("Average time per image (FPS) is:", len(cocoGT.getImgIds()) / all_time)
    cocoEval.summarize()