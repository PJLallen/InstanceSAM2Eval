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


from torchvision.ops.boxes import box_area
from scipy.optimize import linear_sum_assignment
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 model evaluation")
    parser.add_argument('--annotation_path', type=str, required=True, help='Path to the annotation JSON file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image directory')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the results')
    parser.add_argument('--sam_checkpoint', type=str, required=True, help='Path to SAM checkpoint')
    parser.add_argument('--model_cfg', type=str, required=True, help='Path to model config YAML')
    return parser.parse_args()

# torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

# if torch.cuda.get_device_properties(0).major >= 8:
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            # cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1) 
            cv2.drawContours(img, contours, -1, (1,1,1,1), thickness=7) 

    ax.imshow(img)

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs can't be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

if __name__ == '__main__':
    args = parse_args()
    print(args)
    matplotlib.use('Agg')
    dataset_results, all_time = [], 0
    
    sam = sam_model_registry[args.model_cfg](checkpoint=args.sam_checkpoint)
    sam.to(device='cuda')

    mask_generator: SamAutomaticMaskGenerator = SamAutomaticMaskGenerator(sam, pred_iou_thresh=0.5)
    matcher = HungarianMatcher(cost_class=1, cost_bbox=1, cost_giou=1)

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
        masks = mask_generator.generate(image)

        plt.figure(figsize=(20,20))
        plt.imshow(image)
        show_anns(masks)
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