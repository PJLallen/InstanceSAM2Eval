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

from sam2.build_sam import build_sam2
from torchvision.ops.boxes import box_area
from scipy.optimize import linear_sum_assignment
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs
    # (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ANNOTATION_PATH = r"/root/segment-anything-2/data/USIS10K/foreground_annotations/foreground_test_annotations.json"
# IMAGE_PATH = r"/root/segment-anything-2/data/USIS10K/test"
ANNOTATION_PATH = r"/research/d1/rshr/ttzhang/d2/dataset/json/test2026.json"
IMAGE_PATH = r"/research/d1/rshr/ttzhang/d2/dataset/COD10K/Test_Image_CAM"

# sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
# model_cfg = "sam2_hiera_l.yaml"
# sam2_checkpoint = "../checkpoints/sam2_hiera_base_plus.pt"
# model_cfg = "sam2_hiera_b+.yaml"
sam2_checkpoint = "../checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"


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
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


if __name__ == '__main__':
    matplotlib.use('Agg')
    dataset_results, all_time = [], 0
    # ------------------------ Load SAM2 model ------------------------

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device='cuda', apply_postprocessing=False)
    mask_generator: SAM2AutomaticMaskGenerator = SAM2AutomaticMaskGenerator(sam2, pred_iou_thresh=0.5)
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
    matcher = HungarianMatcher(cost_class=1, cost_bbox=1, cost_giou=1)

    # ------------------------ Load COCO dataset -----------------------

    cocoGT = COCO(ANNOTATION_PATH)
    categories = cocoGT.dataset['categories']
    classes = dict([(ann["id"], ann["name"]) for ann in categories])
    print("categories:", classes)
    # assert len(classes) == 1, "Only one class (foreground) is supported."

    # ------------------------ Evaluate SAM2 model ----------------------

    for img_id in tqdm(cocoGT.getImgIds()):
        # --------------------- Use SAM2 to predict masks ----------------
        start = time.perf_counter()  # predict masks time begin
        img_dict = cocoGT.loadImgs(img_id)[0]
        file_name, height, width = img_dict["file_name"], img_dict["height"], img_dict["width"]
        image = np.array(Image.open(os.path.join(IMAGE_PATH, file_name)).convert("RGB"))
        masks = mask_generator.generate(image)
        end = time.perf_counter()
        all_time += end - start  # predict masks time ends

        if len(masks) == 0:
            print(f"No masks for {file_name}")
            continue

        # --------------------- Match predictions to targets --------------
        ann_ids = cocoGT.getAnnIds(imgIds=img_dict['id'], iscrowd=None)
        anns = cocoGT.loadAnns(ann_ids)
        targets = {"true_labels": [], "boxes": [], "labels": []}
        for ann in anns:
            x, y, w, h = ann['bbox']
            targets["boxes"].append([x, y, x + w, y + h])
            targets["true_labels"].append(ann['category_id'])
            targets["labels"].append(0)
        targets["boxes"] = torch.tensor(targets["boxes"], dtype=torch.float32)
        targets["labels"] = torch.tensor(targets["labels"], dtype=torch.int32)

        outputs = {"pred_logits": [], "pred_boxes": [], "mask": [], "scores": []}
        for mask_dict in masks:
            mask = mask_dict["segmentation"]
            score = mask_dict["predicted_iou"]
            x, y, w, h = bbox = mask_dict["bbox"]
            outputs["pred_boxes"].append([x, y, x + w, y + h])
            outputs["pred_logits"].append(0)
            outputs["mask"].append(mask)
            outputs["scores"].append(score)

        outputs["pred_boxes"] = torch.tensor(outputs["pred_boxes"], dtype=torch.float32).view(1, -1, 4)
        outputs["pred_logits"] = torch.tensor(outputs["pred_logits"], dtype=torch.float32).view(1, -1, 1)
        # print("targets:", targets)
        # print("outputs:", outputs)
        index_i, index_j = matcher(outputs, [targets])[0]
        for i, j in zip(index_i, index_j):
            mask, score = outputs["mask"][int(i)], outputs["scores"][int(i)]
            rle = mask_util.encode(np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
            rle['counts'] = rle["counts"].decode("utf-8")
            dataset_results.append({
                'image_id': img_id, 'category_id': targets["true_labels"][int(j)], 'segmentation': rle,
                "score": float(score)
            })

    # ------------------------- Save the results -------------------------
    with open("./SAM2_auto_eval_cod10k_tiny.json", "w") as f:
        json.dump(dataset_results, f)

    # ------------------------- Evaluate the results ---------------------
    cocoDt = cocoGT.loadRes("./SAM2_auto_eval_cod10k_tiny.json")
    cocoEval = COCOeval(cocoGT, cocoDt, "segm")
    cocoEval.evaluate()
    cocoEval.accumulate()
    print("Average time per image (FPS) is:", len(cocoGT.getImgIds()) / all_time)
    cocoEval.summarize()