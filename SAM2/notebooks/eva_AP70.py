import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

# def summarize_AP70(cocoEval):
#     stats = cocoEval.stats
#     print("AP @[ IoU=0.70 | area=all | maxDets=100 ] = {:.3f}".format(stats[1]))

def main():
    parser = argparse.ArgumentParser(description="Evaluate COCO results at AP 70")
    parser.add_argument('--json_file', type=str, required=True, help='Path to the JSON results file')
    parser.add_argument('--annotation_path', type=str, required=True, help='Path to the GT JSON results file')
    args = parser.parse_args()
    print(args)
    
    cocoGT = COCO(args.annotation_path)
    categories = cocoGT.dataset['categories']
    classes = dict([(ann["id"], ann["name"]) for ann in categories])
    print("categories:", classes)

    cocoDt = cocoGT.loadRes(args.json_file)
    cocoEval = COCOeval(cocoGT, cocoDt, "segm")
    cocoEval.params.iouThrs = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

    cocoEval.evaluate()
    cocoEval.accumulate()

    # 假设 all_time 是已定义的总处理时间
    all_time = 1.0  # 替换为实际的时间计算
    print("Average time per image (FPS) is:", len(cocoGT.getImgIds()) / all_time)

    cocoEval.summarize()
    ap70_index = list(cocoEval.params.iouThrs).index(0.7)
    ap70 = cocoEval.eval['precision'][ap70_index, :, :, 0, 2]  # 假设你想要计算的类别
    print(f"AP70: {np.mean(ap70):.3f}")

if __name__ == "__main__":
    main()