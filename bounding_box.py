'''Calculate IoU metrics for bounding boxes in video frames.'''
from datasets_multimodal import LaSOT
import numpy as np

def calc_iou(boxA, boxB):
    '''Given two bounding boxes, calculates the Intersection over Union (IoU) score.'''
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def get_bounding_box_score(target_frames, pred_frames, bounding_boxes):
    '''Calculates the IoU score for the bounding boxes of the target and predicted frames.'''
    t, c, h, w = pred_frames.shape
    scores = []
    pred_bboxes = []
    for i in range(t):
        # process each frame
        pred_frame = pred_frames[i]

        # Find the bounding box of the target frame
        bounding_box = bounding_boxes[i]
        gold_bounding_box = [bounding_box[0], bounding_box[1], bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]]

        # Find x1,y1,x2,y2 of the bounding box
        pred_frame = pred_frame.permute(1, 2, 0)
        pred_frame = pred_frame.cpu().numpy()
        if pred_frame.max() <= 1:
            pred_frame = pred_frame * 255
        red = pred_frame[:, :, 0]
        green = pred_frame[:, :, 1]
        blue = pred_frame[:, :, 2]

        red_indices = np.where((red == 255) & (green == 0) & (blue == 0))
        if len(red_indices[0]) == 0:
            red_indices = np.where((red >= 200) & (green < 10) & (blue < 10))
        if len(red_indices[0]) == 0:
            red_indices = np.where((red >= 100) & (green < 20) & (blue < 20))
        if len(red_indices[0]) == 0:
            red_indices = np.where((red >= 50) & (green < 50) & (blue < 50))
        if len(red_indices[0]) == 0:
            red_indices = np.where((red >= 10) & (green < 100) & (blue < 100))

        top_left = (np.min(red_indices[0]), np.min(red_indices[1]))
        bottom_right = (np.max(red_indices[0]), np.max(red_indices[1]))

        # correct for line thickness
        line_thickness = 1
        top_left = (top_left[0] - line_thickness, top_left[1] - line_thickness)
        bottom_right = (bottom_right[0] + line_thickness, bottom_right[1] + line_thickness)

        # x1, y1, x2, y2
        pred_bounding_box = [top_left[1], top_left[0], bottom_right[1], bottom_right[0]]
        pred_bboxes.append(pred_bounding_box)
        iou = calc_iou(gold_bounding_box, pred_bounding_box)
        scores.append(iou)
    return {
        "score": np.mean(scores),
        "bounding_boxes": pred_bboxes
    }


if __name__ == "__main__":
    ds = LaSOT(split="train", img_type="rgb", frame_size=64, input_frames=10)
    example = ds[0]
    print("IoU:", get_bounding_box_score(example["targets"], example["targets"], example["config"]["bounding_boxes"]))
