import cv2
import numpy as np
from datasets_multimodal import LaSOT

def calculate_color_metric(pred_frames):
    '''Color is a no-reference metric that is proposed to evaluate the colorfulness of an image.
    Hasler and Suesstrunk.
    It uses statistics calculated on A, B components in LAB colorspace. '''
    t, c, h, w = pred_frames.shape
    scores = []
    for i in range(t):

        # (c,h,w)
        pred_frame = pred_frames[i]

        # Convert to LAB
        pred_frame = pred_frame.permute(1, 2, 0)
        pred_frame = pred_frame.cpu().numpy()

        if pred_frame.max() <= 1:
            pred_frame = pred_frame * 255

        pred_frame = pred_frame.astype(np.uint8)
        pred_frame = cv2.cvtColor(pred_frame, cv2.COLOR_RGB2LAB)


        # Split Lab channels
        L, a, b = cv2.split(pred_frame)

        # Compute mean and standard deviation of a and b channels
        a_mean = np.mean(a)
        a_std = np.std(a)
        b_mean = np.mean(b)
        b_std = np.std(b)

        # Calculate color metric
        sigma_ab = np.sqrt(a_std ** 2 + b_std ** 2)
        mu_ab = np.sqrt(a_mean ** 2 + b_mean ** 2)

        # Calculate colorfulness
        return sigma_ab + 0.37 * mu_ab

# def calculate_warp_error(pred_frames, target_frames):
#     ''' Warp Error is a metric that evaluates the temporal stability of a video.
#     By warping one frame to another using corresponding optical flow, we can compare their differences. In our benchmark, we calculate WarpError on the A, B components in LAB colorspace.'''
#     t, c, h, w = pred_frames.shape
#     scores = []
#     for i in range(t):
#         pred_frame = pred_frames[i]
#         target_frame = target_frames[i]

#         # Convert to LAB
#         pred_frame = pred_frame.permute(1, 2, 0)
#         pred_frame = pred_frame.cpu().numpy().astype(np.uint8)
#         pred_frame = cv2.cvtColor(pred_frame, cv2.COLOR_RGB2LAB)

#         target_frame = target_frame.permute(1, 2, 0)
#         target_frame = target_frame.cpu().numpy().astype(np.uint8)
#         target_frame = cv2.cvtColor(target_frame, cv2.COLOR_RGB2LAB)

#         # Split Lab channels
#         L, a, b = cv2.split(pred_frame)
#         L_target, a_target, b_target = cv2.split(target_frame)

#         # Calculate optical flow
#         flow_a = cv2.calcOpticalFlowFarneback(a, a_target, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#         flow_b = cv2.calcOpticalFlowFarneback(b, b_target, None, 0.5, 3, 15, 3, 5, 1.2, 0)

#         # Warp a and b channels
#         a_warped = cv2.remap(a, flow_a, None, cv2.INTER_LINEAR)
#         b_warped = cv2.remap(b, flow_b, None, cv2.INTER_LINEAR)

#         # Calculate warp error
#         warp_error = np.mean(np.abs(a_warped - a_target) + np.abs(b_warped - b_target))

#         # Calculate warp error
#         return warp_error

import numpy as np
from scipy import stats


def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * stats.entropy(p, M) + 0.5 * stats.entropy(q, M)


def compute_JS_bgr(video_frames, dilation=1):
    """
    Compute the Jensen-Shannon divergence for each color channel between frames in a video sequence.

    Parameters:
    - video_frames (np.ndarray): Video sequence in the shape (T, C, H, W), where T is the number of frames,
                                 C is the number of color channels, H is the height, and W is the width.
    - dilation (int): Frame gap for computing divergence.

    Returns:
    - JS_b_list, JS_g_list, JS_r_list: Lists of Jensen-Shannon divergences for each color channel (B, G, R).
    """
    T, C, H, W = video_frames.shape

    hist_b_list = []   # [img1_histb, img2_histb, ...]
    hist_g_list = []
    hist_r_list = []

    for t in range(T):
        # print(os.path.join(input_dir, img_name))

        #img_in = cv2.imread(os.path.join(input_dir, img_name))
        img_in = video_frames[t]
        # img_in = img_in.permute(1, 2, 0) # no permute in numpy
        # swap (C, H, W) to (H, W, C)
        img_in = np.transpose(img_in, (1, 2, 0))
        H, W, C = img_in.shape

        hist_b = cv2.calcHist([img_in], [0], None, [256], [0,256]) # B
        hist_g = cv2.calcHist([img_in], [1], None, [256], [0,256]) # G
        hist_r = cv2.calcHist([img_in], [2], None, [256], [0,256]) # R

        hist_b = hist_b / (H * W)
        hist_g = hist_g / (H * W)
        hist_r = hist_r / (H * W)

        hist_b_list.append(hist_b)
        hist_g_list.append(hist_g)
        hist_r_list.append(hist_r)

    JS_b_list = []
    JS_g_list = []
    JS_r_list = []

    for i in range(len(hist_b_list)):
        if i + dilation > len(hist_b_list) - 1:
            break
        hist_b_img1 = hist_b_list[i]
        hist_b_img2 = hist_b_list[i + dilation]
        JS_b = JS_divergence(hist_b_img1, hist_b_img2)
        JS_b_list.append(JS_b)

        hist_g_img1 = hist_g_list[i]
        hist_g_img2 = hist_g_list[i+dilation]
        JS_g = JS_divergence(hist_g_img1, hist_g_img2)
        JS_g_list.append(JS_g)

        hist_r_img1 = hist_r_list[i]
        hist_r_img2 = hist_r_list[i+dilation]
        JS_r = JS_divergence(hist_r_img1, hist_r_img2)
        JS_r_list.append(JS_r)

    return JS_b_list, JS_g_list, JS_r_list


def calculate_cdc(video_sequence, dilation=[1, 2, 4], weight=[1/3, 1/3, 1/3]):
    """
    Calculate the color distribution consistency (CDC) for a video sequence.

    Parameters:
    - video_sequence (np.ndarray): Video sequence in the shape (T, C, H, W).
    - dilation (list of int): List of frame gaps for calculating JS divergence.
    - weight (list of float): List of weights corresponding to each dilation.

    Returns:
    - cdc (float): The color distribution consistency metric.
    """

    # Rescale the video sequence to [0, 255]
    if video_sequence.max() <= 1:
        video_sequence = video_sequence * 255
    video_sequence = video_sequence.cpu().numpy().astype(np.uint8)


    JS_b_mean, JS_g_mean, JS_r_mean = 0, 0, 0

    for d, w in zip(dilation, weight):
        JS_b_list, JS_g_list, JS_r_list = compute_JS_bgr(video_sequence, d)
        JS_b_mean += w * np.mean(JS_b_list)
        JS_g_mean += w * np.mean(JS_g_list)
        JS_r_mean += w * np.mean(JS_r_list)

    cdc = np.mean([JS_b_mean, JS_g_mean, JS_r_mean])
    return cdc

if __name__ == "__main__":
    # Example usage
    ds = LaSOT(split="train", img_type="rgb", frame_size=64, input_frames=10)
    pred_frames = ds[0]["targets"]
    target_frames = ds[0]["targets"]

    print("Colorfulness↑", calculate_color_metric(pred_frames))
    #print("Warp error↓", calculate_warp_error(pred_frames, target_frames))
    print("CDC↓", calculate_cdc(pred_frames))
