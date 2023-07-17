from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

def maskoff(img, blur=(5,5), threshold=10, mask_mode="exact", return_mask=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, blur, 0)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask = np.zeros_like(img)  # masking to remove any logos / extraneous stuff cl
    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # get contour with largest radius
    max_rad = [0, 0, 0]
    contour_list = []
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius > max_rad[-1]:
            max_rad = [x, y, radius]
            contour_list = [contour]  # only keep the largest contour
    # _ = cv2.fillPoly(mask, pts =contour_list, color=(1,1,1))
    # img = img * mask # apply mask
    # crop around this contour
    # use minEnclosing circle parameters as found above
    cx, cy, cr = max_rad
    # fit circle using RANSAC to the raw contour found above - fails in many insta
    # ransac = CircleRANSAC(contour_list[0][:,0,0], contour_list[0][:,0,1], 50)
    # ransac.execute_ransac()
    # cx, cy, cr = ransac.best_model[0], ransac.best_model[1], ransac.best_model[2
    # convert to int
    cr = int(cr)
    cy = int(cy)
    cx = int(cx)
    if mask_mode == "exact":
        _ = cv2.fillPoly(mask, contour_list, (255, 255, 255))  # get exact mask
    else:
        _ = cv2.circle(mask, (cx, cy), cr, (255, 255, 255), -1)  # filled circle
    img = img * (mask == 255)
    if return_mask:
        return img, mask
    return img

def smoothed_gray(img, blur_filter):
    _img = (img * 255.).astype(np.uint8)
    gray = cv2.cvtColor(_img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, blur_filter, 0)
    return gray

def resize_contour_crop(
    img: np.ndarray,
    image_size: Tuple[int, ...],
    threshold: int = 10,
    blur: Tuple[int, int] = (5, 5),
    mode: str = "crop",
    interpolation: str = "area",
    return_mask: bool = False,
    mask_mode: str = "circle",
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """
    For final resize step
    mode: crop - will crop height or widht to match shorter side (resulting image looks slightly zoomed in)
    mode: pad  - will pad shorter side to match longer side (resulting image looks slightly zoomed out)

    interpolation for cv2.resize: defaults to 'area', but can accept 'linear'
    """
    scale = image_size[0] // 2
    h, w = img.shape[:2]
    # threshold to binary
    gray = smoothed_gray(img, blur)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    mask = np.zeros_like(img)  # masking to remove any logos / extraneous stuff close to fundus
    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # get contour with largest radius
    max_rad = [0, 0, 0]
    contour_list = []
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius > max_rad[-1]:
            max_rad = [x, y, radius]
            contour_list = [contour]  # only keep the largest contour

    # use minEnclosing circle parameters as found above
    cx, cy, cr = max_rad

    # convert to int
    cr = int(cr)
    cy = int(cy)
    cx = int(cx)

    if mask_mode == "exact":
        _ = cv2.fillPoly(mask, contour_list, (255, 255, 255))  # get exact mask
    else:
        _ = cv2.circle(mask, (cx, cy), cr, (255, 255, 255), -1)  # filled circle

    img = img * (mask == 255)

    cropped_img = img[max(0, cy - cr) : min(h, cy + cr), max(0, cx - cr) : min(w, cx + cr)]
    cropped_mask = mask[max(0, cy - cr) : min(h, cy + cr), max(0, cx - cr) : min(w, cx + cr)]
    nh, nw = cropped_img.shape[:2]
    # resize with aspect ratio
    """
    if interpolation == "linear":
        interp = cv2.INTER_LINEAR
    elif interpolation == "area":
        interp = cv2.INTER_AREA
    else:
        raise ValueError("Interpolation method must be linear or area")
    """
    if image_size[0] <= h:
        # intending to shrink the image
        interp = cv2.INTER_AREA
    else:
        # intending the enlarge the image
        interp = cv2.INTER_LINEAR

    if mode == "crop":
        if nh < nw:
            oh = int(scale * 2)
            ow = int(nw * float(oh) / float(nh))
        else:
            ow = int(scale * 2)
            oh = int(nh * float(ow) / float(nw))

        scaled_img = cv2.resize(cropped_img, (ow, oh), interpolation=interp)
        scaled_mask = cv2.resize(cropped_mask, (ow, oh), interpolation=interp)
        # crop around to make square
        scaled_img = scaled_img[
            oh // 2 - scale : oh // 2 + scale, ow // 2 - scale : ow // 2 + scale
        ]
        scaled_mask = scaled_mask[
            oh // 2 - scale : oh // 2 + scale, ow // 2 - scale : ow // 2 + scale
        ]

    elif mode == "pad":
        if nh > nw:
            oh = int(scale * 2)
            ow = int(nw * float(oh) / float(nh))
        else:
            ow = int(scale * 2)
            oh = int(nh * float(ow) / float(nw))

        scaled_img = cv2.resize(cropped_img, (ow, oh), interpolation=interp)
        padded_img = np.zeros((int(scale * 2), int(scale * 2), 3), dtype=float)
        padded_img[
            scale - oh // 2 : scale - oh // 2 + oh, scale - ow // 2 : scale - ow // 2 + ow
        ] = scaled_img

        scaled_img = padded_img

        scaled_mask = cv2.resize(cropped_mask, (ow, oh), interpolation=interp)
        padded_mask = np.zeros((int(scale * 2), int(scale * 2), 3), dtype=np.uint8)
        padded_mask[
            scale - oh // 2 : scale - oh // 2 + oh, scale - ow // 2 : scale - ow // 2 + ow
        ] = scaled_mask

        scaled_mask = padded_mask

    if return_mask:
        return scaled_img, {
            "mask": scaled_mask,
        }
    else:
        return scaled_img

def square_crop(
    img: np.ndarray,
    threshold: int = 10,
    blur: Tuple[int, int] = (5, 5),
    return_mask: bool = False,
    mask_mode: str = "circle",
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """
    For final resize step
    mode: crop - will crop height or widht to match shorter side (resulting image looks slightly zoomed in)
    mode: pad  - will pad shorter side to match longer side (resulting image looks slightly zoomed out)

    interpolation for cv2.resize: defaults to 'area', but can accept 'linear'
    """
    h, w = img.shape[:2]
    # threshold to binary
    gray = smoothed_gray(img, blur)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    mask = np.zeros_like(img)  # masking to remove any logos / extraneous stuff close to fundus
    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # get contour with largest radius
    max_rad = [0, 0, 0]
    contour_list = []
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius > max_rad[-1]:
            max_rad = [x, y, radius]
            contour_list = [contour]  # only keep the largest contour

    # use minEnclosing circle parameters as found above
    cx, cy, cr = max_rad

    # convert to int
    cr = int(cr)
    cy = int(cy)
    cx = int(cx)

    if mask_mode == "exact":
        _ = cv2.fillPoly(mask, contour_list, (255, 255, 255))  # get exact mask
    else:
        _ = cv2.circle(mask, (cx, cy), cr, (255, 255, 255), -1)  # filled circle

    img = img * (mask == 255)

    cropped_img = img[max(0, cy - cr) : min(h, cy + cr), max(0, cx - cr) : min(w, cx + cr)]
    cropped_mask = mask[max(0, cy - cr) : min(h, cy + cr), max(0, cx - cr) : min(w, cx + cr)]
    nh, nw = cropped_img.shape[:2]
    
    if nh > nw:
        padded_img = np.zeros((nh, nh, 3), dtype=np.uint8)
        padded_img[:, nh // 2 - nw // 2 : nh // 2 - nw // 2 + nw] = cropped_img
    else:
        padded_img = np.zeros((nw, nw, 3), dtype=np.uint8)
        padded_img[nw // 2 - nh // 2 : nw // 2 - nh // 2 + nh, :] = cropped_img

    return padded_img