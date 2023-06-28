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