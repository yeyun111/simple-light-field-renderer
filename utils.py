"""
Utils for basic image processing 
"""

import numpy
import cv2
from matplotlib import pyplot

DEFAULT_LONG_EDGE_LIMIT = 1280
FLANN_INDEX_LSH = 6
ROI_RATIO = 0.5
DEPTH_MAP_SHORT_EDGE_SIZE = 128
DEFAULT_SHIFT_RANGE = (-1, 2)


def get_edges_from_triangles(triangles):
    edges = []
    for triangle in triangles:
        a, b, c = triangle
        edge0 = (a, b) if a < b else (b, a)
        edge1 = (b, c) if b < c else (c, b)
        edge2 = (a, c) if a < c else (c, a)
        edges += [edge0, edge1, edge2]
    return list(set(edges))


def limit_image_size(img, long_edge=DEFAULT_LONG_EDGE_LIMIT):
    h, w = img.shape[:2]
    fxy = long_edge / max(h, w)
    if fxy < 1:
        img = cv2.resize(img, (0, 0), fx=fxy, fy=fxy)
    return img


def calibrate_rois(rois):
    orb = cv2.ORB_create(nfeatures=200, nlevels=1)
    index_params = dict(algorithm=FLANN_INDEX_LSH, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    kp_n_des_list = [orb.detectAndCompute(x, None) for x in rois]

    kp0, des0 = kp_n_des_list[0]

    mats = []
    for kp, des in kp_n_des_list[1:]:
        matches = flann.knnMatch(des0, des, k=2)
        good_matches = []
        for match in matches:
            if len(match) == 2 and match[0].distance < 0.7 * match[1].distance:
                good_matches.append(match[0])
        dst_pts = numpy.float32([kp0[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        src_pts = numpy.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        mat_affine = cv2.estimateRigidTransform(src_pts, dst_pts, False)
        mats.append(mat_affine)

    return mats


def calibrate_images(images):
    # check if images are same size
    h_ref, w_ref = images[0].shape[:2]
    for image in images[1:]:
        h, w = image.shape[:2]
        if h != h_ref or w != w_ref:
            print('Bad inputs!')
            return None

    cx = w_ref / 2
    cy = h_ref / 2
    short_edge = min(h_ref, w_ref)
    coeff = ROI_RATIO / 2
    x0_roi = int(cx - coeff * short_edge + 0.5)
    x1_roi = int(cx + coeff * short_edge + 0.5)
    y0_roi = int(cy - coeff * short_edge + 0.5)
    y1_roi = int(cy + coeff * short_edge + 0.5)

    calib_rois = [cv2.cvtColor(x[y0_roi:y1_roi+1, x0_roi:x1_roi+1], cv2.COLOR_BGR2GRAY) for x in images]
    affine_mats = calibrate_rois(calib_rois)

    coords = [[0., 0.]]
    for i, m in enumerate(affine_mats):
        images[i + 1] = cv2.warpAffine(images[i + 1], m, (w_ref, h_ref))
        coords.append([m[0][2], m[1][2]])

    coords = numpy.array(coords)
    coords -= numpy.mean(coords, axis=0)

    return images, coords


def cal_depth_map(images, coords, short_edge=DEPTH_MAP_SHORT_EDGE_SIZE, shift_range=DEFAULT_SHIFT_RANGE):
    # check if images are same size
    h_ref, w_ref = images[0].shape[:2]
    for image in images[1:]:
        h, w = image.shape[:2]
        if h != h_ref or w != w_ref:
            print('Bad inputs!')
            return None

    scale = short_edge / min(h_ref, w_ref)
    imgs = []
    if scale < 1:
        for i in range(len(images)):
            imgs.append(cv2.resize(images[i], (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR))
    else:
        scale = 1.

    dcoords = [x * scale for x in coords]
    shifts = numpy.linspace(*shift_range, 100)

    depth_map = numpy.zeros(imgs[0].shape[:2], dtype=numpy.float32)
    min_var_map = numpy.ones(imgs[0].shape[:2], dtype=numpy.float32) * 1e9

    dumb_mat = numpy.array([
        [1, 0],
        [0, 1]
    ])

    h0, w0 = imgs[0].shape[:2]
    still_pixs = numpy.ones(depth_map.shape, dtype=numpy.uint8)
    for i, shift in enumerate(shifts):
        mats = [numpy.hstack([dumb_mat, shift * dcoord.reshape(2, 1)]) for dcoord in dcoords]
        shifted_imgs = [cv2.warpAffine(img, m, (w0, h0)) for img, m in zip(imgs, mats)]
        var_map = numpy.sum(numpy.var(shifted_imgs, axis=0), axis=2)
        prev_depth_map = depth_map.copy()
        depth_map[var_map < min_var_map] = shift / scale
        if i > 0:
            still_pixs[depth_map != prev_depth_map] = 0

        min_var_map = numpy.min([min_var_map, var_map], axis=0)

    # Try to fix never update pixels ...
    blurred_depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0, borderType=cv2.BORDER_REFLECT101)
    depth_map[still_pixs == 1] = blurred_depth_map[still_pixs == 1]

    depth_map = cv2.resize(depth_map, (w_ref, h_ref))
    return depth_map
