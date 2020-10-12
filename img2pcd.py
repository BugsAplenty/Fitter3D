#!/usr/bin/env python3

import cv2
import open3d as o3d
import numpy as np

target_img_path = "./chair_real.jpg"
target_img = cv2.imread(target_img_path)


def crop_to_shape(img, threshold):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(img_grey, (13, 13), 0)
    shape_isolated = cv2.Canny(img_blurred, 40, 40)

    x_object_min = np.min(np.where(shape_isolated > 0)[0])
    x_object_max = np.max(np.where(shape_isolated > 0)[0])
    y_object_min = np.min(np.where(shape_isolated > 0)[1])
    y_object_max = np.max(np.where(shape_isolated > 0)[1])

    cropped = shape_isolated[x_object_min:x_object_max, y_object_min:y_object_max]
    return cropped


def pad_to_square(img):
    min_dim = np.minimum(img.shape[0], img.shape[1])
    max_dim = np.maximum(img.shape[0], img.shape[1])

    if min_dim == img.shape[0]:
        img_padded = cv2.copyMakeBorder(img, int((max_dim - min_dim) / 2),
                                        int(np.ceil((max_dim - min_dim) / 2)), 0, 0,
                                        cv2.BORDER_CONSTANT)
    else:
        img_padded = cv2.copyMakeBorder(img, 0, 0, int((max_dim - min_dim) / 2),
                                        int(np.ceil((max_dim - min_dim) / 2)),
                                        cv2.BORDER_CONSTANT)
    return img_padded


def img_to_points(img):
    points = np.argwhere(img == 255)
    points_x = points[:, 0]
    points_y = points[:, 1]
    points_z = np.zeros(np.size(points_x))
    points_xyz = np.zeros((np.size(points_x), 3))
    points_xyz[:, 0] = points_x
    points_xyz[:, 1] = points_y
    points_xyz[:, 2] = points_z
    return points_xyz


target_img_cropped = crop_to_shape(target_img, 220)
target_img_padded = pad_to_square(target_img_cropped)

points = img_to_points(target_img_padded)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd_downsampled = pcd.uniform_down_sample(every_k_points=3)
o3d.io.write_point_cloud("./chair_flat_points.pcd", pcd_downsampled)

