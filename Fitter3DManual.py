#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider, Button
import open3d as o3d
import numpy as np
import cv2
import copy

matplotlib.use('TkAgg')
model_pcd = o3d.io.read_point_cloud("chair_points.pcd")
image_pcd = o3d.io.read_point_cloud("chair_flat_points.pcd")


class Fitter3DManual:
    def __init__(self, model_pcd, image_pcd):
        self._figheight = 10
        self._figwidth = 10

        self.model_points = np.asarray(model_pcd.points)
        self.model_points = self.model_points - np.mean(self.model_points, axis=0)
        self.image_points = np.asarray(image_pcd.points)
        self.cam_mat = np.array([[1, 0, 1800], [0, 1, 1800], [0, 0, 1]], dtype=np.float32)

        self.rvec = np.array([0, 0, 0], dtype=np.float32)
        self.tvec = np.array([0, 0, 0], dtype=np.float32)
        self.distortion = np.array([0, 0, 0, 0, 0], dtype=np.float32)

        self.pcd_transformed_2d = cv2.projectPoints(self.model_points, self.rvec, self.tvec, self.cam_mat,
                                                    self.distortion)

        self.slider_center_x = None
        self.slider_center_y = None
        self.slider_focal_length = None
        self.slider_rot_x = None
        self.slider_rot_y = None
        self.slider_rot_z = None
        self.slider_trans_x = None
        self.slider_trans_y = None
        self.slider_trans_z = None

    def run(self):
        self.fig = plt.figure(figsize=(self._figheight, self._figwidth))
        self.slider_axes = [plt.axes([0.1, 0.03 * (i + 1), 0.8, 0.03]) for i in range(9)]
        self.viewgraph = plt.axes([0.1, 0.3, 0.8, 0.7])
        self.viewgraph.axis('equal')
        self.viewgraph.invert_yaxis()
        self.viewgraph.set_ylim((0, 1200))
        self.viewgraph.set_xlim((150, 1200))

        self.viewgraph.scatter(self.image_points[:, 1], self.image_points[:, 0], color='red', s=0.1)
        self.viewgraph.scatter(self.pcd_transformed_2d[0][:, 0, 0], self.pcd_transformed_2d[0][:, 0, 1], color='blue',
                               s=0.1)

        self.slider_center_x = Slider(self.slider_axes[0], 'Center X', 0, 1000, valinit=600, valstep=0.1)
        self.slider_center_y = Slider(self.slider_axes[1], 'Center Y', 0, 1000, valinit=600, valstep=0.1)
        self.slider_focal_length = Slider(self.slider_axes[2], 'Focal Length', 0.1, 1800, valinit=1000, valstep=0.1)
        self.slider_rot_x = Slider(self.slider_axes[3], 'Roll', -np.pi, np.pi, valinit=0, valstep=0.01)
        self.slider_rot_y = Slider(self.slider_axes[4], 'Pitch', -np.pi, np.pi, valinit=0, valstep=0.01)
        self.slider_rot_z = Slider(self.slider_axes[5], 'Yaw', -np.pi, np.pi, valinit=0, valstep=0.01)
        self.slider_trans_x = Slider(self.slider_axes[6], 'X', -100, 100, valinit=0, valstep=0.1)
        self.slider_trans_y = Slider(self.slider_axes[7], 'Y', -100, 100, valinit=0, valstep=0.1)
        self.slider_trans_z = Slider(self.slider_axes[8], 'Z', -100, 100, valinit=-100, valstep=0.1)

        self.slider_center_x.on_changed(self.update)
        self.slider_center_y.on_changed(self.update)
        self.slider_focal_length.on_changed(self.update)
        self.slider_rot_x.on_changed(self.update)
        self.slider_rot_y.on_changed(self.update)
        self.slider_rot_z.on_changed(self.update)
        self.slider_trans_x.on_changed(self.update)
        self.slider_trans_y.on_changed(self.update)
        self.slider_trans_z.on_changed(self.update)
        plt.show()

    def update(self, *args):
        self.viewgraph.clear()
        rotmat_x = np.array([[1, 0, 0],
                             [0, np.cos(self.slider_rot_x.val), -np.sin(self.slider_rot_x.val)],
                             [0, np.sin(self.slider_rot_x.val), np.cos(self.slider_rot_y.val)]], dtype=np.float32)
        rotmat_y = np.array([[np.cos(self.slider_rot_y.val), 0, np.sin(self.slider_rot_y.val)],
                             [0, 1, 0],
                             [-np.sin(self.slider_rot_y.val), 0, np.cos(self.slider_rot_y.val)]], dtype=np.float32)
        rotmat_z = np.array([[np.cos(self.slider_rot_z.val), -np.sin(self.slider_rot_z.val), 0],
                             [np.sin(self.slider_rot_z.val), np.cos(self.slider_rot_z.val), 0],
                             [0, 0, 1]], dtype=np.float32)

        rotmat_zyx = np.linalg.multi_dot([rotmat_z, rotmat_y, rotmat_x])

        rvec = cv2.Rodrigues(rotmat_zyx)[0]
        self.tvec = np.array([self.slider_trans_x.val, self.slider_trans_y.val, self.slider_trans_z.val],
                             dtype=np.float32)
        self.cam_mat = np.array([[self.slider_focal_length.val, 0, self.slider_center_x.val],
                                 [0, self.slider_focal_length.val, self.slider_center_y.val],
                                 [0, 0, 1]], dtype=np.float32)

        self.pcd_transformed_2d = cv2.projectPoints(self.model_points, rvec, self.tvec, self.cam_mat,
                                                    self.distortion)
        self.viewgraph.scatter(self.image_points[:, 1], self.image_points[:, 0], color='red', s=0.1)
        self.viewgraph.scatter(self.pcd_transformed_2d[0][:, 0, 0], self.pcd_transformed_2d[0][:, 0, 1], color='blue',
                               s=0.1)


fitter = Fitter3DManual(model_pcd=model_pcd, image_pcd=image_pcd)
fitter.run()
