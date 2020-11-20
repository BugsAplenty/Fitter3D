#!/usr/bin/env python3

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.widgets import Slider, Button
from scipy.spatial.transform import Rotation
from SoftPOSIT import softposit

matplotlib.use('TkAgg')


class Fitter3DManual:
    def __init__(self, model_pcd, image_pcd):
        # Point clouds extracted from processing the image and the STL mesh of the object.
        self.model_points = np.asarray(model_pcd.points)
        self.image_points = np.asarray(image_pcd.points)[:, 0:2]
        self.model_points = self.model_points - np.mean(self.model_points, axis=0)
        self.image_points = self.image_points - np.mean(self.image_points, axis=0)

        # Image camera parameters.
        self._cam_mat_init = np.array([[1800, 0, 0], [0, 1800, 0], [0, 0, 1]], dtype=np.float32)  # Ideally, the focal
        # length should never change, but since the image we use is a digital screencap with unknown camera
        # parameters, we basically have to guess the focal length.
        self.cam_mat = self._cam_mat_init
        self.distortion = np.array([0, 0, 0, 0, 0], dtype=np.float32)

        # Initial Transform and pose of the model relative to the image axis.
        self._roll_init, self._pitch_init, self._yaw_init = np.random.uniform(low=-np.pi, high=np.pi, size=(3,))
        self._rmat_init = Rotation.from_euler('zyx', [self._yaw_init, self._pitch_init, self._roll_init]).as_matrix()
        self._rvec_init = cv2.Rodrigues(self._rmat_init)[0].squeeze()
        self._tvec_init = np.array([0, 0, 100], dtype=np.float32)

        # Slider maximum and minimum values:
        self._center_x_min = -1000
        self._center_x_max = 1000
        self._center_y_min = -1000
        self._center_y_max = 1000
        self._trans_min = -500
        self._trans_max = 500
        self._rot_min = -np.pi
        self._rot_max = np.pi
        self._focal_length_min = 10 ** -20
        self._focal_length_max = 2500

        self.rmat = self._rmat_init
        self.rvec = self._rvec_init
        self.tvec = self._tvec_init

        self.pcd_transformed_2d = cv2.projectPoints(self.model_points, self._rvec_init, self._tvec_init,
                                                    self._cam_mat_init,
                                                    self.distortion)

        self.pcd_transformed_2d = self.pcd_transformed_2d[0].squeeze()

        # Plot parameters.
        self._figheight = 10
        self._figwidth = 10
        self.fig = plt.figure(figsize=(self._figheight, self._figwidth))
        self.slider_axes = [plt.axes([0.1, 0.03 * (i + 1), 0.8, 0.03]) for i in range(9)]
        # self.button_axis = plt.axes([0.94, 0.01, 0.05, 0.05])
        self.viewgraph = plt.axes([0.1, 0.3, 0.8, 0.7])
        self.viewgraph.set_ylim((0, 1200))
        self.viewgraph.set_xlim((150, 1200))
        self.viewgraph.axis('equal')

        # Add the pointclouds to the plot.
        self.viewgraph.scatter(self.image_points[:, 1], self.image_points[:, 0], color='red', s=0.1)
        self.viewgraph.scatter(self.pcd_transformed_2d[:, 0], self.pcd_transformed_2d[:, 1], color='blue',
                               s=0.1)

        # Add sliders to the plot.
        self.slider_center_x = Slider(self.slider_axes[0], 'Center X', self._center_x_min, self._center_x_max,
                                      valinit=self._cam_mat_init[0, 2], valstep=0.1)
        self.slider_center_y = Slider(self.slider_axes[1], 'Center Y', self._center_y_min, self._center_y_max,
                                      valinit=self._cam_mat_init[1, 2], valstep=0.1)
        self.slider_focal_length = Slider(self.slider_axes[2], 'Focal Length', self._focal_length_min,
                                          self._focal_length_max, valinit=self._cam_mat_init[0, 0], valstep=0.1)
        self.slider_rot_x = Slider(self.slider_axes[3], 'Roll', self._rot_min, self._rot_max,
                                   valinit=self._roll_init, valstep=0.1)
        self.slider_rot_y = Slider(self.slider_axes[4], 'Pitch', self._rot_min, self._rot_max,
                                   valinit=self._pitch_init, valstep=0.1)
        self.slider_rot_z = Slider(self.slider_axes[5], 'Yaw', self._rot_min, self._rot_max, valinit=self._yaw_init,
                                   valstep=0.1)
        self.slider_trans_x = Slider(self.slider_axes[6], 'X', self._trans_min, self._trans_max,
                                     valinit=self._tvec_init[0], valstep=0.1)
        self.slider_trans_y = Slider(self.slider_axes[7], 'Y', self._trans_min, self._trans_max,
                                     valinit=self._tvec_init[1], valstep=0.1)
        self.slider_trans_z = Slider(self.slider_axes[8], 'Z', self._trans_min, self._trans_max,
                                     valinit=self._tvec_init[2], valstep=0.1)
        # self.button_softposit = Button(self.button_axis, 'SoftPOSIT!')



    def run(self):
        self.slider_center_x.on_changed(self.update_manual)
        self.slider_center_y.on_changed(self.update_manual)
        self.slider_focal_length.on_changed(self.update_manual)
        self.slider_rot_x.on_changed(self.update_manual)
        self.slider_rot_y.on_changed(self.update_manual)
        self.slider_rot_z.on_changed(self.update_manual)
        self.slider_trans_x.on_changed(self.update_manual)
        self.slider_trans_y.on_changed(self.update_manual)
        self.slider_trans_z.on_changed(self.update_manual)
        # self.button_softposit.on_clicked(self.softposit())
        plt.show()

    def update_manual(self, *args):
        self.viewgraph.clear()
        self.rmat = Rotation.from_euler('zyx', [self.slider_rot_z.val, self.slider_rot_y.val,
                                                self.slider_rot_x.val]).as_matrix()
        self.rvec = cv2.Rodrigues(self.rmat)[0]
        self.tvec = np.array([self.slider_trans_x.val, self.slider_trans_y.val, self.slider_trans_z.val],
                             dtype=np.float32)
        self.cam_mat = np.array([[self.slider_focal_length.val, 0, self.slider_center_x.val],
                                 [0, self.slider_focal_length.val, self.slider_center_y.val],
                                 [0, 0, 1]], dtype=np.float32)

        self.pcd_transformed_2d = cv2.projectPoints(self.model_points, self.rvec, self.tvec, self.cam_mat,
                                                    self.distortion)
        self.pcd_transformed_2d = self.pcd_transformed_2d[0].squeeze()  # Gets rid of the useless attributes of the
        # above method.
        self.viewgraph.scatter(self.image_points[:, 1], self.image_points[:, 0], color='red', s=0.1)
        self.viewgraph.scatter(self.pcd_transformed_2d[:, 0], self.pcd_transformed_2d[:, 1], color='blue',
                               s=0.1)
        plt.draw()


model_pcd = o3d.io.read_point_cloud("chair_points.pcd")
image_pcd = o3d.io.read_point_cloud("chair_flat_points.pcd")
fitter = Fitter3DManual(model_pcd=model_pcd, image_pcd=image_pcd)
fitter.run()

