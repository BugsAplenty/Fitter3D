#!/usr/bin/env python3

# script for tuning parameters
import cv2
import open3d as o3d
import numpy as np


class FeatureExtractor:
    def __init__(self, target_image_path):
        self.img = cv2.imread(target_image_path, 0)
        self.edges = None

    def slider_callback(self, foo):
        pass

    def save_points(self, *args):
        points = np.where(self.edges == 255)
        cv2.destroyAllWindows()

    def run(self):
        # Create the control panel.
        cv2.namedWindow('Control Panel')
        cv2.createTrackbar('threshold1', 'Control Panel', 0, 255,
                           self.slider_callback)  # change the maximum to whatever you like
        cv2.createTrackbar('threshold2', 'Control Panel', 0, 255,
                           self.slider_callback)  # change the maximum to whatever you like
        cv2.createTrackbar('apertureSize', 'Control Panel', 0, 2, self.slider_callback)
        cv2.createTrackbar('L1/L2', 'Control Panel', 0, 1, self.slider_callback)
        cv2.createButton('Save Points', self.save_points)

        while True:
            # get threshold value from trackbar
            th1 = cv2.getTrackbarPos('threshold1', 'Control Panel')
            th2 = cv2.getTrackbarPos('threshold2', 'Control Panel')

            # aperture size can only be 3,5, or 7
            apSize = cv2.getTrackbarPos('apertureSize', 'Control Panel') * 2 + 3

            # true or false for the norm flag
            norm_flag = cv2.getTrackbarPos('L1/L2', 'Control Panel') == 1

            # print out the values
            print('')
            print('threshold1: {}'.format(th1))
            print('threshold2: {}'.format(th2))
            print('apertureSize: {}'.format(apSize))
            print('L2gradient: {}'.format(norm_flag))

            img_resized = cv2.resize(self.img, (int(self.img.shape[1] / 2), int(self.img.shape[0] / 2)))
            self.edges = cv2.Canny(img_resized, th1, th2, apertureSize=apSize, L2gradient=norm_flag)
            cv2.imshow('Image', self.edges)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


target_image_path = "./chair_real.jpg"
feature_extractor = FeatureExtractor(target_image_path)
feature_extractor.run()
