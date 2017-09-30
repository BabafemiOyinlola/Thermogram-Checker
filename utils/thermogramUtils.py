import cv2, numpy as np
import os


class ThermogramUtils:
    @staticmethod
    def detect_edges(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_gray_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

        edges = cv2.Canny(blurred_gray_image, 120, 370)

        return edges

    @staticmethod
    def view_hot_regions(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_range = np.array([0, 50, 50])
        upper_range = np.array([25, 255, 255])

        mask = cv2.inRange(hsv, lower_range, upper_range)

        return mask
