import cv2
import numpy as np

from utils.thermogramUtils import ThermogramUtils


class RegionOfInterestExtractor:

    @staticmethod
    def get_breast_region_thermogram(thermogram):
        edges = ThermogramUtils.detect_edges(thermogram)
        cv2.imshow('edges', edges)

        lower_boundary = RegionOfInterestExtractor.get_breast_lower_boundary(edges)

        breast_height = RegionOfInterestExtractor.get_breast_height(thermogram.shape[0], lower_boundary)

        upper_boundary = lower_boundary - breast_height

        segmented_thermogram = thermogram[upper_boundary:lower_boundary, 0:thermogram.shape[1] - 1]
        segmented_thermogram_gray = cv2.cvtColor(segmented_thermogram, cv2.COLOR_BGR2GRAY)

        ret, segment_threshold = cv2.threshold(segmented_thermogram_gray, 127, 255, cv2.THRESH_BINARY)

        left_boundary = RegionOfInterestExtractor.get_breast_left_boundary(segment_threshold)
        right_boundary = RegionOfInterestExtractor.get_breast_right_boundary(segment_threshold)

        central_axis = int((left_boundary + right_boundary) / 2)

        roi = thermogram[upper_boundary:lower_boundary, left_boundary:right_boundary]
        left_breast = thermogram[upper_boundary:lower_boundary, left_boundary:central_axis]
        right_breast = thermogram[upper_boundary:lower_boundary, central_axis:right_boundary]

        return [roi, left_breast, right_breast]

    @staticmethod
    def get_breast_upper_boundary(edge, breast_height_row):
        hpp_array = np.zeros((breast_height_row - 0))

        for index in range(0, breast_height_row):
            count = len([value for value in edge[index] if value == 255])
            hpp_array[index] = count

        for index in range(1, hpp_array.size):
            i = hpp_array.size - (index + 1)

            current_value = hpp_array[i]
            prev_value = hpp_array[i + 1]

            if current_value > prev_value:
                checker = 0
                k = [hpp_array[i - 1], hpp_array[i - 2], hpp_array[i - 3], hpp_array[i - 4], hpp_array[i - 5]]
                for item in k:
                    if item >= current_value:
                        checker += 1

                if checker > 2:
                    return i + 1
        return -1

    @staticmethod
    def get_breast_lower_boundary(edge):
        image_height = edge.shape[0]

        hpp_array = np.zeros((image_height))

        for index in range(0, image_height):
            count = len([value for value in edge[index] if value == 255])
            hpp_array[index] = count

        for index in range(1, int(hpp_array.size / 2)):
            i = hpp_array.size - (index + 1)

            current_value = hpp_array[i]
            prev_value = hpp_array[i + 1]

            if current_value > prev_value:
                checker = 0
                k = [hpp_array[i - 1], hpp_array[i - 2], hpp_array[i - 3], hpp_array[i - 4], hpp_array[i - 5]]
                for item in k:
                    if item >= current_value:
                        checker += 1

                if checker > 2:
                    return i + 1
        return -1

    @staticmethod
    def get_breast_left_boundary(image):
        image_width = image.shape[1]
        image_height = image.shape[0]

        vpp_array = np.zeros(image_width)

        for index in range(0, image_width):
            count = 0
            for index2 in range(0, image_height):
                if image[index2][index] == 255:
                    count += 1

            vpp_array[index] = count

        for index in range(1, int(vpp_array.size / 2)):

            current_value = vpp_array[index]
            prev_value = vpp_array[index - 1]

            if current_value > prev_value:
                checker = 0
                k = [vpp_array[index + 1], vpp_array[index + 2]]
                for item in k:
                    if item >= current_value:
                        checker += 1

                if checker == 2:
                    return index
        return -1

    @staticmethod
    def get_breast_right_boundary(image):
        image_width = image.shape[1]
        image_height = image.shape[0]

        vpp_array = np.zeros(image_width)

        for index in range(0, image_width):
            count = 0
            for index2 in range(0, image_height):
                if image[index2][index] == 255:
                    count += 1

            vpp_array[index] = count

        for index in range(1, int(vpp_array.size / 2)):
            i = vpp_array.size - (index + 1)

            current_value = vpp_array[i]
            prev_value = vpp_array[i + 1]

            if current_value > prev_value:
                checker = 0
                k = [vpp_array[i - 1], vpp_array[i - 2]]
                for item in k:
                    if item >= current_value:
                        checker += 1

                if checker == 2:
                    return i
        return -1

    @staticmethod
    def get_breast_height(image_height, lower_boundary):
        adjustment = int(0.05 * image_height)

        if (image_height - lower_boundary) < 100:
            return int(2 * image_height / 3) - adjustment
        else:
            return int(0.5 * image_height) - adjustment
