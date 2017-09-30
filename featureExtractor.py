import os, cv2

from region_detectors.armpitsAndNeckRegionDetector import ArmpitsAndNeckRegionDetector
from region_detectors.inframammaryFoldRegionDetector import InframammaryFoldRegionDetector
from region_detectors.regionOfInterestExtractor import RegionOfInterestExtractor
from utils.thermogramUtils import ThermogramUtils


class FeatureExtractor:
    @staticmethod
    def generate_feature_images_from_normal_images():
        return FeatureExtractor.generate_feature_images_from_images('assets/breast-thermograms/non-cancerous')

    @staticmethod
    def generate_feature_images_from_cancerous_images():
        return FeatureExtractor.generate_feature_images_from_images('assets/breast-thermograms/cancerous')

    @staticmethod
    def generate_feature_images_from_images(directory):
        features = []

        for (dir, dirs, files) in os.walk(directory):
            for filename in files:
                image = cv2.imread(directory + '/' + filename, cv2.IMREAD_COLOR)
                if image is not None:
                    feature_image = FeatureExtractor.generate_feature_image_from_image(image)
                    features.append(feature_image)
        return features

    @staticmethod
    def generate_feature_image_from_image(image):
        cv2.imshow('original image', image)
        roi = FeatureExtractor.extract_ROI(image)
        cv2.imshow('ROI', roi)
        roi_warm_region = ThermogramUtils.view_hot_regions(roi)
        cv2.imshow('warm region', roi_warm_region)
        roi_warm_region_masked = FeatureExtractor.mask_all_relevant_regions(roi_warm_region)
        cv2.imshow('feature image', roi_warm_region_masked)
        cv2.waitKey(0)
        return roi_warm_region_masked

    @staticmethod
    def extract_ROI(image):
        [roi, left_breast, right_breast] = RegionOfInterestExtractor.get_breast_region_thermogram(image)

        return roi

    @staticmethod
    def mask_all_relevant_regions(warm_roi_image):
        result = FeatureExtractor.mask_inframammary_fold_regions(warm_roi_image)
        cv2.imshow('mask fold warm region', result)
        result = FeatureExtractor.mask_left_armpit_regions(result)
        cv2.imshow('mask left armpit warm region', result)
        result = FeatureExtractor.mask_right_armpit_regions(result)
        cv2.imshow('mask right armpit region', result)
        result = FeatureExtractor.mask_neck_regions(result)
        return result

    @staticmethod
    def mask_inframammary_fold_regions(warm_region_roi):
        fold_region_extractor = InframammaryFoldRegionDetector()
        fold_region_extractor.train()

        width = warm_region_roi.shape[1]
        height = warm_region_roi.shape[0]

        for i in range(0, height):
            for j in range(0, width):
                if warm_region_roi[i, j] != 0:
                    height_ratio = int(i / height * 100)
                    width_ratio = int(j / width * 100)

                    is_in_fold_region = fold_region_extractor.is_in_fold_region(width_ratio, height_ratio)
                    if is_in_fold_region:
                        warm_region_roi[i, j] = 0

        return warm_region_roi

    @staticmethod
    def mask_left_armpit_regions(warm_region_roi):
        return FeatureExtractor.mask_armpit_and_neck_regions(warm_region_roi, 'assets/roi/warm-left-armpit-regions',
                                                             'text-files/left-armpit-fold-train.txt')

    @staticmethod
    def mask_right_armpit_regions(warm_region_roi):
        return FeatureExtractor.mask_armpit_and_neck_regions(warm_region_roi, 'assets/roi/warm-right-armpit-regions',
                                                             'text-files/right-armpit-fold-train.txt')

    @staticmethod
    def mask_neck_regions(warm_region_roi):
        return FeatureExtractor.mask_armpit_and_neck_regions(warm_region_roi, 'assets/roi/warm-neck-regions',
                                                             'text-files/neck-fold-train.txt')

    @staticmethod
    def mask_armpit_and_neck_regions(warm_region_roi, directory, file_name):
        armpit_region_extractor = ArmpitsAndNeckRegionDetector(directory, file_name)
        armpit_region_extractor.train()

        width = warm_region_roi.shape[1]
        height = warm_region_roi.shape[0]

        for i in range(0, height):
            for j in range(0, width):
                if warm_region_roi[i, j] != 0:
                    height_ratio = int(i / height * 100)
                    width_ratio = int(j / width * 100)

                    is_in_fold_region = armpit_region_extractor.is_in_armpit_region(width_ratio, height_ratio)
                    if is_in_fold_region:
                        warm_region_roi[i, j] = 0

        return warm_region_roi
