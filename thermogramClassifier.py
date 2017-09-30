from pattern_recognition.anomalyDetection import AnomalyDetection
from featureExtractor import FeatureExtractor


class ThermogramClassifier:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.detector_algorithm = AnomalyDetection(1)

        self.trained = False

        self.normal_thermogram_features_file_name = 'text-files/normal-features-train.txt'
        self.cancerous_thermogram_features_file_name = 'text-files/cancerous-features-train.txt'

    def train(self):
        normal_thermograms_feature_images = self.feature_extractor.generate_feature_images_from_normal_images()
        normal_features = []

        normal_features_file = open(self.normal_thermogram_features_file_name, 'w')

        for feature_image in normal_thermograms_feature_images:
            feature = ThermogramClassifier.get_feature_from_feature_image(feature_image)
            normal_features_file.write(str(feature) + '\n')
            normal_features.append([feature])

        normal_features_file.close()
        self.detector_algorithm.train([[item for sublist in normal_features for item in sublist]])

        cancerous_features = []

        cancerous_features_file = open(self.cancerous_thermogram_features_file_name, 'w')

        cancerous_thermogramps_feature_images = self.feature_extractor.generate_feature_images_from_cancerous_images()
        for feature_image in cancerous_thermogramps_feature_images:
            feature = ThermogramClassifier.get_feature_from_feature_image(feature_image)
            cancerous_features_file.write(str(feature) + '\n')
            cancerous_features.append([feature])

        cancerous_features_file.close()
        self.detector_algorithm.set_ephsilon(normal_features, cancerous_features)

        self.trained = True

    def train_from_text_file(self):
        file = open(self.normal_thermogram_features_file_name, 'r')

        normal_features = []

        for line in file:
            content = line.strip()
            normal_features.append([int(content)])

        file.close()
        if len(normal_features) == 0:
            raise Exception(
                'Failed to train using text file. call \'train()\' method to train and initialise text file')

        file = open(self.cancerous_thermogram_features_file_name, 'r')

        cancerous_features = []

        for line in file:
            content = line.strip()
            cancerous_features.append([int(content)])

        file.close()

        self.detector_algorithm.train([[item for sublist in normal_features for item in sublist]])
        self.detector_algorithm.set_ephsilon(normal_features, cancerous_features)
        self.trained = True

    def is_cancerous(self, image):
        if not self.trained:
            raise Exception('Thermogram classifier has not been trained.')

        feature_image = self.feature_extractor.generate_feature_image_from_image(image)
        feature = ThermogramClassifier.get_feature_from_feature_image(feature_image)
        return self.detector_algorithm.is_anomalous([feature])

    @staticmethod
    def get_feature_from_feature_image(feature_image):
        image_width = feature_image.shape[1]
        image_height = feature_image.shape[0]

        count = 0
        for index in range(0, image_width):
            for index2 in range(0, image_height):
                if feature_image[index2][index] == 255:
                    count += 1
        return int((count * 100) / (image_width * image_height))
