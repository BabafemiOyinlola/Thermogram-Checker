import cv2
import os
from pattern_recognition.anomalyDetection import AnomalyDetection


class InframammaryFoldRegionDetector:
    def __init__(self):
        self.file_name = 'text-files/mammary-fold-train.txt'

        self.widths = []
        self.heights = []

        self.trained = False

        self.detector_algorithm = AnomalyDetection(2)
        self.ephsilon = 0.0001

    def train(self):
        print('getting text file data from training images')
        # self.initialise_text_file()
        self.use_text_file_data()

        print('training anomaly detection algorithm')
        self.train_anomaly_detector_algorithm()

    def initialise_text_file(self):
        images = InframammaryFoldRegionDetector.get_training_images()
        file = open(self.file_name, 'w')

        dict = {}

        for image in images:
            width = image.shape[1]
            height = image.shape[0]

            for i in range(0, height):
                for j in range(0, width):
                    if image[i, j] == 255:
                        height_ratio = int(i / height * 100)
                        width_ratio = int(j / width * 100)

                        info = str(width_ratio) + ", " + str(height_ratio)

                        value = dict.get(info, None)
                        if value is not None:
                            value += 1
                            dict[info] = value
                        else:
                            dict[info] = 1

        for key in dict.keys():
            file.write(str(key) + ', ' + str(dict[key]) + '\n')

        file.close()

    def use_text_file_data(self):
        file = open(self.file_name, 'r')

        widths = []
        heights = []

        for line in file:
            content = line.strip()
            width, height, value = content.split(", ")

            width = int(width)
            height = int(height)
            value = int(value)

            #if height < 70 and (width > 60 or width < 40):
            #    continue
            for j in range(0, value):
                widths.append(width)
                heights.append(height)

        file.close()
        self.widths = widths
        self.heights = heights

    def train_anomaly_detector_algorithm(self):
        print('training anomaly detection algorithm')
        self.detector_algorithm.train([self.widths, self.heights])
        self.trained = True

    def is_in_fold_region(self, width_ratio, height_ratio):
        if not self.trained:
            raise Exception('Train fold region detection first')
        probability = self.detector_algorithm.find_probability([width_ratio, height_ratio])

        return probability >= self.ephsilon

    @staticmethod
    def get_training_images():
        images = []
        directory = 'roi/warm-inframammary-fold-regions'
        for (dir, dirs, files) in os.walk(directory):
            for filename in files:
                image = cv2.imread(directory + '/' + filename, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if image is not None:
                    images.append(image)

        return images
