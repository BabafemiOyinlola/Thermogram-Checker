import math


class AnomalyDetection:
    def __init__(self, number_of_features):
        self.features_count = number_of_features
        print('Number of features expected for training: ' + str(number_of_features))

        self.trained = False
        self.mean_set = [0 for i in range(0, number_of_features)]
        self.variance_set = [0 for i in range(0, number_of_features)]

        self.ephsilon_set = False
        self.ephsilon = 0

    def train(self, features):
        if len(features) != self.features_count:
            raise Exception('Inconsistent number of features')

        for i in range(0, self.features_count):
            mean = 0
            for value in features[i]:
                mean += value

            mean = mean / len(features[i])
            self.mean_set[i] = mean

            variance = 0
            for value in features[i]:
                variance += ((value - mean) ** 2)

            variance = variance / len(features[i])
            self.variance_set[i] = variance

        self.trained = True

    def find_probability(self, new_feature_set):
        if len(new_feature_set) != self.features_count:
            raise Exception('Number of features in new example is inconsistent with the training data')

        if not self.trained:
            raise Exception('Algorithm has not been trained.')

        probability = 1
        for i in range(0, self.features_count):
            mean = self.mean_set[i]
            variance = self.variance_set[i]

            a = math.exp(-((new_feature_set[i] - mean) ** 2) / (2 * variance))
            b = (2 * math.pi * variance) ** 0.5

            prob = a / b
            probability *= prob

        return probability

    def set_ephsilon(self, normal_data, anomalous_data):
        min_normal_prob = 1
        max_anomalous_prob = 0

        for example in normal_data:
            probability = self.find_probability(example)
            if probability < min_normal_prob:
                min_normal_prob = probability

        for example in anomalous_data:
            probability = self.find_probability(example)
            if probability > max_anomalous_prob:
                max_anomalous_prob = probability

        if max_anomalous_prob > min_normal_prob:
            print('uncertain value set for ephsilon')

        eph = (max_anomalous_prob + min_normal_prob) / 2
        self.ephsilon = eph
        self.ephsilon_set = True

    def is_anomalous(self, test_data):
        if not self.ephsilon_set:
            raise Exception('Ephsilon value has not been set.')

        probability = self.find_probability(test_data)

        if probability >= self.ephsilon:
            return False

        return True
