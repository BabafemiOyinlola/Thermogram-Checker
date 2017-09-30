import cv2
from thermogramClassifier import ThermogramClassifier

classifier = ThermogramClassifier()
classifier.train_from_text_file()

sample_image = cv2.imread('assets/breast-thermograms/non-cancerous/capture1.png')
thermogram_is_cancerous = classifier.is_cancerous(sample_image)

print()

if thermogram_is_cancerous:
    print('THERMOGRAM IS CANCEROUS')
else:
    print('NO PROBLEM')

print('done')

