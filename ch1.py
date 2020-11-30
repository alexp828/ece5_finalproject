import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm

"""
References

(1) https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
(2) https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
(3) https://debuggercafe.com/image-classification-with-mnist-dataset/

"""

WEBCAM_OUTPUT_PATH = 'output.png'
DIGIT_DIM = 8
ROI_SIZE = (DIGIT_DIM, DIGIT_DIM)
THRESH_VALUE = 90

digits = datasets.load_digits()
kn = KNeighborsClassifier(n_neighbors=10)
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# kn = svm.SVC(gamma=0.0001)

X_train, X_test, y_train, y_target = train_test_split(data, digits.target, test_size=0.10, shuffle=False)
kn.fit(X_train, y_train)
print("SCORE: " + str(kn.score(X_test, y_target)))

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame2 = np.copy(frame)
    frame = cv2.resize(frame, None, None, fx=0.5, fy=0.5)

    img_gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    # img_gray = cv2.bilateralFilter(img_gray,9,75,75)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # img_gray = cv2.GaussianBlur(img_gray, (15, 15), 0)

    ret, img_thresh = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY_INV)
    thresh_2 = cv2.bitwise_not(img_thresh)
    #img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = [cv2.boundingRect(c) for c in contours]

    for rect in bounding_boxes:
        try:
            height = rect[3]
            width = rect[2]
            # check to ensure that small artifacts (eg noise, scratch marks) aren't counted as numbers
            if (height * width) < 2500:
                continue
            dx = (height//2) - (width//2)
            # square
            width = height

            #bounding coordinates
            b1 = np.array([rect[0], rect[1]])
            b2 = np.array([rect[0] + width, rect[1] + height])
            b1[0] = b1[0] - dx
            b2[0] = b2[0] - dx

            #bring out b1 and b2 by a fixed amount
            adj = 5
            b1 = b1 - adj
            b2 = b2 + adj

            ROI = img_thresh[int(b1[1]):int(b1[1]+height), int(b1[0]):int(b1[0]+width)]
            # check to make sure ROI is not empty
            if (ROI.size == 0):
                continue
            ROI_1 = cv2.resize(ROI, ROI_SIZE, interpolation=cv2.INTER_NEAREST)
            ROI_2 = cv2.resize(ROI, ROI_SIZE, interpolation=cv2.INTER_AREA)
            # TODO have a function for ROI_2
            ROI = 0.75*ROI_2 + 0.25*ROI_1
            ROI = np.reshape(ROI, DIGIT_DIM**2)

            r1 = (int(b1[0]), int(b1[1]))
            r2 = (int(b2[0]), int(b2[1]))

            prediction = kn.predict([ROI])

            cv2.rectangle(frame, r1, r2, (0, 255, 0), 2)
            cv2.putText(frame, "Prediction: " + str(prediction[0]), r1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        except:
            pass

    cv2.imshow('webcam view', frame)
    #cv2.imshow('thresh view', img_thresh)
    # cv2.imshow('gray view', img_gray)
    # cv2.imshow('thresh view', img_thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite(WEBCAM_OUTPUT_PATH, frame2)
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# apply transformations/thresh to gray
frame = cv2.imread(WEBCAM_OUTPUT_PATH)
img_gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

ret, img_thresh = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bounding_boxes = [cv2.boundingRect(c) for c in contours]

regions = np.zeros((0))

plt.figure(1)
for rect in bounding_boxes:

    height = rect[3]
    width = rect[2]
    # check to ensure that small artifacts (eg noise, scratch marks) aren't counted as numbers
    if (height * width) < 2500:
        continue
    dx = (height//2) - (width//2)
    # square
    width = height

    #bounding coordinates
    b1 = np.array([rect[0], rect[1]])
    b2 = np.array([rect[0] + width, rect[1] + height])
    b1[0] = b1[0] - dx
    b2[0] = b2[0] - dx

    #bring out b1 and b2 by a fixed amount
    adj = 5
    b1 = b1 - adj
    b2 = b2 + adj

    ROI = img_thresh[int(b1[1]):int(b1[1]+height), int(b1[0]):int(b1[0]+width)]
    # check to make sure ROI is not empty
    if (ROI.size == 0):
        continue
    ROI_1 = cv2.resize(ROI, ROI_SIZE, interpolation=cv2.INTER_NEAREST)
    ROI_2 = cv2.resize(ROI, ROI_SIZE, interpolation=cv2.INTER_AREA)
    # TODO have a function for ROI_2
    ROI = 0.75*ROI_2 + 0.25*ROI_1
    ROI = np.reshape(ROI, DIGIT_DIM**2)

    regions = np.append(regions, ROI)

    r1 = (int(b1[0]), int(b1[1]))
    r2 = (int(b2[0]), int(b2[1]))

    cv2.rectangle(frame, r1, r2, (0, 255, 0), 2)
    cv2.putText(frame, "Prediction: " + str(prediction[0]), r1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

regions = np.reshape(regions, (len(regions)//(DIGIT_DIM**2), DIGIT_DIM**2))
print(regions)
print("Shape of regions: " + str(np.shape(regions)))
prediction = kn.predict(regions)

f = plt.figure(1)
f.set_figwidth(15)

for i in range(np.shape(regions)[0]):
    ROI = regions[i]
    ROI = np.reshape(ROI, ROI_SIZE)
    plt.subplot(1, np.shape(regions)[0], i+1)
    plt.title("Prediction: " + str(prediction[i]))
    plt.imshow(ROI)

plt.show()



# run classification on regions
