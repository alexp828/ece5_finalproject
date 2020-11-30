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
(4) https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column

"""

WEBCAM_OUTPUT_PATH = 'output.png'
DIGIT_DIM = 8
ROI_SIZE = (DIGIT_DIM, DIGIT_DIM)
THRESH_VALUE = 90
FRAME_RESOLUTION = (1280, 720)
BLUR_AMOUNT = 5
BOUNDING_BOX_ADJ = 5
BOUNDING_BOX_SIZE_THRESH = 2500

img_raw = None
img_gray = None
img_thresh = None

# Part 3: advanced image processing
"""
Given a frame (image), return the marked up frame
and an array of ROI (regions of interest)
in the form of a (n * 64) array, where n is the number of regions.
The array is n * 2, in the form of [[p1, x1], [p2, x2], ... , [pn, xn]]
"""
def get_predictions(frame):

    global img_raw
    global img_gray
    global img_thresh

    #resize frame to 720p, using inter-area (best for downscaling)
    frame = cv2.resize(frame, FRAME_RESOLUTION, cv2.INTER_AREA)
    img_raw = np.copy(frame)

    # B/W AND SMOOTH
    img_gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    img_gray = cv2.GaussianBlur(img_gray, (BLUR_AMOUNT, BLUR_AMOUNT), 0)

    # THRESHOLD HERE
    ret, img_thresh = cv2.threshold(img_gray, THRESH_VALUE, 255, cv2.THRESH_BINARY_INV)
    # thresh_2 = cv2.bitwise_not(img_thresh)

    # CONTOURS, BOUNDING BOXES
    contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]

    predictions = np.zeros((0))
    box_ctr = 0

    for rect in bounding_boxes:
        try:
            height = rect[3]
            width = rect[2]
            
            # check to ensure that small artifacts (eg noise, scratch marks) aren't counted as numbers
            if (height * width) < BOUNDING_BOX_SIZE_THRESH:
                continue

            box_ctr = box_ctr + 1
            dx = (height//2) - (width//2)
            # square bounding box
            width = height

            #bounding coordinates
            b1 = np.array([rect[0], rect[1]])
            b2 = np.array([rect[0] + width, rect[1] + height])
            b1[0] = b1[0] - dx
            b2[0] = b2[0] - dx

            #bring out b1 and b2 by a fixed amount
            adj = BOUNDING_BOX_ADJ
            b1 = b1 - adj
            b2 = b2 + adj

            ROI = img_thresh[int(b1[1]):int(b1[1]+height), int(b1[0]):int(b1[0]+width)]
            # check to make sure ROI is not empty
            if (ROI.size == 0):
                continue
            ROI_1 = cv2.resize(ROI, ROI_SIZE, interpolation=cv2.INTER_NEAREST)
            ROI_2 = cv2.resize(ROI, ROI_SIZE, interpolation=cv2.INTER_AREA)

            # TODO have a function for ROI_2, for now just return weighted sum
            ROI = 0.75*ROI_2 + 0.25*ROI_1
            ROI = np.reshape(ROI, DIGIT_DIM**2)

            r1 = (int(b1[0]), int(b1[1]))
            r2 = (int(b2[0]), int(b2[1]))

            # ensure returning an array of predictions along with x values
            # of bounding boxes, so the digits can be sorted
            prediction = kn.predict([ROI])[0]
            p = np.array([prediction, b1[0]])
            predictions = np.append(predictions, p)

            cv2.rectangle(frame, r1, r2, (0, 255, 0), 2)
            cv2.putText(frame, "Prediction: " + str(prediction), r1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        except:
            pass

        # if (None==frame):
        #     print("frame is NONE")
        # if (0==len(predictions)):
        #     print("predictions is NONE")

    ret = np.array([frame, predictions])
    return ret


# Part 3: Machine Learning
digits = datasets.load_digits()
kn = KNeighborsClassifier(n_neighbors=10)
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

X_train, X_test, y_train, y_target = train_test_split(data, digits.target, test_size=0.10, shuffle=False)
kn.fit(X_train, y_train)
print("SCORE: " + str(kn.score(X_test, y_target)))



#part 1? basic image processing
cap = cv2.VideoCapture(0)
frame = None
predictions = None
while(True):

    try:
        ret, f = cap.read()
        res = get_predictions(f)
        frame = res[0]
        predictions = res[1]

        cv2.imshow('webcam view', frame)
    except:
        pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite(WEBCAM_OUTPUT_PATH, frame)
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
# cv2.imshow('frame', frame)
predictions = np.reshape(predictions, (len(predictions)//2, 2))
predictions = predictions[predictions[:,-1].argsort()]

for i in range(len(predictions)):
    prediction = predictions[i][0]
    xval = predictions[i][1]
    # print("Prediction: " + str(prediction) + "\nx: " + str(xval) + "\n")

cv2.waitKey(0)

f = plt.figure(1)
# f.set_figwidth(15)
# f.set_figheight(15)
plt.subplot(2,2,1)
plt.title("img_raw")
plt.imshow(img_raw)
plt.subplot(2,2,2)
plt.title("img_gray")
plt.imshow(img_gray)
plt.subplot(2,2,3)
plt.title("img_thresh")
plt.imshow(img_thresh)
plt.subplot(2,2,4)
plt.title("frame")
plt.imshow(frame)
plt.show()

## PART 4: ARDUINO... GET NUMBER FROM predictions
