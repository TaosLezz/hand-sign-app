import math
import cv2
from HandTracking import HandDetector
from Classification import Classifier
import numpy as np
# from keras.models import load_model
import traceback

# import tensorflow as tf
# print(tf.__version__)

img_size = 400
# model = load_model('C:\\Users\\hieu\\PycharmProjects\\handsign\\hand_new\\keras_model.h5')
classifier = Classifier("C:\\Users\\Lenovo\\PycharmProjects\\pythonProject\\streamlit_app\\models\\best_model_1106.h5",
                        "C:\\Users\\Lenovo\\PycharmProjects\\pythonProject\\streamlit_app\\models\\labels_7chars.txt")
white = np.ones((img_size, img_size), np.uint8) * 255
cv2.imwrite("white.jpg", white)

capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

offset = 26
step = 1
flag = False
suv = 0

# labels = ['den truong', 'di', 'duoc', 'giao tiep', 'hoc', 'muon', 'toi']
labels = ['chung toi', 'cong bang', 'den truong', 'di','doi xu','duoc','giao tiep',
          'hoa nhap','hoc','moi nguoi', 'muon', 'toi','xung quanh']
bfh = 0
dicttt=dict()
count=0
kok=[]

while True:
    try:
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)
        hands, frame = hd.findHands(frame, draw=False, flipType=True)
        print(frame.shape)
        if hands:
            # #print(" --------- lmlist=",hands[1])
            hand = hands[0]
            x, y, w, h = hand['bbox']
            image = frame[y - offset:y + h + offset, x - offset:x + w + offset]
            white = cv2.imread("C:\\Users\\Lenovo\\PycharmProjects\\hand_sign_vn\\white.jpg")
            # img_final=img_final1=img_final2=0
            handz, image = hd2.findHands(image, draw=True, flipType=True)
            if handz:
                hand = handz[0]
                pts = hand['lmList']
                # x1,y1,w1,h1=hand['bbox']

                os = ((img_size - w) // 2) - 15
                os1 = ((img_size - h) // 2) - 15
                for t in range(0, 4, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 0), 3)
                for t in range(5, 8, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 0), 3)
                for t in range(9, 12, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 0), 3)
                for t in range(13, 16, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 0), 3)
                for t in range(17, 20, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 0), 3)
                cv2.line(white, (pts[5][0] + os, pts[5][1] + os1), (pts[9][0] + os, pts[9][1] + os1), (0, 255, 0),
                         3)
                cv2.line(white, (pts[9][0] + os, pts[9][1] + os1), (pts[13][0] + os, pts[13][1] + os1), (0, 255, 0),
                         3)
                cv2.line(white, (pts[13][0] + os, pts[13][1] + os1), (pts[17][0] + os, pts[17][1] + os1),
                         (0, 255, 0), 3)
                cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[5][0] + os, pts[5][1] + os1), (0, 255, 0),
                         3)
                cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0),
                         3)

                for i in range(21):
                    cv2.circle(white, (pts[i][0] + os, pts[i][1] + os1), 2, (0, 0, 255), 1)

                # cv2.imshow("2", white)
                # cv2.imshow("5", skeleton5)
                print(white.shape)
                # print(model.predict(white))
                # white = white.reshape(1, img_size, img_size, 3)
                # prob = np.array(model.predict(white)[0], dtype='float32')
                prediction, index = classifier.getPrediction(white, draw=False)
                # ch1 = np.argmax(prob, axis=0)
                # print(model.predict(white))

                if prediction[index] > 0.9:  # Chỉ hiển thị nhãn nếu độ chính xác lớn hơn 0.8
                    label = labels[index]
                else:
                    label = "ko nhan ra"

                frame = cv2.putText(frame, label, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

                # frame = cv2.putText(frame, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)



        cv2.imshow("frame", frame)
        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:
            # esc key
            break


    except Exception:
        print("==", traceback.format_exc())




capture.release()
cv2.destroyAllWindows()
