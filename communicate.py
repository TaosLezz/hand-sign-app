import cv2
from HandTracking import HandDetector
from Classification import Classifier
import numpy as np
import traceback
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import time
import tracemalloc
tracemalloc.start()

img_size = 400
classifier = Classifier("best_model_1106.h5",
                        "labels_13chars.txt")
white = np.ones((img_size, img_size), np.uint8) * 255
cv2.imwrite("white.jpg", white)

detector = HandDetector(maxHands=1)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)
"""
labels for keras_model_7chars.h5
"""
# labels = ['den truong', 'di', 'duoc', 'giao tiep', 'hoc', 'muon', 'toi']
# labels = ['đến trường', 'đi', 'được', 'giao tiếp', 'học', 'muốn', 'tôi']
"""
labels for keras_model_13chars.h5
"""
# labels = ['chung toi', 'cong bang', 'den truong', 'di','doi xu','duoc','giao tiep',
#           'hoa nhap','hoc','moi nguoi', 'muon', 'toi','xung quanh']
labels = ['chúng tôi', 'công bằng','đến trường', 'đi', 'đối xử','được', 'giao tiếp',
          'hòa nhập với','học','mọi người', 'muốn', 'tôi', 'xung quanh']
# Streamlit app title
st.title("Hand Sign Detection")
class VideoProcessor:
    def __init__(self):
        self.offset = 26
        self.img_size = 400
        self.current_label = ""

    def recv(self, frame):
        global current_label
        frm = frame.to_ndarray(format="bgr24")
        try:
            frm = cv2.flip(frm, 1)
            hands, frm = hd.findHands(frm, draw=False, flipType=True)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                image = frm[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]
                white = cv2.imread("white.jpg")
                handz, image = hd2.findHands(image, draw=True, flipType=True)
                if handz:
                    hand = handz[0]
                    pts = hand['lmList']

                    os = ((self.img_size - w) // 2) - 15
                    os1 = ((self.img_size - h) // 2) - 15
                    for t in range(0, 4, 1):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(5, 8, 1):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(9, 12, 1):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(13, 16, 1):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(17, 20, 1):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[5][0] + os, pts[5][1] + os1), (pts[9][0] + os, pts[9][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[9][0] + os, pts[9][1] + os1), (pts[13][0] + os, pts[13][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[13][0] + os, pts[13][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[5][0] + os, pts[5][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)

                    for i in range(21):
                        cv2.circle(white, (pts[i][0] + os, pts[i][1] + os1), 2, (0, 0, 255), 1)

                    prediction, index = classifier.getPrediction(white, draw=False)

                    if prediction[index] > 0.9:
                        self.current_label = labels[index]
                    else:
                        self.current_label = "không nhận ra"
                    # print("trong ham: ",  self.current_label)
                    # frm = cv2.putText(frm, self.current_label, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

            return av.VideoFrame.from_ndarray(frm, format='bgr24')
        except Exception as e:
            st.error(f"An error occurred: {traceback.format_exc()}")
            return av.VideoFrame.from_ndarray(frm, format='bgr24')

def fun_label(video_processor):
    return video_processor.current_label


col1, col2 = st.columns([2, 1])

# Left column for the video stream
with col1:
    webrtc_ctx =webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
                    rtc_configuration=RTCConfiguration(
                        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                    ))


with col2:
    result_container = st.empty()


while webrtc_ctx.video_processor:
    current_label = fun_label(webrtc_ctx.video_processor)
    print("ngoai ham:", current_label)

    # Cập nhật kết quả
    # result_container.markdown(f"<div style='font-size:30px;'>Predict: {current_label}</div>", unsafe_allow_html=True)
    result_container.markdown("<div style='font-size:30px;'>Predict:</div>"
                              f"<div style='font-size:25px;'>{current_label}</div>", unsafe_allow_html=True)

    time.sleep(1)
huongdan_button = st.button("Trợ giúp")
if huongdan_button:
    st.query_params["page"] = "huongdan_button"

# Điều hướng dựa trên tham số truy vấn
query_params = st.query_params
if "page" in query_params:
    page = query_params["page"]
    if page == "huongdan_button":
        with open("huong_dan_dung.py", encoding="utf-8") as f:
            exec(f.read())
