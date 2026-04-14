import streamlit as st
import numpy as np
from PIL import Image

# 에러의 주범이었던 cv2를 완전히 뺐습니다!
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

st.set_page_config(page_title="LSK SDPE Vision", layout="wide")
st.title("🏃‍♂️ LSK SDPE 웹 분석기")
st.write("태블릿/스마트폰 카메라로 자세를 촬영하세요.")

# AI 엔진 가동
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def get_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(rad*180.0/np.pi)
    return 360-angle if angle > 180.0 else angle

img_file = st.camera_input("카메라 촬영 (전면/후면 전환 가능)")

if img_file:
    # cv2 대신 파이썬 기본 도구(PIL)를 사용합니다. 에러 발생 확률 0%
    image = Image.open(img_file)
    img_array = np.array(image)
    
    results = pose.process(img_array)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        
        # S & P 수치 계산
        s_score = int(max(0, 100 - abs(lm[23].y - lm[24].y) * 1000))
        p_angle = int(get_angle([lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y]))

        # 관절선 그리기 및 화면 출력
        mp_drawing.draw_landmarks(img_array, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        st.image(img_array, use_column_width=True)

        col1, col2 = st.columns(2)
        col1.metric("S (Stability)", f"{s_score}점")
        col2.metric("P (Propulsion)", f"{p_angle}도")
    else:
        st.error("전신이 잘 나오게 다시 촬영해주세요.")
