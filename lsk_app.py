import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="LSK SDPE Vision", layout="wide")
st.title("🏃‍♂️ LSK SDPE 웹 분석기")
st.write("태블릿/스마트폰 카메라로 자세를 촬영하세요.")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def get_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(rad*180.0/np.pi)
    return 360-angle if angle > 180.0 else angle

img_file = st.camera_input("카메라 촬영 (전면/후면 전환 가능)")

if img_file:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        s_score = int(max(0, 100 - abs(lm[23].y - lm[24].y) * 1000))
        p_angle = int(get_angle([lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y]))

        # 관절선 그리기
        mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

        col1, col2 = st.columns(2)
        col1.metric("S (Stability)", f"{s_score}점")
        col2.metric("P (Propulsion)", f"{p_angle}도")
    else:
        st.error("전신이 잘 나오게 다시 촬영해주세요.")
