import cv2
import mediapipe as mp
import numpy as np
import os
import logging


# 경고 메시지 억제
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# MediaPipe Face Mesh 초기화
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 전역 변수
gaze_count = {
    "center": 0,
    "right": 0,
    "left": 0,
    "total": 0
}

# 눈 랜드마크 인덱스
LEFT_EYE_LANDMARKS = [33, 133, 157, 158, 159, 160, 161, 173, 246]  # 왼쪽 눈 윤곽점
RIGHT_EYE_LANDMARKS = [362, 263, 387, 386, 385, 384, 398, 466]    # 오른쪽 눈 윤곽점
LEFT_IRIS = [469, 470, 471, 472]   # 왼쪽 홍채점
RIGHT_IRIS = [474, 475, 476, 477]  # 오른쪽 홍채점

def reset_gaze_count():
    """시선 추적 카운터 초기화"""
    global gaze_count
    gaze_count = {
        "center": 0,
        "right": 0,
        "left": 0,
        "total": 0
    }

def calculate_gaze_direction(landmarks, frame_width):
    """정밀한 시선 방향 계산"""
    try:
        # 양쪽 눈의 외곽 경계점 찾기
        left_eye_outline = np.array([(landmarks.landmark[33].x, landmarks.landmark[133].x)])
        right_eye_outline = np.array([(landmarks.landmark[362].x, landmarks.landmark[263].x)])
        
        # 홍채 중심점 계산
        left_iris_center = np.mean([landmarks.landmark[idx].x for idx in LEFT_IRIS])
        right_iris_center = np.mean([landmarks.landmark[idx].x for idx in RIGHT_IRIS])
        
        # 각 눈에서 홍채의 상대적 위치 계산
        left_ratio = (left_iris_center - left_eye_outline[0][0]) / (left_eye_outline[0][1] - left_eye_outline[0][0])
        right_ratio = (right_iris_center - right_eye_outline[0][0]) / (right_eye_outline[0][1] - right_eye_outline[0][0])
        
        # 양쪽 눈의 평균 비율 계산
        avg_ratio = (left_ratio + right_ratio) / 2
        
        # 움직임의 정도 계산
        movement = avg_ratio - 0.5
        
        # 보정된 비율 반환
        return avg_ratio, abs(movement)
        
    except Exception as e:
        print(f"시선 방향 계산 오류: {e}")
        return 0.5, 0

def track_eyes(frame):
    """향상된 시선 추적 함수"""
    global gaze_count
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    if not results.multi_face_landmarks:
        return frame
    
    landmarks = results.multi_face_landmarks[0]
    h, w = frame.shape[:2]
    
    # 시선 방향 계산
    gaze_ratio, movement_strength = calculate_gaze_direction(landmarks, w)
    gaze_count["total"] += 1
    
    # 시선 방향 판단 (개선된 임계값)
    threshold = 0.04  # 미세한 움직임은 중앙으로 판단
    if movement_strength < threshold:
        gaze_count["center"] += 1
        direction = "Center"
        color = (0, 255, 0)
    else:
        if gaze_ratio < 0.5:
            gaze_count["left"] += 1
            direction = "Left"
            color = (0, 0, 255)
        else:
            gaze_count["right"] += 1
            direction = "Right"
            color = (255, 0, 0)
    
    # 결과 시각화
    cv2.putText(frame, f"Looking {direction}", (50, 40), 
               cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
    
    # 시선 추적 시각화
    for idx in LEFT_IRIS + RIGHT_IRIS:
        pos = (int(landmarks.landmark[idx].x * w), int(landmarks.landmark[idx].y * h))
        cv2.circle(frame, pos, 1, color, -1)
    
    # 눈 윤곽 시각화
    for idx in LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS:
        pos = (int(landmarks.landmark[idx].x * w), int(landmarks.landmark[idx].y * h))
        cv2.circle(frame, pos, 1, (255, 255, 255), -1)
    
    # 디버그 정보
    cv2.putText(frame, f"Direction Ratio: {gaze_ratio:.3f}", (50, 70),
               cv2.FONT_HERSHEY_DUPLEX, 0.6, (147, 58, 31), 1)
    cv2.putText(frame, f"Movement: {movement_strength:.3f}", (50, 90),
               cv2.FONT_HERSHEY_DUPLEX, 0.6, (147, 58, 31), 1)
    
    return frame

def get_current_gaze_stats():
    """현재 시선 추적 통계 반환"""
    global gaze_count
    
    if gaze_count["total"] == 0:
        return {
            "counts": dict(gaze_count),
            "percentages": {
                "center": 0.0,
                "right": 0.0,
                "left": 0.0
            }
        }
    
    percentages = {
        "center": round(gaze_count["center"] / gaze_count["total"] * 100, 2),
        "right": round(gaze_count["right"] / gaze_count["total"] * 100, 2),
        "left": round(gaze_count["left"] / gaze_count["total"] * 100, 2)
    }
    
    return {
        "counts": dict(gaze_count),
        "percentages": percentages
    }