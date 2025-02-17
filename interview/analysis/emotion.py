from keras.models import load_model
import numpy as np
import dlib
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf

emotion_offsets = (20, 40)
emotions = {
    0: {"emotion": "Angry", "color": (193, 69, 42)},
    1: {"emotion": "Disgust", "color": (164, 175, 49)},
    2: {"emotion": "Fear", "color": (40, 52, 155)},
    3: {"emotion": "Happy", "color": (23, 164, 28)},
    4: {"emotion": "Sad", "color": (164, 93, 23)},
    5: {"emotion": "Surprise", "color": (218, 229, 97)},
    6: {"emotion": "Neutral", "color": (108, 72, 200)},
}

# 전역 변수로 모델과 관련 변수들 선언
global_detector = None
global_predictor = None
global_emotion_classifier = None
global_emotion_target_size = None

# 스레드풀 생성
executor = ThreadPoolExecutor(max_workers=2)

def load_emotion_models():
    global global_detector, global_predictor, global_emotion_classifier, global_emotion_target_size
    
    if global_detector is not None:
        return
        
    # GPU 메모리 동적 할당 설정
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU 설정 오류: {e}")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    face_landmarks = os.path.join(current_dir, "dataset", "shape_predictor_68_face_landmarks.dat")
    emotion_model_path = os.path.join(current_dir, "dataset", "emotion_model.hdf5")
    
    global_detector = dlib.get_frontal_face_detector()
    global_predictor = dlib.shape_predictor(face_landmarks)
    global_emotion_classifier = load_model(emotion_model_path, compile=False)
    global_emotion_target_size = global_emotion_classifier.input_shape[1:3]

def preprocess_face(gray_face):
    """얼굴 이미지 전처리"""
    try:
        gray_face = cv2.resize(gray_face, (global_emotion_target_size))
        gray_face = gray_face.astype('float32') / 255.0
        gray_face = (gray_face - 0.5) * 2.0
        gray_face = np.expand_dims(np.expand_dims(gray_face, 0), -1)
        return gray_face
    except:
        return None

def process_face(rect, gray_frame, frame):
    """단일 얼굴 처리"""
    shape = global_predictor(gray_frame, rect)
    
    # 얼굴 기울기 분석
    eye_left_y = shape.part(36).y
    eye_right_y = shape.part(45).y
    
    # 기울기 상태 확인
    if abs(eye_left_y - eye_right_y) > 5:
        if eye_left_y > eye_right_y:
            lean_state = "left"
        else:
            lean_state = "right"
    else:
        lean_state = "center"
    
    # 얼굴 영역 추출 및 전처리
    x, y, w, h = rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top()
    gray_face = gray_frame[y:y + h, x:x + w]
    processed_face = preprocess_face(gray_face)
    
    if processed_face is not None:
        # 감정 예측
        emotion_prediction = global_emotion_classifier.predict(processed_face, verbose=0)
        emotion_probability = np.max(emotion_prediction)
        
        if emotion_probability > 0.36:
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_label = emotions[emotion_label_arg]['emotion']
            color = emotions[emotion_label_arg]['color']
            
            return {
                'rect': (x, y, w, h),
                'emotion': emotion_label,
                'color': color,
                'lean': lean_state,
                'probability': emotion_probability
            }
    
    return None

def emotion_recognition(frame):
    if global_detector is None:
        load_emotion_models()
    
    # 프레임 크기 조정 (성능 향상을 위해)
    scale = 0.5
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 검출
    rects = global_detector(gray_frame, 0)
    
    # 스레드풀을 사용한 병렬 처리
    face_results = list(executor.map(
        lambda rect: process_face(
            dlib.rectangle(
                int(rect.left()/scale), 
                int(rect.top()/scale),
                int(rect.right()/scale), 
                int(rect.bottom()/scale)
            ),
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            frame
        ),
        rects
    ))
    
    # 결과 처리 및 시각화
    for result in face_results:
        if result:
            x, y, w, h = result['rect']
            color = result['color']
            emotion_label = result['emotion']
            lean_state = result['lean']
            
            # 통계 업데이트
            emotion_count[emotion_label] += 1
            emotion_count['total'] += 1
            lean_count[lean_state] += 1
            lean_count['total'] += 1
            
            # 시각화
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.line(frame, (x, y + h), (x + 20, y + h + 20), color, thickness=2)
            cv2.rectangle(frame, (x + 20, y + h + 20), (x + 110, y + h + 40), color, -1)
            cv2.putText(frame, emotion_label, (x + 25, y + h + 36),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # 기울기 메시지
            alert_message = f"Face is {lean_state}"
            cv2.putText(frame, alert_message, (20, frame.shape[0] - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    return frame, emotion_count, lean_count

# 나머지 함수들은 동일하게 유지
emotion_count = {
    "Angry": 0, "Disgust": 0, "Fear": 0,
    "Happy": 0, "Sad": 0, "Surprise": 0,
    "Neutral": 0, "total": 0
}

lean_count = {
    "left": 0, "center": 0, "right": 0, "total": 0
}

def get_emotion_percentages():
    emotion_percentages = {}
    for emotion_label in emotions.values():
        emotion = emotion_label['emotion']
        if emotion_count['total'] > 0:
            emotion_percentages[emotion] = round(emotion_count[emotion] / emotion_count['total'] * 100, 2)
        else:
            emotion_percentages[emotion] = 0.0
    return emotion_percentages

def get_lean_percentages():
    lean_percentages = {}
    if lean_count["total"] > 0:
        for lean in ['center', 'right', 'left']:
            lean_percentages[lean] = round(lean_count[lean] / lean_count["total"] * 100, 2)
    else:
        lean_percentages = {"center": 0.0, "right": 0.0, "left": 0.0}
    return lean_percentages