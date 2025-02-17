import os
import uuid
import librosa
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from sklearn.preprocessing import MinMaxScaler
import speech_recognition as sr
import tensorflow as tf
from tensorflow import keras
import noisereduce as nr
from concurrent.futures import ThreadPoolExecutor

# custom optimizer 정의
class CustomOptimizer(tf.keras.optimizers.legacy.Adam):
    def __init__(self, *args, **kwargs):
        if 'weight_decay' in kwargs:
            del kwargs['weight_decay']
        super().__init__(*args, **kwargs)

# 모델 경로 설정
FILLER_DETERMINE_MODEL_PATH = r"C:\Users\User\Desktop\BIT40\Frontend\dataset\filler_determine_model_by_train2.h5"  
FILLER_CLASSIFIER_MODEL_PATH = r"C:\Users\User\Desktop\BIT40\Frontend\dataset\filler_classifier_updated.h5"

# custom_objects 설정
custom_objects = {
    'Adam': CustomOptimizer,
    'optimizer': CustomOptimizer
}

# 전역 변수
filler_determine_model = None
filler_classifier_model = None
scaler = MinMaxScaler()

# 상수 정의
FRAME_LENGTH = 0.025
FRAME_STRIDE = 0.0010
TARGET_DBFS = -20.0
MIN_FILLER_GAP = 800
MIN_FILLER_LENGTH = 250
SILENCE_THRESH = -36

def Silent_load_models():
    """모델 로드 (프로그램 시작시 한 번만 실행)"""
    global filler_determine_model, filler_classifier_model
    
    try:
        filler_determine_model = tf.keras.models.load_model(
            FILLER_DETERMINE_MODEL_PATH,
            custom_objects=custom_objects,
            compile=False
        )
        
        filler_classifier_model = tf.keras.models.load_model(
            FILLER_CLASSIFIER_MODEL_PATH,
            custom_objects=custom_objects,
            compile=False
        )
        
        optimizer = CustomOptimizer(learning_rate=0.001)
        filler_determine_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        filler_classifier_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
    except Exception as e:
        print(f"모델 로딩 오류: {str(e)}")
        raise

def match_target_amplitude(sound, target_dBFS):
    """오디오의 진폭을 조정"""
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def reduce_noise(wav):
    """잡음 제거"""
    return nr.reduce_noise(y=wav, sr=16000)

def predict_filler(audio_file):
    """추임새 여부 예측"""
    temp_file_name = f"temp_{uuid.uuid4()}.wav"
    try:
        # 오디오 길이 제한으로 처리 속도 향상
        audio_sample = audio_file[:3000] if len(audio_file) > 3000 else audio_file
        audio_sample.export(temp_file_name, format="wav")
        
        wav, sr = librosa.load(temp_file_name, sr=16000, duration=3.0)
        mfcc = librosa.feature.mfcc(y=wav)
        mfcc_scaled = scaler.fit_transform(mfcc.T).T
        padded_mfcc = pad2d(mfcc_scaled, 40)
        padded_mfcc = np.expand_dims(padded_mfcc, 0)
        result = filler_determine_model.predict(padded_mfcc, verbose=0)
        return 1 if result[0][0] >= 0.7 else 0
    finally:
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)

def predict_filler_type(audio_file):
    """추임새 유형 분류"""
    temp_file_name = f"temp_{uuid.uuid4()}.wav"
    try:
        audio_sample = audio_file[:3000] if len(audio_file) > 3000 else audio_file
        audio_sample.export(temp_file_name, format="wav")
        
        wav, sr = librosa.load(temp_file_name, sr=16000, duration=3.0)
        mfcc = librosa.feature.mfcc(y=wav, sr=16000, n_mfcc=100, n_fft=400, hop_length=160)
        mfcc_scaled = scaler.fit_transform(mfcc.T).T
        padded_mfcc = pad2d(mfcc_scaled, 40)
        padded_mfcc = np.expand_dims(padded_mfcc, 0)
        result = filler_classifier_model.predict(padded_mfcc, verbose=0)
        return np.argmax(result)
    finally:
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)

def process_speech_segment(args):
    """음성 구간 처리 - 병렬 처리용"""
    segment, source_file = args
    temp_file = f"temp_{uuid.uuid4()}.wav"
    
    try:
        segment.export(temp_file, format="wav")
        r = sr.Recognizer()
        with sr.AudioFile(temp_file) as source:
            audio = r.record(source)
            return r.recognize_google(audio_data=audio, language="ko-KR")
    except:
        return None
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def create_json(audio_file):
    """오디오 파일 분석하여 JSON 생성"""
    intervals_jsons = []
    
    intervals = detect_nonsilent(
        audio_file,
        min_silence_len=250,
        silence_thresh=SILENCE_THRESH,
        seek_step=1
    )
    
    if not intervals:
        return []
        
    if intervals[0][0] != 0:
        intervals_jsons.append({'start':0, 'end':intervals[0][0], 'tag':'0000'})
    
    non_silence_start = intervals[0][0]
    before_silence_start = intervals[0][1]
    last_filler_end = 0
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for interval in intervals:
            interval_audio = audio_file[interval[0]:interval[1]]
            interval_length = interval[1] - interval[0]
            
            if interval_length >= MIN_FILLER_LENGTH:
                futures.append((interval, executor.submit(predict_filler, interval_audio)))
            
        for interval, future in futures:
            interval_audio = audio_file[interval[0]:interval[1]]
            is_filler = future.result()
            
            if (interval[0] - before_silence_start) >= MIN_FILLER_GAP:
                intervals_jsons.append({
                    'start': non_silence_start,
                    'end': before_silence_start + 200,
                    'tag': '1000'
                })
                non_silence_start = interval[0] - 200
                intervals_jsons.append({
                    'start': before_silence_start,
                    'end': interval[0],
                    'tag': '0000'
                })
            
            if (interval[0] - last_filler_end) >= MIN_FILLER_GAP and is_filler == 0:
                intervals_jsons.append({
                    'start': non_silence_start,
                    'end': interval[0],
                    'tag': '1000'
                })
                intervals_jsons.append({
                    'start': interval[0],
                    'end': interval[1],
                    'tag': '1111'
                })
                last_filler_end = interval[1]
                non_silence_start = interval[1]
            
            before_silence_start = interval[1]
    
    if non_silence_start != len(audio_file):
        intervals_jsons.append({
            'start': non_silence_start,
            'end': len(audio_file),
            'tag': '1000'
        })
    
    return intervals_jsons

def STT_with_json(audio_file, jsons):
    """음성 인식 및 분석 결과 생성"""
    first_silence = 0
    first_silence_interval = 0
    num = 0
    unrecognizable_start = 0
    transcript_json = []
    statistics_filler_json = []
    statistics_silence_json = []
    filler_1 = filler_2 = filler_3 = 0
    audio_total_length = audio_file.duration_seconds
    silence_interval = 0
    
    # 음성 구간 수집
    speech_segments = []
    
    for json_data in jsons:
        if json_data['tag'] == '0000':
            if num == 0:
                first_silence += (json_data['end']-json_data['start'])/1000
            else:
                silence_interval += (json_data['end']-json_data['start'])/1000
                transcript_json.append({
                    'start': json_data['start'],
                    'end': json_data['end'],
                    'tag': '0000',
                    'result': f"({round((json_data['end']-json_data['start'])/1000)}초).."
                })
                
        elif json_data['tag'] == '1111':
            if num == 0:
                silence = f"({round(first_silence)}초).."
                transcript_json.append({
                    'start': 0,
                    'end': json_data['start'],
                    'tag': '0000',
                    'result': silence
                })
                first_silence_interval = first_silence
            
            segment = audio_file[json_data['start']:json_data['end']]
            filler_type = predict_filler_type(segment)
            
            if (json_data['end'] - json_data['start']) >= MIN_FILLER_LENGTH:
                if filler_type == 0:
                    transcript_json.append({
                        'start': json_data['start'],
                        'end': json_data['end'],
                        'tag': '1001',
                        'result': '어(추임새)'
                    })
                    filler_1 += 1
                elif filler_type == 1:
                    transcript_json.append({
                        'start': json_data['start'],
                        'end': json_data['end'],
                        'tag': '1010',
                        'result': '음(추임새)'
                    })
                    filler_2 += 1
                else:
                    transcript_json.append({
                        'start': json_data['start'],
                        'end': json_data['end'],
                        'tag': '1100',
                        'result': '그(추임새)'
                    })
                    filler_3 += 1
                num += 1
                
        elif json_data['tag'] == '1000':
            segment = audio_file[
                unrecognizable_start if unrecognizable_start != 0 else json_data['start']:
                json_data['end']
            ]
            speech_segments.append((segment, json_data))
    
    # 병렬로 음성 인식 처리
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_speech_segment, (segment, data)) 
                  for segment, data in speech_segments]
        
        for (_, json_data), future in zip(speech_segments, futures):
            text = future.result()
            if text:
                if num == 0:
                    silence = f"({round(first_silence)}초).."
                    transcript_json.append({
                        'start': 0,
                        'end': json_data['start'],
                        'tag': '0000',
                        'result': silence
                    })
                    first_silence_interval = first_silence
                
                transcript_json.append({
                    'start': unrecognizable_start if unrecognizable_start != 0 else json_data['start'],
                    'end': json_data['end'],
                    'tag': '1000',
                    'result': text
                })
                unrecognizable_start = 0
                num += 1
            else:
                if unrecognizable_start == 0:
                    unrecognizable_start = json_data['start']
    
    statistics_filler_json.append({'어':filler_1, '음':filler_2, '그':filler_3})
    statistics_silence_json.append({
        '통역개시지연시간': 100 * first_silence_interval/audio_total_length,
        '침묵시간': 100 * silence_interval/audio_total_length,
        '발화시간': 100 * (audio_total_length - first_silence - silence_interval)/audio_total_length
    })
    
    return transcript_json, statistics_filler_json, statistics_silence_json

def make_transcript(audio_file_path):
    """전체 전사 프로세스 실행"""
    audio = AudioSegment.from_wav(audio_file_path)
    normalized_audio = match_target_amplitude(audio, TARGET_DBFS)
    intervals_jsons = create_json(normalized_audio)
    return STT_with_json(normalized_audio, intervals_jsons)

def return_transcript(wav_file_dir):
    """분석 결과 반환"""
    transcript_json, statistics_filler_json, statistics_silence_json = make_transcript(wav_file_dir)
    filler_count = count_filler_tags(transcript_json)
    silence_count = count_silence_count(transcript_json)
    return filler_count, silence_count

def analyze_audio(wav_file_path):
    """오디오 파일 분석"""
    try:
        return return_transcript(wav_file_path)
    except Exception as e:
        print(f"오디오 분석 중 오류 발생: {e}")
        return 0, 0
    finally:
        if os.path.exists(wav_file_path):
            os.remove(wav_file_path)

def count_filler_tags(transcript_json):
    """추임새 개수 계산"""
    return sum(1 for item in transcript_json if item['tag'] == '1001')

def count_silence_count(transcript_json):
    """의미있는 침묵 개수 계산"""
    return sum(
        1 for item in transcript_json
        if item['tag'] == '0000' and (item['end'] - item['start']) / 1000 >= 2
    )

# 패딩 함수
pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))
