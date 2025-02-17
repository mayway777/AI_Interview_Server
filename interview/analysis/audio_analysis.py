from pydub import AudioSegment 
import librosa
import os
import soundfile as sf
import torch
import warnings
import noisereduce as nr
import whisper

# 전역 변수로 Whisper 모델 선언
global_whisper_model = None

def setup_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU 활성화: {torch.cuda.device_count()}개 사용 가능")
        return device
    else:
        print("GPU를 사용할 수 없습니다. CPU 사용.")
        return torch.device("cpu")

def load_whisper_model(model_name="large-v3-turbo"):
    global global_whisper_model
    if global_whisper_model is None:
        device = setup_gpu()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"Loading {model_name} model...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                global_whisper_model = whisper.load_model(model_name, device=device)
            
            global_whisper_model = global_whisper_model.float()
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Whisper 모델 로드 중 오류: {e}")
            global_whisper_model = None
    return global_whisper_model

def wav_to_text(wav_file):
    global global_whisper_model
    try:
        if global_whisper_model is None:
            global_whisper_model = load_whisper_model()
        if global_whisper_model is None:
            raise Exception("모델 로드 실패")

        print("음성을 텍스트로 변환 중...")
        result = global_whisper_model.transcribe(
            wav_file, 
            language="ko",
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        )
        text = result["text"]

        print("변환된 문장 text: ", text)
        return text

    except Exception as e:
        print(f"예상치 못한 오류 발생: {str(e)}")
        return ""

def analyze_speed(wav_file_path, average_speed=65):
    try:
        print(f"분석 시작: {wav_file_path}")
        
        result_text = wav_to_text(wav_file_path)
        if not result_text:
            print("텍스트 변환 실패")
            return None, None, None

        text_count = len(result_text.split())
        print(f"분석된 단어 수: {text_count}")

        audio = AudioSegment.from_wav(wav_file_path)
        audio_duration = audio.duration_seconds
        print(f"오디오 길이: {audio_duration:.2f}초")

        words_per_minute = (text_count / audio_duration) * 60 if audio_duration > 0 else 0
        print(f"분당 단어 수: {words_per_minute:.1f}")

        percentage_difference = ((words_per_minute - average_speed) / average_speed) * 100
        print(f"평균 대비 차이: {percentage_difference:.1f}%")

        words_per_minute_round = round(words_per_minute) if words_per_minute else 0
        
        return result_text, words_per_minute_round, percentage_difference

    except Exception as e:
        print(f"분석 중 오류 발생: {str(e)}")
        return None, None, None

def convert_to_wav_with_denoise(input_file_path):
    try:
        print(f"입력 파일 경로: {input_file_path}")
        if not os.path.exists(input_file_path):
            print(f"입력 파일이 존재하지 않습니다: {input_file_path}")
            return None

        print("오디오 추출 시작...")
        audio = AudioSegment.from_file(input_file_path)
        
        wav_file_path = input_file_path.rsplit('.', 1)[0] + '.wav'
        print(f"WAV 파일 경로: {wav_file_path}")
        
        print("WAV 파일 저장 중...")
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(wav_file_path, format="wav")
        print(f"WAV 파일 저장됨: {wav_file_path}")
        
        if not os.path.exists(wav_file_path):
            print("WAV 파일 생성 실패")
            return None
        
        print("잡음 제거를 위해 WAV 파일 로드 중...")
        wav, sr = librosa.load(wav_file_path, sr=None)
        
        print("잡음 제거 중...")
        denoised_wav = nr.reduce_noise(y=wav, sr=sr)
        
        print("잡음이 제거된 파일 저장 중...")
        sf.write(wav_file_path, denoised_wav, sr)
        
        if os.path.exists(wav_file_path):
            print(f"최종 파일 생성 완료: {wav_file_path}")
            return wav_file_path
        else:
            print("최종 파일 생성 실패")
            return None
        
    except Exception as e:
        print(f"변환 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
