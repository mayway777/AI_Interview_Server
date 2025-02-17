# 🚀 프로젝트 설정 가이드

## 1. 🐍 Python 환경 설정
이 프로젝트는 **Python 3.10** 버전에서 실행됩니다. 원활한 실행을 위해 아래의 단계를 따라주세요.

### 1.1 🛠 Python 버전 확인
```sh
python --version
```
✅ Python 3.10이 설치되어 있지 않다면 [공식 웹사이트](https://www.python.org/downloads/)에서 다운로드 후 설치하세요.

## 2. 📦 패키지 설치
### 2.1 🔽 `requirements.txt` 다운로드
깃허브에서 `requirements.txt` 파일을 다운로드합니다.

```sh
git clone https://github.com/mayway777/AI_Interview_Server.git
cd AI_Interview_Server
```

### 2.2 🏗 가상환경 설정 (선택 사항)
가상환경을 설정하여 패키지를 격리할 수 있습니다.

```sh
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate    # Windows
```

### 2.3 ⚡ 패키지 설치
프로젝트에서 필요한 패키지를 설치합니다.

```sh
pip install -r requirements.txt
```
(추가설치필요)
```sh
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## 3. ▶ 프로젝트 실행
설정이 완료되면 프로젝트를 실행할 수 있습니다.

```sh
python main.py
```

## 4. 🤖 AI 면접 분석 서버
이 서버는 **AI 면접 영상을 분석**하기 위한 실시간 처리 시스템으로, **FastAPI** 기반으로 구현되었습니다. 주요 기능은 다음과 같습니다:

- 🎥 **영상 분석 모듈**
  - **시선 추적 분석 (Eye.py)**: MediaPipe Face Mesh 기술을 사용하여 시선을 추적하고 분석합니다.
  - **감정 상태 분석 (emotion.py)**: CNN 모델을 사용하여 7가지 기본 감정을 분류합니다.

- 🎙 **음성 분석 모듈**
  - **음성-텍스트 변환 (audio_analysis.py)**: Whisper 모델을 사용하여 음성을 텍스트로 변환하고 말하기 속도를 분석합니다.
  - **침묵/추임새 분석 (Silent_FillerWords.py)**: 딥러닝 모델로 음성에서 추임새와 침묵 구간을 검출합니다.
  - **음성 변동성 분석 (Volatility.py)**: 음성 피치 변동성을 분석하여 음성의 안정성을 평가합니다.

### 📂 Dataset
- `emotion_model.hdf5`: 7가지 감정을 분류하는 CNN 모델
- `filler_classifier.h5`: 추임새 유형을 분석하는 모델
- `filler_determine_model_by_train2.h5`: 추임새 여부를 분류하는 모델
- `shape_predictor_68_face_landmarks.dat`: dlib의 얼굴 랜드마크 데이터

이 데이터셋들은 **실시간 AI 면접 분석**을 위한 핵심 모델들로 구성됩니다.

## 5. 🔗 웹 연동
이 서버는 **Next.js 기반 EMPAI 웹**의 AI 면접 분석 기능과 연동됩니다. 면접자의 **영상/음성/답변 분석 결과**를 클라이언트에 실시간으로 제공하며, MongoDB에 저장된 데이터를 활용하여 피드백을 생성합니다.

🔗 EMPAI 웹사이트: [EMPAI GitHub 저장소](https://github.com/mayway777/AI_Interview_Server.git)

---
🎉 **설치 및 설정이 완료되었습니다! AI 면접 분석을 시작해보세요!** 🚀
