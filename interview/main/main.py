import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from werkzeug.utils import secure_filename
import cv2
from pathlib import Path
from typing import Optional, Generator,Tuple
import json
from io import BytesIO
from PIL import Image
from urllib.parse import unquote
import numpy as np
import asyncio
from contextlib import asynccontextmanager
from filelock import FileLock 
from pymongo import MongoClient
from bson import ObjectId
import os
import logging

# 분석 모듈 Import
from analysis.DB import video_db
from analysis.Eye import track_eyes, reset_gaze_count, get_current_gaze_stats
from analysis.emotion import emotion_recognition, get_emotion_percentages, get_lean_percentages, load_emotion_models
from analysis.audio_analysis import convert_to_wav_with_denoise, analyze_speed, load_whisper_model
from analysis.Silent_FillerWords import analyze_audio, Silent_load_models
from analysis.Volatility import calculate_pitch_variability
from analysis.feedback import InterviewEvaluator

global_executor = ThreadPoolExecutor(max_workers=8) 

# 상수 정의
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_LOCK_FILE = "model_load.lock"
CHUNK_SIZE = 1024 * 1024     # 1MB 청크 크기  

@asynccontextmanager
async def lifespan(app: FastAPI):
    """모델 로드 및 리소스 관리를 위한 개선된 라이프사이클 핸들러"""
    # 파일 잠금으로 동시 로드 방지
    with FileLock(MODEL_LOCK_FILE + ".lock"):
        if not os.path.exists(MODEL_LOCK_FILE):
            print("🔄 모델 로드 시작...")
            load_whisper_model()
            load_emotion_models()
            Silent_load_models()
            Path(MODEL_LOCK_FILE).touch()  # 완료 표시
            print("✅ 모델 로드 완료")
    
    yield
    
    # 서버 종료 시 잠금 파일 정리
    if os.path.exists(MODEL_LOCK_FILE):
        os.remove(MODEL_LOCK_FILE)

app = FastAPI(lifespan=lifespan)

logging.basicConfig(level=logging.INFO)

def is_model_loaded():
    """감정 인식 모델이 로드되었는지 확인"""
    try:
        return track_eyes is not None and emotion_recognition is not None
    except:
        return False

def initialize_analysis_document(userUid: str, resumeUid: str, resume_title: str, job_code: str, timestamp: str, company: Optional[str] = None):
    """초기 분석 문서 생성"""
    initial_doc = {
        "uid": userUid,
        "self_id": resumeUid,
        "title": resume_title,
        "job_code": job_code,
        "time": timestamp,
        "company": company,
        userUid: {}
    }
    return video_db.create_initial_document(initial_doc)

def save_uploaded_file(file: UploadFile, uid: str, filename: str) -> Tuple[str, str]:
    """파일을 저장하고 파일 경로와 안전한 파일명 반환"""
    user_folder = os.path.join(UPLOAD_FOLDER, uid)
    os.makedirs(user_folder, exist_ok=True)
    
    secure_name = secure_filename(filename)
    file_path = os.path.join(user_folder, secure_name)
    
    with open(file_path, "wb") as buffer:
        file.file.seek(0)
        shutil.copyfileobj(file.file, buffer)
    
    return file_path, secure_name

def analyze_audio_features(wav_file_path: str, job_code: str, current_question: str, resume_data: list):
    """오디오 특징 분석"""
    try:
        # 음성 변동성 분석
        voice_variability = calculate_pitch_variability(wav_file_path)
        
        # 말하기 속도 분석
        result_text, words_per_minute, speed_difference = analyze_speed(wav_file_path)
        
        # 추임새, 침묵 분석
        filler_count, silence_count = analyze_audio(wav_file_path)

        # GPT 평가 수행
        try:
            load_dotenv('.env.local')
            OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
            evaluator = InterviewEvaluator(OPENAI_API_KEY)
            
            gpt_evaluation = evaluator.evaluate_answer(
                job_code=job_code,
                resume_data=resume_data,
                question=current_question,
                answer=result_text
            )
        except Exception as gpt_error:
            print(f"GPT 평가 실패: {str(gpt_error)}")
            gpt_evaluation = {
                "error": "GPT 평가 중 오류가 발생했습니다.",
                "details": str(gpt_error)
            }

        return {
            "답변": result_text,
            "말하기속도": words_per_minute,
            "평속대비차이": speed_difference,
            "추임새갯수": filler_count,
            "침묵갯수": silence_count,
            "목소리변동성": voice_variability,
            "GPT_평가": gpt_evaluation
        }
    except Exception as e:
        print(f"오디오 분석 실패: {str(e)}")
        return {
            "답변": None,
            "말하기속도": None,
            "평속대비차이": None,
            "추임새갯수": None,
            "침묵갯수": None,
            "목소리변동성": None,
            "GPT_평가": None
        }

def analyze_video_features(file_path: str):
    """영상 특징 분석"""
    # 시선 추적 카운터 초기화
    reset_gaze_count()
    
    try:
        cap = cv2.VideoCapture(file_path)
        frame_number = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            if frame_number % 6 == 0:
                emotion_recognition(frame)
                track_eyes(frame)

        cap.release()

        # 최종 시선 통계 가져오기
        final_gaze_stats = get_current_gaze_stats()

        return {
            "감정_%": get_emotion_percentages(),
            "머리기울기_%": get_lean_percentages(),
            "아이트래킹_%": final_gaze_stats['percentages']
        }
    except Exception as e:
        print(f"영상 분석 실패: {str(e)}")
        return {
            "감정_%": None,
            "머리기울기_%": None,
            "아이트래킹_%": None
        }

def process_analysis(
    userUid, resumeUid, job_code, resume_title, 
    timestamp, company, questions, files, data
):
    """분석 처리 함수"""
    # 모델 로딩 확인
    if not is_model_loaded():
        load_whisper_model()
        load_emotion_models()
        Silent_load_models()

    try:
        # 초기 문서 생성
        doc_id = initialize_analysis_document(userUid, resumeUid, resume_title, job_code, timestamp, company)
        if not doc_id:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "초기 문서 생성 실패"
                }
            )

        # 공통 정보 생성
        combined_title = f"{resume_title}_{timestamp}"

        # 파일 저장 및 분석
        analyzed_count = 0
        for i, (question, file) in enumerate(zip(questions, files), 1):
            try:
                # 빈 파일이나 질문 건너뛰기
                if not file or not question:
                    continue
                
                original_extension = file.filename.split('.')[-1]
                filename = f"{combined_title}_video{i}.{original_extension}"
                
                # 동일한 파일명으로 저장
                file_path, secure_name = save_uploaded_file(file, userUid, filename)
                resume_data = json.loads(data)
                
                # WAV 파일로 변환
                wav_file_path = convert_to_wav_with_denoise(file_path)
                
                with ThreadPoolExecutor(max_workers=2) as executor:
                    # 병렬로 오디오와 비디오 분석
                    audio_future = executor.submit(
                        analyze_audio_features, 
                        wav_file_path, 
                        job_code, 
                        question, 
                        resume_data
                    )
                    
                    video_future = executor.submit(
                        analyze_video_features, 
                        file_path
                    )
                    
                    # 결과 대기
                    audio_analysis = audio_future.result()
                    video_analysis = video_future.result()

                # 분석 결과 통합 (파일 객체 제외)
                analysis_result = {
                    "video_number": i,
                    "video_filename": secure_name,
                    "question": question,
                    "답변": audio_analysis.get("답변"),
                    "말하기속도": audio_analysis.get("말하기속도"),
                    "평속대비차이": audio_analysis.get("평속대비차이"),
                    "추임새갯수": audio_analysis.get("추임새갯수"),
                    "침묵갯수": audio_analysis.get("침묵갯수"),
                    "목소리변동성": audio_analysis.get("목소리변동성"),
                    "감정_%": video_analysis.get("감정_%"),
                    "머리기울기_%": video_analysis.get("머리기울기_%"),
                    "아이트래킹_%": video_analysis.get("아이트래킹_%"),
                    "GPT_평가": audio_analysis.get("GPT_평가")
                }
                
                # DB에 분석 결과 업데이트
                success = video_db.update_analysis_results(doc_id, userUid, i, analysis_result)
                if success:
                    analyzed_count += 1
                    print(f"비디오 {i} 분석 결과 저장 성공")

                # 임시 WAV 파일 삭제
                if os.path.exists(wav_file_path):
                    os.remove(wav_file_path)

            except Exception as e:
                print(f"영상 {i} 분석 중 오류: {str(e)}")
                
                # 오류 시 기본 결과 저장
                error_result = {
                    "video_number": i,
                    "video_filename": filename if 'filename' in locals() else f"error_video{i}",
                    "question": question,
                    "답변": None,
                    "말하기속도": None,
                    "평속대비차이": None,
                    "추임새갯수": None,
                    "침묵갯수": None,
                    "목소리변동성": None,
                    "감정_%": None,
                    "머리기울기_%": None,
                    "아이트래킹_%": None,
                    "GPT_평가": None
                }
                video_db.update_analysis_results(doc_id, userUid, i, error_result)

        return JSONResponse(
            content={
                "success": True,
                "message": f"분석이 완료되었습니다. 총 {analyzed_count}개의 영상이 분석되었습니다.",
                "doc_id": str(doc_id),
                "analyzed_count": analyzed_count,
                "total_videos": len([q for q in questions if q])
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"요청 처리 중 오류 발생: {str(e)}"
            }
        )

@app.post("/analyze")
async def analyze_video(
    userUid: str = Form(...),
    resumeUid: str = Form(...),
    job_code: str = Form(...),
    resume_title: str = Form(...),
    timestamp: str = Form(...),
    company: str = Form(None),
    question_0: str = Form(None),
    question_1: str = Form(None),
    question_2: str = Form(None),
    question_3: str = Form(None),
    videoFile_0: UploadFile = File(None),
    videoFile_1: UploadFile = File(None),
    videoFile_2: UploadFile = File(None),
    videoFile_3: UploadFile = File(None),
    data: str = Form(...),
):
    # 질문과 파일 쌍 검증
    questions = [question_0, question_1, question_2, question_3]
    files = [videoFile_0, videoFile_1, videoFile_2, videoFile_3]
    
    print(questions)
    print(files)
    # 질문이 있는데 파일이 없거나, 파일이 있는데 질문이 없는 경우 체크
    valid_pairs = [(q, f) for q, f in zip(questions, files) if q and f]
    
    if len(valid_pairs) < 1:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "최소 1개 이상의 질문과 비디오 파일이 필요합니다."
            }
        )

    # 최소 1개 이상의 질문-파일 쌍이 있는지 확인
    valid_pairs = [(q, f) for q, f in zip(questions, files) if q and f]
    if not valid_pairs:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "최소 1개 이상의 질문과 비디오 파일이 필요합니다."
            }
        )

    # 글로벌 스레드 풀에서 작업 실행
    future = global_executor.submit(
        process_analysis, 
        userUid, resumeUid, job_code, resume_title, 
        timestamp, company, questions, files, data
    )
    
    # 결과 대기
    result = await asyncio.wrap_future(future)
    return result

@app.get('/health')
async def health_check():
    return {'status': 'ok'}

@app.post("/video/preview")
async def get_video_preview(request: Request):
   body = await request.json()
   uid = body.get('uid')
   filename = body.get('filename')
   
   if not uid or not filename:
       raise HTTPException(status_code=400, detail="Missing uid or filename")
   
   decoded_filename = unquote(filename)
   
   video_path = os.path.join(UPLOAD_FOLDER, uid, decoded_filename)
   
   if not os.path.exists(video_path):
       raise HTTPException(status_code=404, detail="Video not found")
   
   try:
       # 비디오의 첫 프레임 추출
       cap = cv2.VideoCapture(video_path)
       ret, frame = cap.read()
       cap.release()
       
       if not ret:
           raise HTTPException(status_code=500, detail="Could not read video frame")
       
       # BGR에서 RGB로 변환
       frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       image = Image.fromarray(frame_rgb)
       
       # 이미지를 바이트로 변환
       img_byte_arr = BytesIO()
       image.save(img_byte_arr, format='JPEG', quality=85)
       img_byte_arr.seek(0)
       
       return Response(
           content=img_byte_arr.getvalue(),
           media_type="image/jpeg"
       )
       
   except Exception as e:
       raise HTTPException(status_code=500, detail=str(e))

@app.post("/video")
async def stream_video(request: Request):
   body = await request.json()
   uid = body.get('uid')
   filename = body.get('filename')
   
   if not uid or not filename:
       raise HTTPException(status_code=400, detail="Missing uid or filename")
   
   decoded_filename = unquote(filename)
   
   video_path = os.path.join(UPLOAD_FOLDER, uid, decoded_filename)
   
   if not os.path.exists(video_path):
       raise HTTPException(status_code=404, detail="Video not found")
   
   file_size = os.path.getsize(video_path)
   
   # Range 헤더 처리
   range_header = request.headers.get('range')
   
   if range_header:
       start, end = range_header.replace('bytes=', '').split('-')
       start = int(start)
       end = int(end) if end else min(start + CHUNK_SIZE, file_size - 1)
   else:
       start = 0
       end = min(CHUNK_SIZE, file_size - 1)
       
   # 실제 전송할 크기
   chunk_size = end - start + 1
   
   headers = {
       'Content-Range': f'bytes {start}-{end}/{file_size}',
       'Accept-Ranges': 'bytes',
       'Content-Length': str(chunk_size),
       'Content-Type': 'video/webm',
   }
   
   return StreamingResponse(
       video_chunk_generator(video_path, start, end),
       status_code=206 if range_header else 200,
       headers=headers
   )

def video_chunk_generator(file_path: str, start: int, end: int) -> Generator[bytes, None, None]:
   """비디오를 청크 단위로 생성하는 제너레이터"""
   with open(file_path, 'rb') as video_file:
       video_file.seek(start)
       remaining = end - start + 1
       while remaining > 0:
           # 64KB 단위로 읽기
           chunk_size = min(64 * 1024, remaining)
           data = video_file.read(chunk_size)
           if not data:
               break
           remaining -= len(data)
           yield data

@app.delete("/delete/{doc_id}")
async def delete_interview(doc_id: str):
    try:
        # MongoDB 클라이언트 초기화
        mongo_uri = os.environ.get('MONGODB_URI')
        if not mongo_uri:
            raise HTTPException(status_code=500, detail="MongoDB URI가 설정되지 않았습니다")
            
        client = MongoClient(mongo_uri)
        db = client['EmpAI']
        collection = db['video_analysis']
        
        # ObjectId 유효성 검사
        if not ObjectId.is_valid(doc_id):
            raise HTTPException(status_code=400, detail="유효하지 않은 document ID입니다")
        
        # ObjectId로 변환하여 문서 조회
        document = collection.find_one({"_id": ObjectId(doc_id)})
        
        if not document:
            raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다")
        
        user_uid = document.get('uid')
        if not user_uid:
            raise HTTPException(status_code=400, detail="잘못된 문서 형식입니다")

        # 해당 문서의 모든 영상 정보 가져오기
        user_videos = document.get(user_uid, {})
        deleted_files = []
        deletion_errors = []

        # 실제 존재하는 영상만 삭제 시도
        for video_num, video_data in user_videos.items():
            if isinstance(video_data, dict) and 'video_filename' in video_data:
                filename = video_data['video_filename']
                video_path = os.path.join(UPLOAD_FOLDER, user_uid, filename)
                
                if os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                        deleted_files.append(filename)
                        print(f"영상 {video_num} 삭제 성공: {filename}")
                    except Exception as e:
                        error_msg = f"영상 {video_num} ({filename}): {str(e)}"
                        deletion_errors.append(error_msg)
                        print(f"영상 삭제 실패: {error_msg}")

        # 삭제 결과 반환
        if deleted_files:
            success_message = f"삭제된 영상: {', '.join(deleted_files)}"
            if deletion_errors:
                success_message += f"\n삭제 실패: {', '.join(deletion_errors)}"
            
            return JSONResponse(
                content={
                    "success": True,
                    "message": success_message,
                    "deleted_files": deleted_files,
                    "errors": deletion_errors
                }
            )
        else:
            raise HTTPException(
                status_code=404,
                detail="삭제할 영상 파일을 찾을 수 없습니다"
            )
            
    except Exception as e:
        print(f"삭제 중 오류 발생: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"삭제 중 오류 발생: {str(e)}"
            }
        )
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
    "main:app",
    host="0.0.0.0",
    port=5001,
    workers=1,
    reload=False,
    lifespan="on",
    factory=False,
    log_level="debug"  # 더 자세한 로깅
)