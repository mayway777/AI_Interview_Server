from pymongo import MongoClient
from dotenv import load_dotenv
import os

# .env.local 파일 로드
load_dotenv('.env.local')

def safe_int_convert(value):
    """문자열을 정수로 안전하게 변환하는 함수"""
    if value is None:
        return 0
    try:
        if isinstance(value, str):
            num_str = ''.join(filter(str.isdigit, value)) 
            return int(num_str) if num_str else None
        elif isinstance(value, (int, float)):
            return int(value)
    except:
        return 0
    return 0

def extract_gpt_score(gpt_eval):
    """GPT 평가에서 총점을 추출하는 함수"""
    # gpt_eval이 None이거나 빈 딕셔너리인 경우
    if not gpt_eval:
        return 0  # 명시적으로 0 반환
    
    # 딕셔너리가 아닌 경우 처리
    if not isinstance(gpt_eval, dict):
        return 0
    
    try:
        # 총점 키가 없는 경우 처리
        score_str = gpt_eval.get('총점', '0')
        
        # 점수 문자열에서 숫자 추출
        if score_str:
            # 문자열의 숫자만 추출 (소수점 포함)
            score = int(''.join(filter(str.isdigit, str(score_str))))
            return score
        
        # 추출된 숫자가 없으면 0 반환
        return 0
    except:
        return 0

def calculate_speaking_speed_score(speed):
    if speed is None:
        return 0
    try:
        speed = int(speed)
        if 100 <= speed <= 120:
            return 10
        elif (85 <= speed < 100) or (120 < speed <= 130):
            return 8
        elif (75 <= speed < 85) or (130 < speed <= 140):
            return 6
        elif (60 <= speed < 75) or (140 < speed <= 150):
            return 4
        else:
            return 2
    except:
        return 0

def calculate_filler_score(count):
    if count is None:
        return 0
    try:
        count = int(count)
        if 0 <= count <= 2:
            return 5
        elif 3 <= count <= 5:
            return 4
        elif 6 <= count <= 9:
            return 3
        elif 10 <= count <= 15:
            return 2
        else:
            return 1
    except:
        return 0

def calculate_silence_score(count):
    if count is None:
        return 0
    try:
        count = int(count)
        if 0 <= count <= 2:
            return 5
        elif 3 <= count <= 5:
            return 4
        elif 6 <= count <= 9:
            return 3
        elif 10 <= count <= 15:
            return 2
        else:
            return 1
    except:
        return 0

def calculate_voice_variation_score(variation):
    """목소리 변동성 점수 계산"""
    if variation is None:
        return 0
    
    try:
        if isinstance(variation, str):
            percent = int(''.join(filter(str.isdigit, variation)))
        else:
            percent = int(variation)
            
        if 30 <= percent <= 40:
            return 10
        elif (25 <= percent < 30) or (40 < percent <= 45):
            return 8
        elif (20 <= percent < 25) or (45 < percent <= 50):
            return 6
        elif (15 <= percent < 19) or (50 < percent <= 55):
            return 4
        else:  
            return 2
    except:
        return 0

def calculate_expression_score(emotion_data):
    """표정 분석 점수 계산"""
    if emotion_data is None or not isinstance(emotion_data, dict):
        return 0
    
    try:
        neutral = float(emotion_data.get('Neutral', 0))+ \
                float(emotion_data.get('Surprise', 0))
        
        positive = float(emotion_data.get('Happy', 0))
       
        negative = float(emotion_data.get('Angry', 0)) + \
                  float(emotion_data.get('Disgust', 0)) + \
                  float(emotion_data.get('Fear', 0)) + \
                  float(emotion_data.get('Sad', 0))
                  
        
        if (40 <= neutral <= 60) and (20 <= positive <= 60) and (0 <= negative <= 5):
            return 10
        elif (45 <= neutral <= 90) and (5 <= positive <= 70) and (0 <= negative <= 10):
            return 8
        elif (30 <= neutral <= 100) and (0 <= positive <= 90) and (0 <= negative <= 20):
            return 6
        elif (20 <= neutral <= 100) and (0 <= positive <= 100) and (0 <= negative <= 60):
            return 4
        else:
            return 2
    except:
        return 0

def calculate_head_tilt_score(tilt_data):
    """머리 기울기 점수 계산"""
    if tilt_data is None or not isinstance(tilt_data, dict):
        return 0
    
    try:
        center_percent = float(tilt_data.get('center', 0))
        if center_percent >= 90:
            return 5
        elif 80 <= center_percent < 90:
            return 4
        elif 70 <= center_percent < 80:
            return 3
        elif 60 <= center_percent < 70:
            return 2
        else:
            return 1
    except:
        return 0

def calculate_eye_tracking_score(tracking_data):
    """시선 추적 점수 계산"""
    if tracking_data is None or not isinstance(tracking_data, dict):
        return 0
    
    try:
        center_percent = float(tracking_data.get('center', 0))
        if center_percent >= 90:
            return 5
        elif 80 <= center_percent < 90:
            return 4
        elif 70 <= center_percent < 80:
            return 3
        elif 60 <= center_percent < 70:
            return 2
        else:
            return 1
    except:
        return 0

class VideoDB:
    def __init__(self):
        self.uri = os.getenv('MONGODB_URI')
        if not self.uri:
            raise ValueError("MongoDB URI is not set in environment variables")
        
        try:
            self.client = MongoClient(self.uri)
            self.db = self.client['EmpAI']
            self.collection = self.db['video_analysis']
        except Exception as e:
            print(f"MongoDB 연결 실패: {str(e)}")
            raise

    def calculate_scores(self, analysis_result):
        """분석 결과를 바탕으로 점수 계산"""
        if not analysis_result:
            return None

        scores = {
            "말하기속도": calculate_speaking_speed_score(analysis_result.get("말하기속도")),
            "추임새/침묵": (
                (calculate_filler_score(analysis_result.get("추임새갯수")) + 
                 calculate_silence_score(analysis_result.get("침묵갯수"))) / 2 * 2 
                if analysis_result.get("추임새갯수") is not None and 
                   analysis_result.get("침묵갯수") is not None 
                else None
            ),
            "목소리변동성": calculate_voice_variation_score(analysis_result.get("목소리변동성")),
            "표정분석": calculate_expression_score(analysis_result.get("감정_%")),
            "머리기울기": calculate_head_tilt_score(analysis_result.get("머리기울기_%")),
            "시선분석": calculate_eye_tracking_score(analysis_result.get("아이트래킹_%")),
            "답변평가": extract_gpt_score(analysis_result.get("GPT_평가")) 
        }

        
        return scores

    def create_initial_document(self, initial_doc):
        """초기 분석 문서 생성"""
        try:
            existing_doc = self.collection.find_one({
                "uid": initial_doc["uid"],
                "self_id": initial_doc["self_id"],
                "time": initial_doc["time"]
            })
            
            if existing_doc:
                return existing_doc["_id"]
            
            result = self.collection.insert_one(initial_doc)
            return result.inserted_id
            
        except Exception as e:
            print(f"초기 문서 생성 중 오류 발생: {str(e)}")
            raise

    def update_analysis_results(self, doc_id, uid, video_number, analysis_result):
    
        scores = self.calculate_scores(analysis_result)
        try:
                formatted_result = {
                    "video_number": video_number,
                    "video_filename": analysis_result.get("video_filename"),
                    "question": analysis_result.get("question"),
                    "답변": analysis_result.get("답변"),
                    "감정_%": analysis_result.get("감정_%") or None,
                    "머리기울기_%": analysis_result.get("머리기울기_%") or None,
                    "아이트래킹_%": analysis_result.get("아이트래킹_%") or None,
                    
                    "말하기속도": safe_int_convert(analysis_result.get("말하기속도")) or None,
                    "평속대비차이": analysis_result.get("평속대비차이") or None,
                    "추임새갯수": safe_int_convert(analysis_result.get("추임새갯수")) or None,
                    "침묵갯수": safe_int_convert(analysis_result.get("침묵갯수")) or None,
                    "목소리변동성": analysis_result.get("목소리변동성") or None,
                    
                    "Score": scores or {
                        "말하기속도": 0,
                        "추임새/침묵": 0,
                        "목소리변동성": 0,
                        "표정분석": 0,
                        "머리기울기": 0,
                        "시선분석": 0,
                        "답변평가": 0
                    },
                    "Evaluation": analysis_result.get("GPT_평가") or {
                        "답변강점": "오류",
                        "답변개선사항": "오류",
                        "답변종합평가": "오류",
                        "긍정키워드": "오류",
                        "부정키워드":"오류",
                        "세부점수": {
                            "질문이해도와답변적합성": 0,
                            "논리성과전달력": 0,
                            "자기소개서기반답변평가": 0,
                            "실무전문성": 0,
                            "문제해결력": 0,
                            "답변의완성도": 0
                        },
                        "총점": 0
                    },
                }
                
                update_path = f"{uid}.{video_number}"
                result = self.collection.update_one(
                    {"_id": doc_id},
                    {"$set": {update_path: formatted_result}}
                )
                
                return result.modified_count > 0
            
        except Exception as e:
                print(f"분석 결과 업데이트 중 오류 발생: {str(e)}")
                return False

# DB 인스턴스 생성
video_db = VideoDB()

# 외부에서 사용할 함수들
def create_initial_document(initial_doc):
    return video_db.create_initial_document(initial_doc)

def update_analysis_results(doc_id, uid, video_number, analysis_result):
    return video_db.update_analysis_results(doc_id, uid, video_number, analysis_result)