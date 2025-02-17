from openai import OpenAI
from typing import Dict, Any

class InterviewEvaluator:
   def __init__(self, api_key: str):
       self.client = OpenAI(api_key=api_key)

   def evaluate_answer(self, job_code: str, resume_data: list, question: str, answer: str) -> Dict[str, Any]:
       # 자기소개서 데이터 포맷팅
       
       print(answer)
       print(question)
       resume_content = "\n".join([
            f"Q: {item['question']}\nA: {item['answer']}" 
            for item in resume_data
        ])
       
       prompt = f"""You are a professional interviewer in the {job_code} field with 20 years of experience. Please evaluate the following interview answers strictly and objectively.

        [Evaluation Target Materials]
        1. Self-introduction: {resume_content}
        2. Interview question: {question}
        3. Answer content: {answer}

        [Scoring Guidelines]
        This interview evaluates the {answer} answer to the {question} question. Only this answer needs to be evaluated.

        - Each evaluation item score must not exceed the maximum score for that item.
        - The question comprehension evaluation is conducted after all items are scored, and a negative score is applied according to the item.
        - Answers unrelated to the question are heavily penalized even if other items are excellent.
        - The relevance of the self-introduction is strictly evaluated according to the IF-ELSE conditions.
        - Typos and grammatical errors are excluded from the evaluation (limits of voice recognition).
        - The score for each item is given only within the range set according to the evaluation criteria, and the maximum score must not be exceeded.
        - The total score is negative Even if it is a score of 0
        - If it is 40 or more, it will be treated as 0 points.
        - Please print the text in Korean

        [Evaluation criteria] - Total score of 50 points
        [Major considerations during evaluation]

        Short answers that are less than 30 characters or cannot be evaluated will be treated as 0 points.

        1. Question comprehension and answer appropriateness (maximum -5 points)
        - Evaluate how well the answer matches the interview question
        - Accurately grasp the essential intent of the question
        - Evaluate the focus and direction of the answer
        Excellent (0 points): If the question is accurately understood and an appropriate answer is provided.
        Good (-1 point): If the intent of the question is partially understood and the answer is somewhat unrelated.
        Average (-3 points): If the answer deviates from the main point of the question.
        Poor (-5 points): If the answer contains profanity, inappropriate language, or is completely unrelated to the question.

        2. Logic and communication (maximum 15 points)
        Evaluate based on the logical structure and communication of the answer, and focus on the overall flow rather than being limited to key words. - Evaluation of logical thinking and communication skills
        - Clarity and persuasiveness of idea connection
        Excellent (14~15 points): The logical structure is very clear and the answer is persuasive, and sufficient evidence and examples are provided to support the argument.
        Good (8~13 points): Basic logic exists, but the explanation is somewhat insufficient, so the persuasiveness is low.
        Average (5~7 points): The logical structure and explanation are very insufficient, so the argument is unclear or difficult to understand.
        Insufficient (0~4 points): The logical development is unclear or not persuasive, and the answer is not understandable.

        3. Evaluation of answers based on self-introduction (maximum 8 points)
        Evaluate the connection with the content of the self-introduction.
        IF When mentioning the content of the self-introduction:
        Excellent (8 points): If it is consistent with the content of the self-introduction and developed into an example.
        Good (4~7 points): If it is consistent with the content of the self-introduction but lacks specificity.
        Insufficient (0~3 points): If it is consistent with the content of the self-introduction but abstract. ELSE (if answer is new):
        Excellent (8 points): New content demonstrates job competency well and is appropriate to the context.
        Good (4~7 points): Job-related but lacks contextual connection.
        Poor (0~3 points): Job-related or contextual connection is lacking.

        4. Practical expertise (up to 7 points)
        Evaluated based on specialized knowledge and practical experience in the relevant field.
        - Depth of expertise in the relevant field
        - Understanding and application of the relevant field
        Excellent (7 points): When specialized knowledge and practical experience in the relevant field are abundantly demonstrated, and when in-depth understanding is explained with cases.
        Good (4~6 points): When basic specialized knowledge and practical understanding in the relevant field are demonstrated, and when one explains one's experience using general cases.
        Average (2~3 points): When specialized knowledge and practical experience in the relevant field are somewhat lacking, and understanding of the topic is limited.
        Poor (0~1 point): When expertise or practical understanding in the relevant field is severely lacking, and related knowledge is lacking.

        5. Problem-solving ability (up to 10 points)
        Evaluated the ability to identify problems and present solutions. - Problem recognition ability
        - Creativity and feasibility of solution
        Excellent (10 points): When the essence of the problem is accurately identified and a solution is logically presented.
        Good (6~9 points): When the essence of the problem is identified to some extent, but the solution process is somewhat insufficient.
        Average (3~5 points): When the essence of the problem is identified to some extent, but the solution process is very insufficient.
        Insufficient (0~2 points): When the problem identification or solution process is unclear.

        6. Completeness of the answer (maximum 10 points)
        The answer is evaluated based on clarity, logical structure, appropriateness to the question, and expertise.
        - Comprehensive answer quality
        - Overall evaluation from the interviewer's perspective
        Excellent (10 points): When the answer is clear, logically consistent, highly reliable, including professional knowledge and evidence, and provides sufficient information without additional questions through appropriate examples and explanations.
        Good (6~9 points): The answer has basic completeness and generally satisfies the key elements of the question, but some content is lacking or needs to be supplemented, and the concept is relatively clear, but detailed evidence, examples, and additional explanations are lacking.
        Moderate (3-5 points): The answer only partially satisfies the question, is vague in explanation or key content, lacks logical structure or consistency, has low expertise or reliability, and is difficult to obtain sufficient information from the answer alone.
        Insufficient (0-2 points): The answer has little to do with the question or does not satisfy key elements, is difficult to understand due to lack of logical flow or many illogical expressions, has unclear concepts or brief explanations, is unreliable, and does not provide substantial information.

        Responses must be written in the following JSON format:
        {{
        "세부점수": {{
                "질문이해도와답변적합성": [Score],
                "논리성과전달력": [Score],
                "자기소개서기반답변평가": "[Score]",
                "실무전문성": [Score],
                "문제해결력": [Score],
                "답변의완성도": [Score]
        }},
        - Total score is the sum of sub-scores: If the total score is negative, it is treated as 0 points. 
        "총점": "[Total Score]점",
        "답변강점": "The specific and objective strengths found in the applicant's answers are analyzed in detail from the job perspective. Clearly explain how each strength can have a positive impact on the job, and present potential value in actual work situations. Strengths are described with specific cases based on the applicant's actual experience and capabilities.",
        "답변개선사항": "The weaknesses of the answer are analyzed constructively and specifically. Clearly point out areas that need improvement, and suggest specific, actionable methods for each improvement point. The improvement plan is not a simple criticism, but a practical and positive approach that opens up the applicant's growth potential. The improvement suggestions are linked to the job requirements and give contextual meaning.",
        "답변종합평가": "The applicant's answers are comprehensively analyzed from a 360-degree perspective. All evaluation factors such as question comprehension, logic, expertise, and potential are reviewed in a three-dimensional manner to determine suitability for the job. The strengths and improvements of the answer are evaluated in a balanced manner, and the applicant's growth potential and actual job performance ability are comprehensively diagnosed.",
        "긍정키워드": "Extract 0 to 3 key positive keywords that represent the strengths and potential of the answer. These keywords should be able to concisely express the applicant's capabilities and job suitability.",
        "부정키워드": "Extract 0 to 3 key negative keywords that need improvement from the answer. These keywords should be able to clearly point out the applicant's weaknesses and suggest future growth directions."
        }}"""
       try:
           response = self.client.chat.completions.create(
               model="gpt-4o-mini",
               messages=[
                   {"role": "system", "content": prompt}
               ],
               temperature=0.5,
               max_tokens=2000
           )

           import json
           evaluation_text = response.choices[0].message.content
           # JSON 부분만 추출하여 파싱
           json_str = evaluation_text.strip()
           evaluation_dict = json.loads(json_str)
           print(evaluation_dict)
           return evaluation_dict

       except Exception as e:
           return {
               "총점": "0점",
               "답변강점": "평가 중 오류 발생",
               "답변개선사항": "평가 중 오류 발생",
               "답변종합평가": "평가 중 오류 발생"
           }