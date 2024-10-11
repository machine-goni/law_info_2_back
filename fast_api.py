'''
fastapi 는 pip install 로 설치해줘야 하고
서버를 돌리기 위해 pip install "uvicorn[standard]" 도 실행
'''

# backend 를 FastAPI 로 구현

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware      # 보안을 위해 CORS 설정을 해줘야 한다.
from fastapi import Request                            # 들어오는 request 의 origin 을 확인해보고 싶으면 Request 를 import 해주면 된다.
from pydantic import BaseModel
from receive_questions import RecvQuestions
#import json
#import logging

import atexit   # 프로그램 종료시 호출을 위해
#import signal
#import sys


# 설정 로깅
#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)


'''
POST 메세지를 받을 클래스. FastAPI 에서는 이걸 model 이라고 부른다.
클래는 안의 변수는 메세지로 받을 param 이다.
'''

class User_inputs(BaseModel):
    question : str
    case_type : str
    
class User_inputs_advice(BaseModel):
    dialogue_session_id: str
    is_post_conversation : bool
    status : str
    question : str
    add_info : str
    
class User_inputs_paper_1(BaseModel):
    reason : str
    fact : str
    ask : str
    point : str
    receiver : str
    sender : str
    phone : str
    appendix : str
    style : str
    
class User_inputs_paper_2(BaseModel):
    sender_name : str
    receiver_name : str
    court : str
    amount : str
    ask_interest : str
    transmittal_fee : str
    stamp_fee : str
    ask_reason : str
    ask_reason_detail : str
    appendix : str
    
class User_inputs_paper_4(BaseModel):
    dialogue_session_id: str
    is_post_conversation : bool
    sender_name : str
    receiver_name : str
    case_no : str
    case_name : str
    case_purpose : str
    case_cause : str
    case_prove : str
    case_appendix : str
    case_court : str
    rebut : str
    appendix : str
    add_info : str
    
class User_inputs_paper_5(BaseModel):
    dialogue_session_id: str
    is_post_conversation : bool
    sender_name : str
    receiver_name : str
    receiver_etc : str
    purpose : str
    crime_time : str
    crime_history : str
    damage : str
    reason : str
    evidence : str
    etc_accuse : str
    station : str
    add_info : str
    
class User_inputs_paper_6(BaseModel):
    dialogue_session_id: str
    is_post_conversation : bool
    sender_name : str
    receiver_name : str
    case_name : str
    purpose : str
    reason : str
    evidence : str
    court : str
    add_info : str


# FastAPI instance
app = FastAPI()


#-- CORS(Cross-Origin Resource Sharing, 교차-출처 리소스 공유) 설정 ----------------

""" 
`origins` 리스트에 명시된 도메인에서만 API에 접근할 수 있도록 설정한다. (origins 의 도메인 체크는 붙어오는 hearder로 
확인하는 것인데 보통 브라우저에서 도메인을 hearder 에 붙여주지만 streamlit 은 그게 안붙어온다. 그래서 streamlit 으로 
백날 CORS 테스트해봐야 프리패스라는 결과에 직면할 것이다. 괜히 streamlit 으로 테스트 한다고 힘빼지 말자. 굳이 streamlit 으로 
테스트해보고 싶다면 request 보낼때 hearder origin 에 도메인을 넣어서 보내주면 된다. 하지만 실서비스에서는 보안상 당연히 하지 말아야 한다)
또한, 세션을 사용하여 특정 사용자만 접근할 수 있도록 할 수도 있다. 
FastAPI에서 세션을 사용하려면 `fastapi_sessions`와 같은 라이브러리를 사용할 수 있다. 
세션을 통해 사용자가 로그인했는지 확인하고, 로그인하지 않은 사용자는 접근할 수 없도록 할 수 있다.

참고: https://fastapi.tiangolo.com/ko/tutorial/cors/#corsmiddleware
"""
# 허용할 도메인 리스트
# 테스트시 주의할 점은 백/프론트, 브라우져까지 완전히 다시 껏다 켜야 제대로 확인할 수 있다. 그러지 않을 경우 차단되야할때 안되고 안되야할때 되는 경우가 있다.
origins = [
    #"*",
    "https://legal-with-ai.pages.dev",
    #"http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # 교차-출처 요청을 보낼 수 있는 출처의 리스트. 모든 출처를 허용하기 위해 ['*'] 를 사용할 수 있다.
    allow_credentials=True,     # 교차-출처 요청시 쿠키 지원 여부를 설정. 기본값은 False. 또한 해당 항목을 허용할 경우 allow_origins 는 ['*'] 로 설정할 수 없으며, 출처를 반드시 특정한다.
    allow_methods=["*"],        # 교차-출처 요청을 허용하는 HTTP 메소드의 리스트. 기본값은 ['GET'] 이다. ['*'] 을 사용하여 모든 표준 메소드들을 허용할 수 있다.
    allow_headers=["*"],        # 교차-출처를 지원하는 HTTP 요청 헤더의 리스트. 기본값은 [] 이다. 모든 헤더들을 허용하기 위해 ['*'] 를 사용할 수 있다. Accept, Accept-Language, Content-Language 그리고 Content-Type 헤더는 CORS 요청시 언제나 허용된다.
)

# 위의 CORS 설정에 추가하여 허용 origin 이 아니라면 exception 을 내도록 코드 추가
# 추가하는 이유는 CORS 에 의해 막힌다 해도 이는 브라우저 측에서 막는 것이기 때문에 서버 내부 프로세스는 수행이 된다.
# 그래서 막힌 요청에 대해서는 프로세스가 중단되도록 하기 위함이다.
@app.middleware("http")
async def check_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    #print(f"origin: {origin}")

    # 요청 헤더의 Origin이 허용된 origins 목록에 없으면 403 오류 발생
    if (origin not in origins) and ("*" not in origins):
        # `HTTPException`이 발생하면 해당 요청에 대한 처리가 중단되지만, 서버 자체는 정상적으로 동작하며 다른 요청을 계속 처리할 수 있다. 
        # 그렇기 때문에 HTTPException 대해 `try`/`except` 블록으로 감싸줄 필요는 없다.
        raise HTTPException(status_code=403)

    response = await call_next(request)
    return response

#-------------------------------------------------------------------- CORS 설정 --


# backend main code 로 frontend 의 입력을 전달할 receiver
receiver = RecvQuestions()

# 프로그램 종료 시 호출
def on_exit():
    print("프로그램 종료 중...")
    if receiver != None:
        receiver.stop_timer()

atexit.register(on_exit)        # on_exit 등록
'''
# KeyboardInterrupt 예외 처리
def signal_handler(sig, frame):
    print("Ctrl + C 감지됨")
    on_exit()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
'''


'''
FastAPI instance 로 REST API 를 정의 한다.
@app.post("/init") 안의 "/init" 는 route
'''

@app.post("/init")
async def operate():
    try:
        if receiver != None:
            #logging.info("Receiver is initialized")
            return True
        else:
            #logging.info("Receiver is not initialized")
            return False
    
    except Exception as e:
        #logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 판례검색과 질문
@app.post("/question")
async def operate(input:User_inputs):
#async def operate(request: Request, input:User_inputs):
    try:
        # request: Request 를 받아 아래처럼 origin 을 확인해 볼 수 있다. request 는 프론트엔드에서 따로 파라메터를 넣어줄 필요는 없다.
        #origin = request.headers.get("origin")
        #print(f"/question - origin: {origin}")
        
        result = await receiver.question(input.question, input.case_type)
        #print(f"/question - {input.question}: \n{result}")
        
        # 보내기 직전에 json 으로 변환시킨다. json.dumps 대신 JSONResponse 를 사용하면 알아서 맞는 인코딩을 적용해 준다고 한다.
        #return json.dumps(result)
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 법률 조언
@app.post("/advice")
async def operate(input:User_inputs_advice):
    try:
        result = await receiver.advice(input.dialogue_session_id, input.is_post_conversation, input.status, input.question, input.add_info)
        #print(f"/advice - {input.question}: \n{result}")
        
        # 보내기 직전에 json 으로 변환시킨다
        #return json.dumps(result)
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 서류작성 - 내용증명
@app.post("/write-paper-1")
async def operate(input:User_inputs_paper_1):
    try:
        result = await receiver.write_paper_1(input.reason, input.fact, input.ask, input.point, input.receiver, input.sender, input.phone, input.appendix, input.style)
        
        # 보내기 직전에 json 으로 변환시킨다
        #return json.dumps(result)
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 서류작성 - 지급명령신청서
@app.post("/write-paper-2")
async def operate(input:User_inputs_paper_2):
    try:
        result = await receiver.write_paper_2(input.sender_name, \
            input.receiver_name, input.court, \
            input.amount, input.ask_interest, input.transmittal_fee, input.stamp_fee, \
            input.ask_reason, input.ask_reason_detail, input.appendix)
        
        # 보내기 직전에 json 으로 변환시킨다
        #return json.dumps(result)
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 서류작성 - 답변서
@app.post("/write-paper-4")
async def operate(input:User_inputs_paper_4):
    try:
        result = await receiver.write_paper_4(input.dialogue_session_id, input.is_post_conversation, input.sender_name, input.receiver_name, \
            input.case_no, input.case_name, input.case_purpose, input.case_cause, input.case_prove, input.case_appendix, input.case_court, \
            input.rebut, input.appendix, input.add_info)
        
        # 보내기 직전에 json 으로 변환시킨다
        #return json.dumps(result)
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 서류작성 - 고소장
@app.post("/write-paper-5")
async def operate(input:User_inputs_paper_5):
    try:
        result = await receiver.write_paper_5(input.dialogue_session_id, input.is_post_conversation, input.sender_name, input.receiver_name, \
            input.receiver_etc, input.purpose, input.crime_time, input.crime_history, input.damage, input.reason, input.evidence, \
            input.etc_accuse, input.station, input.add_info)
        
        # 보내기 직전에 json 으로 변환시킨다
        #return json.dumps(result)
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 서류작성 - (민사)소장
@app.post("/write-paper-6")
async def operate(input:User_inputs_paper_6):
    try:
        result = await receiver.write_paper_6(input.dialogue_session_id, input.is_post_conversation, input.sender_name, input.receiver_name, \
            input.case_name, input.purpose, input.reason, input.evidence, \
            input.court, input.add_info)
        
        # 보내기 직전에 json 으로 변환시킨다
        #return json.dumps(result)
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# For running the FastAPI server we need to run the following command:
# uvicorn fast_api:app --reload
# fast_api 는 실행할 FastAPI 가 구현되어있는 python script
# 커맨드를 실행하면 접속할 수 있는 local url 이 나온다
# http://127.0.0.1:8000/docs 를 열면 Swagger UI 를 볼 수 있다.