'''
fastapi 는 pip install 로 설치해줘야 하고
서버를 돌리기 위해 pip install "uvicorn[standard]" 도 실행
'''

# backend 를 FastAPI 로 구현

from fastapi import FastAPI
from pydantic import BaseModel
from receive_questions import RecvQuestions
import json

import atexit   # 프로그램 종료시 호출을 위해
#import signal
#import sys

'''
POST 메세지를 받을 클래스. FastAPI 에서는 이걸 model 이라고 부른다.
클래는 안의 변수는 메세지로 받을 param 이다.
'''
#class Workflow_type(BaseModel):
#    workflow_type : str

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
    #sender_addr : str
    #sender_phone : str
    receiver_name : str
    #receiver_addr : str
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
def operate():
    result = receiver.init()
    #print(f"/init: {result}")
    
    return result

'''
@app.post("/build")
def operate(input:Workflow_type):
    result = receiver.build_workflow(input.workflow_type)
    #print(f"/build: {result}")

    return result
'''

# 판례검색과 질문
@app.post("/question")
def operate(input:User_inputs):
    result = receiver.question(input.question, input.case_type)
    #print(f"/question - {input.question}: \n{result}")
    
    # 보내기 직전에 json 으로 변환시킨다
    return json.dumps(result)


# 법률 조언
@app.post("/advice")
def operate(input:User_inputs_advice):
    result = receiver.advice(input.dialogue_session_id, input.is_post_conversation, input.status, input.question, input.add_info)
    #print(f"/advice - {input.question}: \n{result}")
    
    # 보내기 직전에 json 으로 변환시킨다
    return json.dumps(result)


# 서류작성 - 내용증명
@app.post("/write-paper-1")
def operate(input:User_inputs_paper_1):
    result = receiver.write_paper_1(input.reason, input.fact, input.ask, input.point, input.receiver, input.sender, input.phone, input.appendix, input.style)
    
    # 보내기 직전에 json 으로 변환시킨다
    return json.dumps(result)


# 서류작성 - 지급명령신청서
@app.post("/write-paper-2")
def operate(input:User_inputs_paper_2):
    #result = receiver.write_paper_2(input.sender_name, input.sender_addr, input.sender_phone, \
    #    input.receiver_name, input.receiver_addr, input.court, \
    result = receiver.write_paper_2(input.sender_name, \
        input.receiver_name, input.court, \
        input.amount, input.ask_interest, input.transmittal_fee, input.stamp_fee, \
        input.ask_reason, input.ask_reason_detail, input.appendix)
    
    # 보내기 직전에 json 으로 변환시킨다
    return json.dumps(result)


# 서류작성 - 답변서
@app.post("/write-paper-4")
def operate(input:User_inputs_paper_4):
    result = receiver.write_paper_4(input.dialogue_session_id, input.is_post_conversation, input.sender_name, input.receiver_name, \
        input.case_no, input.case_name, input.case_purpose, input.case_cause, input.case_prove, input.case_appendix, input.case_court, \
        input.rebut, input.appendix, input.add_info)
    
    # 보내기 직전에 json 으로 변환시킨다
    return json.dumps(result)


# 서류작성 - 고소장
@app.post("/write-paper-5")
def operate(input:User_inputs_paper_5):
    result = receiver.write_paper_5(input.dialogue_session_id, input.is_post_conversation, input.sender_name, input.receiver_name, \
        input.receiver_etc, input.purpose, input.crime_time, input.crime_history, input.damage, input.reason, input.evidence, \
        input.etc_accuse, input.station, input.add_info)
    
    # 보내기 직전에 json 으로 변환시킨다
    return json.dumps(result)


# 서류작성 - (민사)소장
@app.post("/write-paper-6")
def operate(input:User_inputs_paper_6):
    result = receiver.write_paper_6(input.dialogue_session_id, input.is_post_conversation, input.sender_name, input.receiver_name, \
        input.case_name, input.purpose, input.reason, input.evidence, \
        input.court, input.add_info)
    
    # 보내기 직전에 json 으로 변환시킨다
    return json.dumps(result)


# For running the FastAPI server we need to run the following command:
# uvicorn fast_api:app --reload
# fast_api 는 실행할 FastAPI 가 구현되어있는 python script
# 커맨드를 실행하면 접속할 수 있는 local url 이 나온다
# http://127.0.0.1:8000/docs 를 열면 Swagger UI 를 볼 수 있다.