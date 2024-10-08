# backend main code 로 frontend 의 입력을 전달할 receiver

from ask_questions_experimetal import AskQuestions


class RecvQuestions:
    def __init__(self):
        self.ask_instance = AskQuestions()
    
    
    def stop_timer(self):
        self.ask_instance.stop_timer()
    

    async def question(self, question, case_type) -> dict:
        question_result = await self.ask_instance.question(question, case_type)
        statistics_url = None
        
        # 사건 종류에 따라 보여줄 통계가 다르다
        if case_type == '형사':
            statistics_url = await self.ask_instance.statistics_criminal(question)
            #statistics_url = self.ask_instance.statistics_criminal(f'{question} {question_result.get("answer")}')
        elif case_type == '행정':
            statistics_url = self.ask_instance.statistics_administrative()
            
        result_data = {}
        result_data["answer"] = question_result.get("answer")
        result_data["relevance"] = question_result.get("relevance")
        result_data["vectordb_choice"] = question_result.get("vectordb_choice")
        result_data["etc_relevant_precs"] = question_result.get("etc_relevant_precs")
        result_data["statistics_url"] = statistics_url
        
        #print(f"/RecvQuestions_question - {question}: \n{result_data}")
                        
        return result_data
    
    
    async def advice(self, dialogue_session_id, is_post_conversation, status, question, add_info) -> dict:
        result = await self.ask_instance.advice(dialogue_session_id, is_post_conversation, status, question, add_info)
            
        result_data = {}
        result_data["answer"] = result.get("answer")
        
        #print(f"/RecvQuestions_advice - {question}: \n{result_data}")
                        
        return result_data
    
    
    async def write_paper_1(self, reason, fact, ask, point, receiver, sender, phone, appendix, style) -> dict:
        result = await self.ask_instance.write_paper_1(reason, fact, ask, point, receiver, sender, phone, appendix, style)
            
        result_data = {}
        result_data["answer"] = result.get("answer")
                        
        return result_data
    
    
    async def write_paper_2(self, \
        sender_name, \
        receiver_name, court, \
        amount, ask_interest, transmittal_fee, stamp_fee, \
        ask_reason, ask_reason_detail, appendix) -> dict:
        
        result = await self.ask_instance.write_paper_2(sender_name, \
            receiver_name, court, \
            amount, ask_interest, transmittal_fee, stamp_fee, \
            ask_reason, ask_reason_detail, appendix)
            
        result_data = {}
        result_data["answer"] = result.get("answer")
                        
        return result_data
    
    
    async def write_paper_4(self, dialogue_session_id, is_post_conversation, \
        sender_name, receiver_name, \
        case_no, case_name, case_purpose, case_cause, case_prove, case_appendix, case_court, \
        rebut, appendix, add_info) -> dict:
        
        result = await self.ask_instance.write_paper_4(dialogue_session_id, is_post_conversation, \
            sender_name, receiver_name, \
            case_no, case_name, case_purpose, case_cause, case_prove, case_appendix, case_court, \
            rebut, appendix, add_info)
            
        result_data = {}
        result_data["answer"] = result.get("answer")
                        
        return result_data
    
    
    async def write_paper_5(self, dialogue_session_id, is_post_conversation, \
        sender_name, receiver_name, \
        receiver_etc, purpose, crime_time, crime_history, damage, reason, evidence, \
        etc_accuse, station, add_info) -> dict:
        
        result = await self.ask_instance.write_paper_5(dialogue_session_id, is_post_conversation, \
            sender_name, receiver_name, \
            receiver_etc, purpose, crime_time, crime_history, damage, reason, evidence, \
            etc_accuse, station, add_info)
            
        result_data = {}
        result_data["answer"] = result.get("answer")
                        
        return result_data
    
    
    async def write_paper_6(self, dialogue_session_id, is_post_conversation, \
        sender_name, receiver_name, \
        case_name, purpose, reason, evidence, \
        court, add_info) -> dict:
        
        result = await self.ask_instance.write_paper_6(dialogue_session_id, is_post_conversation, \
            sender_name, receiver_name, \
            case_name, purpose, reason, evidence, \
            court, add_info)
            
        result_data = {}
        result_data["answer"] = result.get("answer")
                        
        return result_data
    
    