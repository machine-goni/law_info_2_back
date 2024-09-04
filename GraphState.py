from typing import TypedDict
from langchain_core.documents.base import Document


# GraphState 상태를 저장하는 용도로 사용
class GraphState(TypedDict):
    question: str                   # 질문
    context: str                    # 문서의 검색 결과
    answer: str                     # 답변
    relevance: bool                 # 답변의 문서에 대한 관련성
    filter: str                     # vector db 검색시 사용할 필터 내용
    retrieved_docs: list[Document]  # Document 타입 그대로의 검색된 문서 리스트
    vectordb_score: float           # context 로 넘겨진 문서의 pinecone 유사도 점수
    bm25_score: float               # context 로 넘겨진 문서의 bm25 유사도 점수
    vectordb_choice: dict           # ensemble(pinecone+bm25) 에서 최고 점수를 받은 top 1 문서 정보
    etc_relevant_precs: list        # 가장은 아니지만 연관성이 있는 차순위 판례번호들
    
    paper_content: dict             # 서류작성 입력사항
    post_conversation: bool         # 이어지는 대화인가?