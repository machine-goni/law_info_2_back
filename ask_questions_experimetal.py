# pip freeze > requirements.txt
# 위 명령어로 requirements.txt 를 뽑을 수 있다
# requirements.txt 를 사용해 일괄 설치하려면 아래명령어.
# pip install -r requirements.txt
# 만약 패키지간 호환성 문제로 에러가 날때 제안되는 방법은 2가지 이다.
'''
To fix this you could try to:
1. 지정한 패키지 버전의 범위를 느슨하게 합니다.
2. pip가 종속성 충돌을 해결하려고 시도할 수 있도록 패키지 버전을 제거하세요.
'''
# requirements.txt 를 사용해 일괄 삭제하려면 아래명령어.
# pip uninstall -r requirements.txt -y


import os
#import json
from dotenv import load_dotenv
from typing import List
import datetime
import gc

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
#from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents.base import Document
# BM25
from langchain_community.retrievers import BM25Retriever
# 아래의 패키지를 사용하고 싶지만 아래의 패키지를 사용하려면 JVM 을 써야하고 그렇게 되면 로컬에서는 상관없지만
# 클라우드에 올릴땐 docker 로 만들어서 올려야 한다. 그래서 그냥 langchain 걸 쓴다.
# BM25 (커스텀 구현한 한국어 형태소 분석기 적용). 아래 3가지 중에 어떤게 가장 성능이 좋은지는 확인을 해봐야 알겠지만,
# 여기서는 kkma 를 사용한다.
#from langchain_teddynote.retrievers import (
    #KiwiBM25Retriever,  # Kiwi + BM25
#    KkmaBM25Retriever,  # KonlPy(Kkma) + BM25
    #OktBM25Retriever    # KonlPy(Okt) + BM25
#)
# Ensemble Retriever
#from langchain.retrievers import EnsembleRetriever
# tavily search
#from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_community.tools.tavily_search import TavilySearchResults
# tavily search error
#from tavily import TavilyClient, UsageLimitExceededError
# web site url 로 문서 load
#from langchain_community.document_loaders import WebBaseLoader

# langgraph 구조에 필요한 패키지들
from GraphState import GraphState
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from utils import format_docs, format_searched_docs, retriever_with_score, pretty_print, get_index_url, get_prompts_by_casetype
from operator import itemgetter
import pprint
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import RunnableConfig, RunnableLambda


PINECONE_INDEX_NAME = "search-precedents"


# LangGraph 구현 및 설명 참조: https://medium.com/@yoony1007/%EB%9E%AD%EC%B2%B4%EC%9D%B8-%EC%BD%94%EB%A6%AC%EC%95%84-%EB%B0%8B%EC%97%85-2024-q2-%ED%85%8C%EB%94%94%EB%85%B8%ED%8A%B8-%EC%B4%88%EB%B3%B4%EC%9E%90%EB%8F%84-%ED%95%A0-%EC%88%98-%EC%9E%88%EB%8A%94-%EA%B3%A0%EA%B8%89-rag-%EB%8B%A4%EC%A4%91-%EC%97%90%EC%9D%B4%EC%A0%84%ED%8A%B8%EC%99%80-langgraph-%EC%A0%9C%EC%9E%91-e8bec8adfef4


class AskQuestions:
    
    def __init__(self):
        load_dotenv()
        os.environ['OPENAI_API_KEY'] = os.getenv("openai_api_key")
        os.environ['PINECONE_API_KEY'] = os.getenv("pinecone_api_key")
        os.environ["TAVILY_API_KEY"] = os.getenv("tavily_api_key")
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("langchain_api_key")
        self.index_go_kr_key = os.getenv("index_go_kr_key")
        
        # embedding model instance 생성
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        # vector store. 이미 문서를 임베딩해서 vector store 에 넣었다면 끌어다 쓰면 된다.
        self.vectorstore = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, self.embeddings)
        
        # LLM Model
        self.model_type = 1
        if self.model_type == 0:
            self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        elif self.model_type == 1:
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        else :
            self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.1)
            
        self.case_type = ""
        self.store = {}     # 세션 기록(대화 히스토리)을 저장할 딕셔너리
        
        
    # -- node 가 될 함수들 (langgraph 에서 node 는 GraphState 를 받고 GraphState 를 넘겨줘야 한다) --
    
    # vector store 에서 관련 있는 문서 검색
    def retrieve_document(self, state: GraphState) -> GraphState:
        nearest_k = 10
        
        # pinecone
        dense_retrieved_docs = retriever_with_score.invoke({"query": state["question"], "vectorstore": self.vectorstore, "k": nearest_k, "filter": state["filter"]})
        
        # 필터링을 안한다면 전혀 유사도가 없어도 결과를 반환하지만, 필터링을 하게되면 빈값이 반환될 수 있다. 이럴땐 그냥 리턴해 준다.
        if dense_retrieved_docs == None:
            return GraphState(retrieved_docs=[])
        
        #print('pinecones choice:')
        #print(dense_retrieved_docs[0])
        #print(dense_retrieved_docs[1])
        #print(dense_retrieved_docs[2])
                
        # BM25(sparse retriever)
        # BM25 에서 검색할 문서는 pinecone 이 찾아낸 k개의 dense_retrieved_docs 를 base 로 한다.
        # 즉, k개의 같은 문서를 가지고 2가지의 retriever 가 각각 순위를 매긴다.
        bm25_retriever = BM25Retriever.from_documents(
            dense_retrieved_docs,
            #metadatas=[{"source": 1}] * len(doc_list_1),
        )
        
        # 위 패키지 임포트 부분에서 설명한 이유로 오리지날을 사용한다.
        # 위의 기본 BM25 를 사용하지 않고 kkma 한글 토크나이저가 적용된 BM25 사용
        #bm25_retriever = KkmaBM25Retriever.from_documents(
        #    dense_retrieved_docs,
            #metadatas=[{"source": 1}] * len(doc_list_1),
        #)
        bm25_retriever.k = nearest_k
        bm25_retrieved_docs = bm25_retriever.invoke(state["question"])
        #print('bm25s choice:')
        #print(bm25_retrieved_docs[0])
        #print(bm25_retrieved_docs[1])
        #print(bm25_retrieved_docs[2])
        
        
        # Ensemble (Pinecone + BM25)
        '''
        # 원래는 아래와 같이 ensemble 하면 간단하게 되지만, 이렇게 하지 않는다.
        # 이유는 dense retriever(vector DB) 를 사용자정의 하여 쓰기 때문에 들어가는 파라미터가 동일하지 않아 문제가 생긴다.
        ensemble_retriever = EnsembleRetriever(
            retrievers = [retriever_with_score, bm25_retriever],
            weights=[0.6, 0.4],
            search_type="similarity"
        )
        ensemble_retriever.invoke()
        '''
        # 그래서 아래처럼 직접 각각 검색된 문서에 가중치를 적용하여 ensemble 시킨다.
        weights = [0.6, 0.4]
        weighted_docs_index = {}
        
        # 일단 vector db 에서 얻은 문서에 가중치를 적용해서 정리해 논다
        for i, doc in enumerate(dense_retrieved_docs):
            # weighted_score = (총개수(만점) - 순위) * 가중치
            weighted_score = (nearest_k - i) * weights[0]
            index = doc.metadata['chunk_index']
            weighted_docs_index[index] = weighted_score
        
        #print(f'weighted_docs_index_1: {weighted_docs_index}')
        
        # 목록 내용과 총개수는 같으니 순위에 따라 가중치를 업데이트 해준다
        for i, doc in enumerate(bm25_retrieved_docs):
            weighted_score = (nearest_k - i) * weights[1]
            index = doc.metadata['chunk_index']
            weighted_docs_index[index] = weighted_docs_index[index] + weighted_score
            #print(f'index:{index}, weighted_score:{weighted_score}')
            
        # 가중치가 적용된 순위로 (내림차순)정렬
        # key 는 정렬 기준이 되는 요소이다. 즉 x[1] 이 기준. x 의 2번째 인 이유는 lamda 를 거치면서 key:value 가 tuple 되고 튜플 중 key는 [0], value는 [1] 이 되기 때문.
        sorted_docs_index = sorted(weighted_docs_index.items(), key=lambda x: x[1], reverse=True)
        #print(f'weighted_docs_index_2: {weighted_docs_index}')
        #print(f'sorted_docs_index: {sorted_docs_index}')
        #print("** ALL **")
        #print(dense_retrieved_docs)
        
        retrieved_doc_list = []     # context 로 사용할 최고 (앙상블)점수 문서와 2,3위 문서를 담을 리스트
        limit_score_diff = 0.2      # pinecone score 가 1 과 limit_score_diff 이상 차이나면 관련 없다고 판단
        for chunk_index, _ in sorted_docs_index:    # sorted_docs_index 는 tuple list 이며, [0]: chunk index, [1]: ensemble score 이다
            if len(retrieved_doc_list) == 3:    # 최대 3개까지만 골라낸다
                break
            else:
                for doc in dense_retrieved_docs:
                    if len(retrieved_doc_list) == 0:    # 최상위 점수의 문서는 무조건 넣는다
                        if str(doc.metadata['chunk_index']) == str(chunk_index):
                            retrieved_doc_list.append(doc)
                            break
                    else:   # 획득 점수 2,3위의 문서를 골라 내기 위한 절차.
                        already_exist = False
                        for doc_in_list in retrieved_doc_list:
                            # 점수 순위상 차례가 되었더라도 같은 판례번호의 문서를 이미 골라냈더라면 다음 문서를 비교
                            if str(doc_in_list.metadata['prec_no']) == str(doc.metadata['prec_no']):
                                already_exist = True
                                break
                            
                        # 같은 판례번호의 문서가 들어있지 않다면 추가하고 다음 점수 순위의 문서 비교
                        # 1 - score 가 limit_score_diff 보다 작아야 관련성 있다고 판단
                        if already_exist == False and (abs(1 - doc.metadata['score']) < limit_score_diff):
                            retrieved_doc_list.append(doc)
                            break
                    
        return GraphState(retrieved_docs=retrieved_doc_list)
        
    
    # 검색되어 선택된 판례문서의 모든 조각을 가져와서 필요한 부분(제일 앞, 제일 끝, 검색된 부분)을 합쳐 다음 노드로 넘긴다
    def merge_retrieved_document(self, state: GraphState) -> GraphState:
        retrieved_docs = state["retrieved_docs"]
        
        if retrieved_docs != None and len(retrieved_docs) > 0:
            top_doc = retrieved_docs[0]
            prec_no = str(top_doc.metadata['prec_no'])
            chunk_index = int(top_doc.metadata['chunk_index'])
            vectorDB_score = float(top_doc.metadata['score'])
            
            etc_prec_numbers = []
            for i in range(1, len(retrieved_docs)):
                etc_prec_numbers.append(str(retrieved_docs[i].metadata['prec_no']))
            
            filters = {
                'prec_no': prec_no,
            }
            
            # 여기서 k 는 한 판례문서가 chunk 로 쪼개진 최대 갯수 이상이여야 한다
            retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"filter": filters, "k":20})
            relevant_docs = retriever.invoke("") # 해당되는 모든 문서를 가져오려면 이렇게 넣어야 한다. 아무것도 안넣으면 에러.
            #print(relevant_docs)
            
            first_of_parts: Document = None
            end_of_parts: Document = None
            head_index = -1
            tail_index = -1
            
            for doc in relevant_docs:
                if head_index == -1:
                    head_index = int(doc.metadata['chunk_index'])   # chunk_index 는 엄청 커질수도 있지만, 파이썬의 int 범위는 무제한이라고 한다.
                    first_of_parts = doc
                elif head_index > int(doc.metadata['chunk_index']):
                    head_index = int(doc.metadata['chunk_index'])
                    first_of_parts = doc
                    
                if tail_index == -1:
                    tail_index = int(doc.metadata['chunk_index'])
                    end_of_parts = doc
                elif tail_index < int(doc.metadata['chunk_index']):
                    tail_index = int(doc.metadata['chunk_index'])
                    end_of_parts = doc
                    
                #print(f"row: {doc.metadata['row']}, Prec_no: {doc.metadata['prec_no']}, Chunk_index: {doc.metadata['chunk_index']}, Source: {doc.metadata['source']}")

            #print(f"Count: {len(relevant_docs)}")
        
            context_doc = ""
            if head_index != chunk_index:
                context_doc = first_of_parts.page_content + "\n" + top_doc.page_content
                
            if (tail_index != head_index) and (tail_index != chunk_index):
                context_doc = context_doc + "\n" + end_of_parts.page_content
            
            #print(f"** context_doc: \n{context_doc}")
            return GraphState(vectordb_score=vectorDB_score, context=context_doc, etc_relevant_precs=etc_prec_numbers)
        
        else:
            return GraphState(vectordb_score=0)
    
    
    # 검색되어 선택된 판례문서의 모든 조각을 가져와서 모두 합쳐 다음 노드로 넘긴다
    def merge_retrieved_all_document(self, state: GraphState) -> GraphState:
        retrieved_docs = state["retrieved_docs"]
        
        if retrieved_docs != None and len(retrieved_docs) > 0:
            top_doc = retrieved_docs[0]
            prec_no = str(top_doc.metadata['prec_no'])
            vectorDB_score = float(top_doc.metadata['score'])
            
            etc_prec_numbers = []
            for i in range(1, len(retrieved_docs)):
                etc_prec_numbers.append(str(retrieved_docs[i].metadata['prec_no']))
            
            filters = {
                'prec_no': prec_no,
            }
            
            # 여기서 k 는 한 판례문서가 chunk 로 쪼개진 최대 갯수 이상이여야 한다
            retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"filter": filters, "k":20})
            relevant_docs = retriever.invoke("") # 해당되는 모든 문서를 가져오려면 이렇게 넣어야 한다. 아무것도 안넣으면 에러.
            
            #print(f"relevant_docs:\n{relevant_docs}")
            #print(f"relevant_docs Count-before: {len(relevant_docs)}")
            sorted_docs = []
            
            force_quit_cnt = 30
            while (len(relevant_docs) > 0) and (force_quit_cnt > 0):
                force_quit_cnt -= 1
                doc = relevant_docs[-1]
                
                if len(sorted_docs) == 0:
                    sorted_docs.append(relevant_docs.pop())
                else:
                    cur_last = sorted_docs[-1]
                    
                    if int(doc.metadata['chunk_index']) > int(cur_last.metadata['chunk_index']):
                        sorted_docs.append(relevant_docs.pop())
                    else:   
                        if len(sorted_docs) == 1:
                                sorted_docs.insert(0, relevant_docs.pop())
                        else: 
                            for i in reversed(range(len(sorted_docs))):
                                if (int(doc.metadata['chunk_index']) < int(sorted_docs[i].metadata['chunk_index'])) and (i == 0):
                                    sorted_docs.insert(0, relevant_docs.pop())                                    
                                    break
                                elif (int(doc.metadata['chunk_index']) < int(sorted_docs[i].metadata['chunk_index'])) and (int(doc.metadata['chunk_index']) > int(sorted_docs[i-1].metadata['chunk_index'])):
                                    sorted_docs.insert(i-1, relevant_docs.pop())                                    
                                    break
                
                '''    
                print(f"row: {doc.metadata['row']}, Prec_no: {doc.metadata['prec_no']}, Chunk_index: {doc.metadata['chunk_index']}, Source: {doc.metadata['source']}")
                print(f"relevant_docs Count-current: {len(relevant_docs)}")
                for temp_doc in relevant_docs:
                    print(f"remain chunk_index: {temp_doc.metadata['chunk_index']}")
                '''

            #print(f"relevant_docs Count-after: {len(relevant_docs)}")
            #print(f"sorted_docs: {sorted_docs}")
        
            context_doc = format_docs(sorted_docs)
            #print(f"** context_doc: \n{context_doc}")
        
            # csv 의 column 대로 dictionary 로 만들어 쓸꺼 쓰고, 뺄꺼 빼고, 나중에 따로 쓸것도 뽑아 낸다.
            column_names = ['prec_no: ', 'case_name: ', 'case_no: ', 'sentence_date: ', 'case_type: ', 'summary: ', 'point: ', 'ref_article: ', 'prec_content: ']
            # 컨텍스트에 넣을 내용은 LLM 이 참고하기 쉽게 컬럼명도 바꿔서 넣어준다. 하지만 진짜 key 를 바꾸는건 아니다.
            column_names_new = {'prec_no':'prec_no: ', 'case_name':'사건명: ', 'case_no':'사건번호: ', 'sentence_date':'sentence_date: ', 'case_type':'사건종류: ', 'summary':'판시사항: ', 'point':'판결요지: ', 'ref_article':'참조조문: ', 'prec_content':'판례전문: '}
            splitted = context_doc.split("\n")
            #print(f"** splited: ")
            content_dict = {}
            content_dict_2 = {}     # 나중에 프론트엔드에서 따로 써먹을 요소만 따로 뽑아 놓는다
            for piece in splitted:
                #print(f"piece:\n{piece}")
                
                for name in column_names:
                    if piece.find(name) != -1:
                        key_name = name[:-2]
                        piece = piece.replace('<br/>', '\n')
                        piece = piece.strip()
                        content_dict[key_name] = piece.replace(name, column_names_new[key_name])
                        
                        if name == 'prec_no: ' or name == 'case_no: ' or name == 'ref_article: ':
                            content_dict_2[key_name] = piece.replace(name, '')
                            
                        break
                
            #print(f"content_dict: \n{content_dict}")
            context_doc = "\n".join([content_dict['case_name'], content_dict['case_no'], content_dict['summary'], content_dict['point'], content_dict['ref_article'], content_dict['prec_content']])
            #print(f"** splited end **")
            
            # 1순위는 본문 전체를, 2/3순위는 판례일련번호만 넘긴다
            return GraphState(vectordb_score=vectorDB_score, context=context_doc, vectordb_choice=content_dict_2, etc_relevant_precs=etc_prec_numbers)
        
        else:
            return GraphState(vectordb_score=0)

    
    def relevance_check(self, state: GraphState) -> GraphState:
        relevance = True
        
        # pinecone score 기준으로 1 에서 차이가 0.15 를 초과하면 웹 서치를 진행
        if abs(1 - state["vectordb_score"]) > 0.15:
            relevance = False
        
        return GraphState(relevance=relevance)


    def is_relevant(self, state: GraphState) -> GraphState:
        return state["relevance"]
    
    
    # 관련성 있는 문서 웹 검색
    def search_on_web(self, state: GraphState) -> GraphState:
        
        # tavily search
        # tavily search 테스트결과 한국어 사이트도 상당히 잘 찾긴 하지만 같은 질문에 대해서도 결과가 일정하지 않게 나올때가 많다.
        # 심할땐 할 때 마다 다르게 검색이 된다. med-info 와 같이 여러 사이트(서울대, 아주대, 아산, MSD)에서 검색을 하는 경우에는 vectorDB 보다 좋을것 같고
        # 판례와 같이 범위가 명확하다면 VectorDB 를 사용하는게 나을 것같다.
        '''
        검색옵션. 참조: https://docs.tavily.com/docs/tavily-api/rest_api
        max_results (optional): The number of maximum search results to return. Default is 5.
        include_domains (optional): A list of domains to specifically include in the search results. Default is None, which includes all domains.
        exclude_domains (optional): A list of domains to specifically exclude from the search results. Default is None, which doesn't exclude any domains.
        '''
        
        include_domains = [] #['https://www.law.go.kr/', 'https://www.scourt.go.kr/'] # 도메인을 특정하여 검색에 포함시키도록 한다.
        '''
        retriever_tavily = TavilySearchAPIRetriever(k=3, include_domains=include_domains, search_depth="advanced", include_raw_content=False)
        relevant_tavily_docs = retriever_tavily.invoke(state["question"])
        print(f"relevant_tavily_docs:\n{relevant_tavily_docs}")
        print()
        search_result = format_docs(relevant_tavily_docs)
        print(f"search_result:\n{search_result}")
        print()
        '''
        
        # 아래에 써있는 이유로 WebBaseLoader 를 사용해서 페이지 전체를 가져왔지만, 이럴필요가 없다.
        # TavilySearchAPIRetriever 의 include_raw_content=True 를 사용하면 내용 전부를 가져올 수 있다.
        '''
        # WebBaseLoader 를 사용해서 tavily search 로 찾은 문서 url 로 부터 웹문서를 가져왔다.
        # 이렇게 한 이유는 tavily search result 의 page_content 가 뒷부분이 잘려서 나오기 때문이다(질문에 관련된 텍스트만 발췌된다).
        # LLM 에 넣을땐 이문서는 이미 retriever 로 검색이 되어져 나온 문서이기때문에 임베딩은 할필요가 없지만, 
        # 문서에 쓸데없는 내용들이 상당량 들어있기 때문에 불필요한 토근이 소모되는 것이 단점이다.
        web_path = relevant_tavily_docs[0].metadata['source']
        # web_paths=(url1, url2) 이런식으로 넣으면 n개의 웹문서를 가져올 수 있다.
        web_loader = WebBaseLoader(web_path=web_path)
        web_docs = web_loader.load()
        print("web_docs:")
        print(web_docs)
        
        # tavily search 와 web loader 를 위한 format_docs
        def format_docs_tavily(relevant_docs):
            searched_doc = ''
            for relevant_doc in relevant_docs:
                web_path = relevant_doc.metadata['source']
                # web_paths=(url1, url2) 이런식으로 넣으면 n개의 웹문서를 가져올 수 있다.
                web_loader = WebBaseLoader(web_path=web_path)
                web_doc = web_loader.load()
                
                print('*** web_doc ***')
                print(web_doc[0])

                web_doc[0].page_content = web_doc[0].page_content.replace("\n", "")
                web_doc[0].page_content = web_doc[0].page_content.replace("\xa0", "")
                
                print('*** web_doc.page_content ***')
                print(web_doc[0].page_content)
                
                # test 로 top-k 중 마지막 문서만 넘긴다. 실제로 사용할땐 제대로 넘겨야 한다.
                searched_doc = web_doc[0].page_content
                
            return searched_doc
            
        search_result = format_docs_tavily(web_docs)
        print(f'searched_text:\n{search_result}')
        '''
        
        # 관련 문서: https://api.python.langchain.com/en/latest/tools/langchain_community.tools.tavily_search.tool.TavilySearchResults.html
        tavily_tool = TavilySearchResults(max_results=5, 
                                        search_depth = "advanced",
                                        include_raw_content = False,    # Include cleaned and parsed HTML of each site search results. Default is False.
                                        include_domains=include_domains, 
                                        # handle_tool_error=True,
                                        # handle_validation_error=True,
                                        # exclude_domains = []
                                        # include_answer = False        # Include a short answer to original query in the search results. Default is False.
                                        # include_images = False
                                        )
        
        # tavily search 의 무료 이용량은 1000건/월 이다. 이용량이 모두 소진되는 등의 문제가 발행하면 LLM 단계로 직행.
        try:
            tavily_result = tavily_tool.invoke({"query": state["question"]})
            search_result = format_searched_docs(tavily_result)
            #print(f'tavily result-1: {tavily_result}')
            #print(f'tavily result-2: {search_result}')
        except Exception as e:
            #print(f'search_on_web - Exception: {e}')
            search_result = ""
        
        return GraphState(context=search_result)
        

    # LLM을 사용하여 답변을 생성
    def llm_answer(self, state: GraphState) -> GraphState:
        question = state["question"]
        context = state["context"]
        #print(f"etc_relevant_precs:{state['etc_relevant_precs']}")
        
        # Track token usage for specific calls.
        # It is currently only implemented for the OpenAI API.
        with get_openai_callback() as cb:
            
            #prompt = hub.pull('rlm/rag-prompt')
           # prompt = ChatPromptTemplate.from_template(
#You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
#"""
#You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Add relevant laws in context as bullet points. Additionally, explain how you will respond. Answer in Korean.
#Question: {question}
#Context: {context}
#Answer:
#"""
                #)

            prompt = ChatPromptTemplate.from_template(get_prompts_by_casetype(self.case_type))
            #print(f"llm_answer:\n{prompt}")
# - 주어진 [Context] 가 답변에 얼마나 연관성이 있는지 0 부터 10 사이의 점수로 표현하고, 답변에 얼마나 참조했는지 0 부터 10 사이의 점수로 표현
# - 웹브라우징을 통해 최근 5년간 한국의 최신 판례와 최근 법률 동향 관련 데이터를 수집한 후 이를 반영하여 최선의 답변 제공
# - 다음 검색된 Context 조각을 검토하고 관련성이 있다면 Context를 기반으로 질문에 답해라. Context가 관련성이 없다면 무시해라. 하지만 답을 모른다면 절대로 지어내지말고 모른다고 답해라.
            
            # ensemble_retriever 에서는 format_docs 를 ensemble 용으로 따로 만들어서 검색문서의 metadata 를 보고 달리 처리를 해야한다.
            # 예를 들어 tavily 같은 경우에는 metadata 안의 'source' 에 'https' 라는 문자열이 있으면 format_docs_tavily 와 같은 처리를 해주면 된다.
            # 아니면 Pinecone 에 문서 embedding 을 할 때 metadata 에 'retriver_type:DB' 라고 넣어주는 것도 좋을것 같다.
            
            """
            {
                #'context': self.retriever,
                #'context': self.retriever | format_docs,
                #'context': retriever | format_docs, # vectorDB 를 사용한 retriever
                'context': retriever_tavily | format_docs_tavily, # tavily search 를 사용한 retriever
                #'context': ensemble_retriever | format_docs, # ensemble retriever. 아직 미완성. tavily 는 format_docs_tavily 를 쓰기 때문에
                'question': RunnablePassthrough()
            } 
            """
            qa_chain = (
                {"question": itemgetter("question"), "context": itemgetter("context")}
                | prompt | self.llm | StrOutputParser()
            )
            response = qa_chain.invoke({"question": question, "context": context})
            
            #print('*** result ***')
            #print('result:', response)
            #print('---' * 20)
            #print(cb)
        
        return GraphState(answer=response)
        
    # -- node 가 될 함수들 (langgraph 에서 node 는 GraphState 를 받고 GraphState 를 넘겨줘야 한다) --
        
    
    # 구현된 node 와 edge 로 workflow 정의
    def build_workflow_rag(self):
        # langgraph.graph에서 StateGraph와 END를 가져옵니다.
        workflow = StateGraph(GraphState)

        # 노드 정의. 확실히 그 이유인지는 모르지만, 추가만 해놓고 안쓰는 노드가 있으면 에러난다.
        workflow.add_node("retrieve", self.retrieve_document)                   # retrieve 검색 노드 추가
        #workflow.add_node("merge_docs", self.merge_retrieved_document)         # 문서 병합(첫,끝,선택부분) 노드 추가
        workflow.add_node("merge_all_docs", self.merge_retrieved_all_document)  # 문서 병합(전체) 노드 추가
        workflow.add_node("relevance_check", self.relevance_check)              # vectorDB 문서의 관련성 체크 노드 추가
        workflow.add_node("search_on_web", self.search_on_web)                  # 웹 검색 노드 추가
        workflow.add_node("llm_answer", self.llm_answer)                        # LLM 노드 추가
        
        
        # 각 노드 연결
        workflow.add_edge("retrieve", "merge_all_docs")  # 검색 -> 병합
        workflow.add_edge("merge_all_docs", "relevance_check")  # 병합 -> 관련성 체크
        
        # 조건부 엣지 추가
        workflow.add_conditional_edges(
            "relevance_check",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
            self.is_relevant,
            {
                True: "llm_answer",     # 관련성이 있으면 LLM 으로 보낸다.
                False: "search_on_web"  # 미달이면 웹 검색
            },
        )
        
        workflow.add_edge("search_on_web", "llm_answer")    # 웹 검색 -> 답변
        workflow.add_edge("llm_answer", END)                # 반드시 마지막으로 'END' 가 와야 한다. 
        

        # 시작 노드 설정
        workflow.set_entry_point("retrieve")

        # Checkpointer: 각 노드간 실행결과 추적하기 위한 메모리(대화에 대한 기록과 유사 개념)
        # 체크포인터를 활용해 특정 시점(Snapshot)으로 되돌리기 기능도 가능
        # 기록 위한 메모리 저장소 설정
        memory = MemorySaver()
        self.app = workflow.compile(checkpointer=memory)
        #app = workflow.compile() # 설정을 안해도 되긴 하는듯
        
        # config 설정 안해주면 에러난다
        # recursion_limit: 최대 노드 실행 개수 지정 (13인 경우: 총 13개의 노드까지 실행). 구현된 노드의 갯수가 아니라 루프를 포함한 전체 실행 노드 개수. 
        # 만약에 5개의 노드가 돌고 돌아 3번씩 실행하게 되면 최소 15개는 되야 한다는 뜻이다.
        # thread_id: 그래프 실행 아이디 기록하고, 추후 추적 목적으로 활용
        self.config = RunnableConfig(
            recursion_limit=12, configurable={"thread_id": "search_precedents"}
        )
        
    
    def run_workflow(self, inputs: GraphState):
        answer = ""
        relevance = False
        vectordb_choice = {}
        etc_relevant_precs = []
        
        # 대화내역 리셋. 자동이긴 하지만 혹시 모르니 가비지 컬렉션도 실행해 주자.
        self.store = {}
        gc.collect()
        
        # app.stream을 통해 입력된 메시지에 대한 출력을 스트리밍합니다.
        try:
            for output in self.app.stream(inputs, config=self.config):
                # 출력된 결과에서 키와 값을 순회합니다.
                for key, value in output.items():
                    # 노드의 이름과 해당 노드에서 나온 출력을 출력합니다.
                    pprint.pprint(f"Output from node '{key}':")
                    #pprint.pprint("---")
                    # 출력 값을 예쁘게 출력합니다.
                    #pprint.pprint(value, indent=2, width=80, depth=None)
                    
                if key == "llm_answer" and "answer" in value:
                    answer = value["answer"]
                    
                if "relevance" in value:
                    relevance = value["relevance"]
                
                if "vectordb_choice" in value:
                    vectordb_choice = value["vectordb_choice"]
                
                if "etc_relevant_precs" in value:
                    etc_relevant_precs = value["etc_relevant_precs"]
                
                # 각 출력 사이에 구분선을 추가합니다.
                #pprint.pprint("\n---\n")
                
        except GraphRecursionError as e:
            #pprint.pprint(f"Recursion limit reached: {e}")
            answer = f"excepttion: {e}"
            
        # relevance, vectordb_choice, etc_relevant_precs 도 넘겨야한다. 그래야 frontend 에서 링크를 걸어줄수 있다. 
        return answer, relevance, vectordb_choice, etc_relevant_precs
        
        
    def question(self, question, filter) -> dict:
        self.case_type = filter
        
        # AgentState 객체를 활용하여 질문을 입력합니다.
        inputs = GraphState(question=question, filter=filter)
        
        #answer = self.run_workflow(inputs)
        answer, relevance, vectordb_choice, etc_relevant_precs = self.run_workflow(inputs)
        #print(f"question!!: {answer}, {relevance}, {vectordb_choice}, {etc_relevant_precs}")
        
        result_dict = {"answer":answer, "relevance":relevance, "vectordb_choice":vectordb_choice, "etc_relevant_precs":etc_relevant_precs}    
        
        return result_dict
        
        
    # 통계 그래프 호출
    # 지표누리(e-나라지표) openAPI 를 사용하여 통계 그래프를 보여준다.
    # 질문+답변 내용을 지표명과 bm25 retriever 로 비교하여 가장 관련있어 보이는 통례를 보여 준다.
    # 사건종류:형사
    def statistics_criminal(self, context) -> List[str]:
        index_list = [
            "1심/2심 무죄 현황", 
            "5대 강력범죄(살인, 강도, 성폭력(강간, 성추행), 방화, 폭행/상해) 현황", 
            "고발사건 처리 현황", 
            "고소사건 처리 현황", 
            "구속영장 청구 발부 현황", 
            "경제범죄사건 처리 현황", 
            "폭력범죄사건 처리 현황", 
            "흉악범죄사건 처리 현황", 
            "교통사범 처리 현황", 
            "소년범죄사건 처리 현황", 
            "환경사범 처리 현황", 
            "피의자 보상금 지급 현황", 
            "형사보상금 지급 현황", 
            "사이버범죄 발생 및 검거 현황", 
            "즉결심판 청구 현황", 
            "개인정보 침해 신고 및 상담"
            ]
        
        '''
        meatadatas = [
            {"source": 0}, 
            {"source": 1}, 
            {"source": 2}, 
            {"source": 3}, 
            {"source": 4}, 
            {"source": 5},
            {"source": 6}, 
            {"source": 7}, 
            {"source": 8}, 
            {"source": 9}, 
            {"source": 10}, 
            {"source": 11}, 
            {"source": 12}, 
            {"source": 13}, 
            {"source": 14}, 
            {"source": 15}
            ]
        '''
        
        bm25_retriever = BM25Retriever.from_texts(          # original
        #bm25_retriever = KkmaBM25Retriever.from_texts(     # custom
            index_list,
            #metadatas = meatadatas
        )
        
        # original
        bm25_retriever.k = 1
        retrieved_result = bm25_retriever.invoke(context)
        
        for i, index_name in enumerate(index_list):
            if retrieved_result[0].page_content == index_name:
                urls = get_index_url(self.index_go_kr_key, i)
                if urls != None and len(urls) > 0:
                    return urls
                break
            
        """
        # custom
        bm25_retriever.k = 3
        retrieved_result = bm25_retriever.search_with_score(context)
        pretty_print(retrieved_result)
        print()
        print(bm25_retriever.invoke(context))
        
        for i, index_name in enumerate(index_list):
            if retrieved_result[0].page_content == index_name:
                urls = get_index_url(self.index_go_kr_key, i)
                # cutoff_score 이상이면 보여주자. cutoff_score 이하는 너무 관련이 없는것 같아서..
                score = float(retrieved_result[0].metadata['score'])
                cutoff_score = 0.01
                #print(f"score: {score}")
                if urls != None and len(urls) > 0 and score >= cutoff_score:
                    return urls
                break
        """
    
    
    # 사건종류:행정 일때 보여줄 통계 데이터
    def statistics_administrative(self) -> List[str]:
        return get_index_url(self.index_go_kr_key, 16)
    
    
    # --- 법률 조언 ---
    def llm_answer_advice(self, state: GraphState) -> GraphState:
        post_conversation = state["post_conversation"]
        paper_content = state["paper_content"]
        paper_content["post_conversation"] = post_conversation
        response = None
        session_id = "법률조언"
        
        with get_openai_callback() as cb:
            # context 로 넘기기위해 입력받은 정보를 합쳐 하나의 문자열로 만든다.
            def integrate_input_info(input_info):
                input_info_list = []
                input_info_list.append(f"의뢰인의 상황: {input_info['status']}")
                input_info_list.append(f"의뢰인의 요구 혹은 질문: {input_info['question']}")
                
                # 기한이 중요할 수도 있기 때문에 오늘 날짜를 넣어준다.
                now = datetime.datetime.now()
                today = now.strftime("%Y년 %m월 %d일")
                input_info_list.append(f"오늘 날짜: {today}")
                
                # 추가 대화이면 정보도 추가된다.
                if input_info['post_conversation'] == True and input_info.get('add_info'):           
                    input_info_list.append(f"추가 질문에 대한 정보: {input_info['add_info']}")
                
                integrated_info = "\n".join(input_info_list)
                #print(f"user_input_info: \n{integrated_info}")
                
                return integrated_info
            
            
            # 대화 히스토리 저장
            # 세션 ID를 기반으로 세션 기록을 가져오는 함수
            def get_session_history(session_ids):
                #print(f"[대화 세션ID]: {session_ids}")
                if session_ids not in self.store:                    # 세션 ID가 store에 없는 경우                    
                    self.store[session_ids] = ChatMessageHistory()   # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
                return self.store[session_ids]                       # 해당 세션 ID에 대한 세션 기록 반환
            
            
            # 첫 대화
            if post_conversation == False:
                prompt = PromptTemplate.from_template(get_prompts_by_casetype("법률조언_1"))
                #print(f"llm_answer_advice-first:\n{prompt}")
            
                qa_chain = (
                    {
                        "context": itemgetter("paper_content") | RunnableLambda(integrate_input_info),
                        "question": itemgetter("question"),
                        "chat_history": itemgetter("chat_history"),
                    }
                    | prompt | self.llm | StrOutputParser())

                chain_with_history = RunnableWithMessageHistory(
                    qa_chain,
                    get_session_history,                    # 세션 기록을 가져오는 함수
                    input_messages_key="question",          # 사용자의 질문이 템플릿 변수에 들어갈 key
                    history_messages_key="chat_history",    # 기록 메시지의 키
                )
                        
                response = chain_with_history.invoke(
                    {"question": "", "paper_content": paper_content},   # 질문, 유저에게 입력받은 정보
                    config={"configurable": {"session_id": session_id}}   # 세션 ID 기준으로 대화를 기록
                    )
            
            else:   # 추가 대화
                prompt = PromptTemplate.from_template(get_prompts_by_casetype("법률조언_2"))
                #print(f"llm_answer_advice-post:\n{prompt}")
                
                qa_chain = (
                    {
                        "context": itemgetter("paper_content") | RunnableLambda(integrate_input_info),
                        "question": itemgetter("question"),
                        "chat_history": itemgetter("chat_history"),
                    }
                    | prompt | self.llm | StrOutputParser())
            
                chain_with_history = RunnableWithMessageHistory(
                    qa_chain,
                    get_session_history,                    # 세션 기록을 가져오는 함수
                    input_messages_key="question",          # 사용자의 질문이 템플릿 변수에 들어갈 key
                    history_messages_key="chat_history",    # 기록 메시지의 키
                )
                
                response = chain_with_history.invoke(
                    {"question": "", "paper_content": paper_content},   # 질문, 유저에게 입력받은 정보
                    config={"configurable": {"session_id": session_id}}   # 세션 ID 기준으로 대화를 기록
                    )
                
                # 추가 대화까지 완료되었다면 대화기록을 삭제한다. 그래야 다음에 쓸데없이 불러오지않지..
                del(self.store[session_id])
                self.store = {}
                
            
            #print('result:', response)
            #print('---' * 20)
            #print(cb)
        
        return GraphState(answer=response)
    
    
    def build_workflow_advice(self):
        workflow = StateGraph(GraphState)

        # 노드 추가
        workflow.add_node("llm_answer", self.llm_answer_advice)
        
        # 노드 연결
        workflow.add_edge("llm_answer", END)
        
        # 시작 노드
        workflow.set_entry_point("llm_answer")

        memory = MemorySaver()
        self.app = workflow.compile(checkpointer=memory)
        
        self.config = RunnableConfig(
            recursion_limit=12, configurable={"thread_id": "search_precedents"}
        )
        
    
    def advice(self, is_post_conversation, status, question, add_info) -> dict:
        paper_content = {"status": status, "question": question, "add_info": add_info}
        inputs = GraphState(paper_content=paper_content, post_conversation=is_post_conversation)
        
        answer, _, _, _ = self.run_workflow(inputs)
        
        result_dict = {"answer":answer}
        
        return result_dict
    
    
    # --- 서류작성 - 내용증명 ---
    def llm_answer_write_paper_1(self, state: GraphState) -> GraphState:
        paper_content = state["paper_content"]
        
        reason = paper_content["reason"]
        fact = paper_content["fact"]
        ask = paper_content["ask"]
        point = paper_content["point"]
        receiver = paper_content["receiver"]
        sender = paper_content["sender"]
        phone = paper_content["phone"]
        appendix = paper_content["appendix"]
        style = paper_content["style"]
        now = datetime.datetime.now()
        today = now.strftime("%Y년 %m월 %d일")
        
        with get_openai_callback() as cb:
            prompt = ChatPromptTemplate.from_template(get_prompts_by_casetype("내용증명"))
            #print(f"llm_answer_write_paper_1:\n{prompt}")
            
            qa_chain = (
                {"reason": itemgetter("reason"), "fact": itemgetter("fact"), "ask": itemgetter("ask"), "point": itemgetter("point"), "receiver": itemgetter("receiver"), "sender": itemgetter("sender"), "phone": itemgetter("phone"), "appendix": itemgetter("appendix"), "style": itemgetter("style"), "today": itemgetter("today")}
                | prompt | self.llm | StrOutputParser()
            )
            response = qa_chain.invoke({"reason": reason, "fact": fact, "ask": ask, "point": point, "receiver": receiver, "sender": sender, "phone": phone, "appendix": appendix, "style": style, "today": today})
            
            #print('*** result ***')
            #print('result:', response)
            #print('---' * 20)
            #print(cb)
        
        return GraphState(answer=response)
    
    
    def build_workflow_write_paper_1(self):
        workflow = StateGraph(GraphState)

        # 노드 추가
        workflow.add_node("llm_answer", self.llm_answer_write_paper_1)
        
        # 노드 연결
        workflow.add_edge("llm_answer", END)
        
        # 시작 노드
        workflow.set_entry_point("llm_answer")

        memory = MemorySaver()
        self.app = workflow.compile(checkpointer=memory)
        
        self.config = RunnableConfig(
            recursion_limit=12, configurable={"thread_id": "search_precedents"}
        )
        
        
    def write_paper_1(self, reason, fact, ask, point, receiver, sender, phone, appendix, style) -> dict:
        paper_content = {"reason": reason, "fact": fact, "ask": ask, "point": point, "receiver": receiver, "sender": sender, "phone": phone, "appendix":appendix, "style": style}
        inputs = GraphState(paper_content=paper_content)
        
        answer, _, _, _ = self.run_workflow(inputs)
        
        result_dict = {"answer":answer}
        
        return result_dict
    
    
    # --- 서류작성 - 지급명령신청서 ---
    def llm_answer_write_paper_2(self, state: GraphState) -> GraphState:
        paper_content = state["paper_content"]
        
        sender_name = paper_content["sender_name"]
        #sender_addr = paper_content["sender_addr"]
        #sender_phone = paper_content["sender_phone"]
        
        receiver_name = paper_content["receiver_name"]
        #receiver_addr = paper_content["receiver_addr"]
        court = paper_content["court"]
        
        amount = paper_content["amount"]
        ask_interest = paper_content["ask_interest"]
        transmittal_fee = paper_content["transmittal_fee"]
        stamp_fee = paper_content["stamp_fee"]
        
        ask_reason = paper_content["ask_reason"]
        ask_reason_detail = paper_content["ask_reason_detail"]
        appendix = paper_content["appendix"]
        
        now = datetime.datetime.now()
        today = now.strftime("%Y년 %m월 %d일")
        
        with get_openai_callback() as cb:
            prompt = ChatPromptTemplate.from_template(get_prompts_by_casetype("지급명령신청서"))
            #print(f"llm_answer_write_paper_2:\n{prompt}")
            
            qa_chain = (
                #{"sender_name": itemgetter("sender_name"), "sender_addr": itemgetter("sender_addr"), "sender_phone": itemgetter("sender_phone"), \
                #"receiver_name": itemgetter("receiver_name"), "receiver_addr": itemgetter("receiver_addr"), "court": itemgetter("court"), \
                {"sender_name": itemgetter("sender_name"), \
                "receiver_name": itemgetter("receiver_name"), "court": itemgetter("court"), \
                "amount": itemgetter("amount"), "ask_interest": itemgetter("ask_interest"), "transmittal_fee": itemgetter("transmittal_fee"), "stamp_fee": itemgetter("stamp_fee"), \
                "ask_reason": itemgetter("ask_reason"), "ask_reason_detail": itemgetter("ask_reason_detail"), "appendix": itemgetter("appendix"), "today": itemgetter("today")}
                | prompt | self.llm | StrOutputParser()
            )
            #response = qa_chain.invoke({"sender_name": sender_name, "sender_addr": sender_addr, "sender_phone": sender_phone, \
            #    "receiver_name": receiver_name, "receiver_addr": receiver_addr, "court": court, \
            response = qa_chain.invoke({"sender_name": sender_name, \
                "receiver_name": receiver_name, "court": court, \
                "amount": amount, "ask_interest": ask_interest, "transmittal_fee": transmittal_fee, "stamp_fee": stamp_fee, \
                "ask_reason": ask_reason, "ask_reason_detail": ask_reason_detail, "appendix": appendix, "today": today})
            
            #print('*** result ***')
            #print('result:', response)
            #print('---' * 20)
            #print(cb)
        
        return GraphState(answer=response)
    
    
    def build_workflow_write_paper_2(self):
        workflow = StateGraph(GraphState)

        # 노드 추가
        workflow.add_node("llm_answer", self.llm_answer_write_paper_2)
        
        # 노드 연결
        workflow.add_edge("llm_answer", END)
        
        # 시작 노드
        workflow.set_entry_point("llm_answer")

        memory = MemorySaver()
        self.app = workflow.compile(checkpointer=memory)
        
        self.config = RunnableConfig(
            recursion_limit=12, configurable={"thread_id": "search_precedents"}
        )
        
    
    def write_paper_2(self, \
        #sender_name, sender_addr, sender_phone, \
        #receiver_name, receiver_addr, court, \
        sender_name, \
        receiver_name, court, \
        amount, ask_interest, transmittal_fee, stamp_fee, \
        ask_reason, ask_reason_detail, appendix) -> dict:
        
        #paper_content = {"sender_name": sender_name, "sender_addr": sender_addr, "sender_phone": sender_phone, \
        #    "receiver_name": receiver_name, "receiver_addr": receiver_addr, "court": court, \
        paper_content = {"sender_name": sender_name, \
            "receiver_name": receiver_name, "court": court, \
            "amount": amount, "ask_interest": ask_interest, "transmittal_fee": transmittal_fee, "stamp_fee": stamp_fee, \
            "ask_reason": ask_reason, "ask_reason_detail": ask_reason_detail, "appendix": appendix}
        inputs = GraphState(paper_content=paper_content)
        
        answer, _, _, _ = self.run_workflow(inputs)
        
        result_dict = {"answer":answer}
        
        return result_dict
    
    
    # --- 서류작성 - 답변서 ---
    def llm_answer_write_paper_4(self, state: GraphState) -> GraphState:
        post_conversation = state["post_conversation"]
        paper_content = state["paper_content"]
        paper_content["post_conversation"] = post_conversation
        response = None
        session_id = "답변서"
        
        with get_openai_callback() as cb:
            # context 로 넘기기위해 입력받은 정보를 합쳐 하나의 문자열로 만든다.
            def integrate_input_info(input_info):
                
                now = datetime.datetime.now()
                today = now.strftime("%Y년 %m월 %d일")
                
                input_info_list = []
                input_info_list.append(f"원고 이름: {input_info['sender_name']}")
                input_info_list.append(f"피고 이름: {input_info['receiver_name']}")
                
                input_info_list.append(f"사건번호: {input_info['case_no']}")
                input_info_list.append(f"소장내용 중 소를 제기하는 이유: {input_info['case_name']}")
                input_info_list.append(f"소장내용 중 청구 취지: {input_info['case_purpose']}")
                input_info_list.append(f"소장내용 중 청구 원인: {input_info['case_cause']}")
                input_info_list.append(f"소장내용 중 입증 방법: {input_info['case_prove']}")
                input_info_list.append(f"소장내용 중 첨부 서류: {input_info['case_appendix']}")
                input_info_list.append(f"관할 법원: {input_info['case_court']}")
                
                input_info_list.append(f"청구 원인에 대한 반박: {input_info['rebut']}")
                input_info_list.append(f"답변서에 첨부할 첨부 서류: {input_info['appendix']}")
                input_info_list.append(f"답변서 제출일: {today}")
                
                # 추가 대화이면 정보도 추가된다.
                if input_info['post_conversation'] == True and input_info.get('add_info'):           
                    input_info_list.append(f"청구 원인에 대한 추가 반박: {input_info['add_info']}")
                
                integrated_info = "\n".join(input_info_list)
                #print(f"user_input_info: \n{integrated_info}")
                
                return integrated_info
            
            
            # 대화 히스토리 저장
            # 세션 ID를 기반으로 세션 기록을 가져오는 함수
            def get_session_history(session_ids):
                #print(f"[대화 세션ID]: {session_ids}")
                if session_ids not in self.store:                    # 세션 ID가 store에 없는 경우                    
                    self.store[session_ids] = ChatMessageHistory()   # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
                return self.store[session_ids]                       # 해당 세션 ID에 대한 세션 기록 반환
            
            
            # 첫 대화
            if post_conversation == False:
                prompt = PromptTemplate.from_template(get_prompts_by_casetype("답변서_1"))
                #print(f"llm_answer_write_paper_4-first:\n{prompt}")
            
                qa_chain = (
                    {
                        "context": itemgetter("paper_content") | RunnableLambda(integrate_input_info),
                        "question": itemgetter("question"),
                        "chat_history": itemgetter("chat_history"),
                    }
                    | prompt | self.llm | StrOutputParser())

                chain_with_history = RunnableWithMessageHistory(
                    qa_chain,
                    get_session_history,                    # 세션 기록을 가져오는 함수
                    input_messages_key="question",          # 사용자의 질문이 템플릿 변수에 들어갈 key
                    history_messages_key="chat_history",    # 기록 메시지의 키
                )
                        
                response = chain_with_history.invoke(
                    {"question": "", "paper_content": paper_content},   # 질문, 유저에게 입력받은 정보
                    config={"configurable": {"session_id": session_id}}   # 세션 ID 기준으로 대화를 기록
                    )
            
            else:   # 추가 대화
                prompt = PromptTemplate.from_template(get_prompts_by_casetype("답변서_2"))
                #print(f"llm_answer_write_paper_4-post:\n{prompt}")
                
                qa_chain = (
                    {
                        "context": itemgetter("paper_content") | RunnableLambda(integrate_input_info),
                        "question": itemgetter("question"),
                        "chat_history": itemgetter("chat_history"),
                    }
                    | prompt | self.llm | StrOutputParser())
            
                chain_with_history = RunnableWithMessageHistory(
                    qa_chain,
                    get_session_history,                    # 세션 기록을 가져오는 함수
                    input_messages_key="question",          # 사용자의 질문이 템플릿 변수에 들어갈 key
                    history_messages_key="chat_history",    # 기록 메시지의 키
                )
                
                response = chain_with_history.invoke(
                    {"question": "", "paper_content": paper_content},   # 질문, 유저에게 입력받은 정보
                    config={"configurable": {"session_id": session_id}}   # 세션 ID 기준으로 대화를 기록
                    )
                
                # 추가 대화까지 완료되었다면 대화기록을 삭제한다. 그래야 다음에 쓸데없이 불러오지않지..
                del(self.store[session_id])
                self.store = {}
                
            
            #print('result:', response)
            #print('---' * 20)
            #print(cb)
        
        return GraphState(answer=response)
    
    
    def build_workflow_write_paper_4(self):
        workflow = StateGraph(GraphState)

        # 노드 추가
        workflow.add_node("llm_answer", self.llm_answer_write_paper_4)
        
        # 노드 연결
        workflow.add_edge("llm_answer", END)
        
        # 시작 노드
        workflow.set_entry_point("llm_answer")

        memory = MemorySaver()
        self.app = workflow.compile(checkpointer=memory)
        
        self.config = RunnableConfig(
            recursion_limit=12, configurable={"thread_id": "search_precedents"}
        )
        
    
    def write_paper_4(self, is_post_conversation, \
        sender_name, receiver_name, \
        case_no, case_name, case_purpose, case_cause, case_prove, case_appendix, case_court, \
        rebut, appendix, add_info) -> dict:
        
        paper_content = {"sender_name": sender_name, "receiver_name": receiver_name, \
            "case_no": case_no, "case_name": case_name, "case_purpose": case_purpose, "case_cause": case_cause, "case_prove": case_prove, "case_appendix": case_appendix, "case_court": case_court, \
            "rebut": rebut, "appendix": appendix, "add_info": add_info}
        inputs = GraphState(paper_content=paper_content, post_conversation=is_post_conversation)
        
        answer, _, _, _ = self.run_workflow(inputs)
        
        result_dict = {"answer":answer}
        
        return result_dict


    # --- 서류작성 - 고소장 ---
    def llm_answer_write_paper_5(self, state: GraphState) -> GraphState:
        post_conversation = state["post_conversation"]
        paper_content = state["paper_content"]
        paper_content["post_conversation"] = post_conversation
        response = None
        session_id = "고소장"
        
        with get_openai_callback() as cb:
            # context 로 넘기기위해 입력받은 정보를 합쳐 하나의 문자열로 만든다.
            def integrate_input_info(input_info):
                now = datetime.datetime.now()
                today = now.strftime("%Y년 %m월 %d일")
                
                input_info_list = []
                input_info_list.append(f"고소인 이름: {input_info['sender_name']}")
                input_info_list.append(f"피고소인 이름: {input_info['receiver_name']}")
                input_info_list.append(f"피고소인 관련 기타 사항: {input_info['receiver_etc']}")
                
                input_info_list.append(f"고발할 범죄항목: {input_info['purpose']}")
                input_info_list.append(f"사건 발생 일시 및 장소: {input_info['crime_time']}")
                input_info_list.append(f"사건 경위: {input_info['crime_history']}")
                input_info_list.append(f"피해 사실: {input_info['damage']}")
                input_info_list.append(f"고소하는 이유와 고소를 결심한 이유: {input_info['reason']}")
                input_info_list.append(f"증거 자료: {input_info['evidence']}")
                input_info_list.append(f"관련 사건의 수사 및 재판 여부: {input_info['etc_accuse']}")
                input_info_list.append(f"고소장 제출일: {today}")
                input_info_list.append(f"관할 경찰서: {input_info['station']}")
                
                # 추가 대화이면 정보도 추가된다.
                if input_info['post_conversation'] == True and input_info.get('add_info'):           
                    input_info_list.append(f"추가 질문에 대한 정보: {input_info['add_info']}")
                
                integrated_info = "\n".join(input_info_list)
                #print(f"user_input_info: \n{integrated_info}")
                
                return integrated_info
            
            
            # 대화 히스토리 저장
            # 세션 ID를 기반으로 세션 기록을 가져오는 함수
            def get_session_history(session_ids):
                #print(f"[대화 세션ID]: {session_ids}")
                if session_ids not in self.store:                    # 세션 ID가 store에 없는 경우                    
                    self.store[session_ids] = ChatMessageHistory()   # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
                return self.store[session_ids]                       # 해당 세션 ID에 대한 세션 기록 반환
            
            
            # 첫 대화
            if post_conversation == False:
                prompt = PromptTemplate.from_template(get_prompts_by_casetype("고소장_1"))
                #print(f"llm_answer_write_paper_5-first:\n{prompt}")
            
                qa_chain = (
                    {
                        "context": itemgetter("paper_content") | RunnableLambda(integrate_input_info),
                        "question": itemgetter("question"),
                        "chat_history": itemgetter("chat_history"),
                    }
                    | prompt | self.llm | StrOutputParser())

                chain_with_history = RunnableWithMessageHistory(
                    qa_chain,
                    get_session_history,                    # 세션 기록을 가져오는 함수
                    input_messages_key="question",          # 사용자의 질문이 템플릿 변수에 들어갈 key
                    history_messages_key="chat_history",    # 기록 메시지의 키
                )
                        
                response = chain_with_history.invoke(
                    {"question": "", "paper_content": paper_content},   # 질문, 유저에게 입력받은 정보
                    config={"configurable": {"session_id": session_id}}   # 세션 ID 기준으로 대화를 기록
                    )
            
            else:   # 추가 대화
                prompt = PromptTemplate.from_template(get_prompts_by_casetype("고소장_2"))
                #print(f"llm_answer_write_paper_5-post:\n{prompt}")
                
                qa_chain = (
                    {
                        "context": itemgetter("paper_content") | RunnableLambda(integrate_input_info),
                        "question": itemgetter("question"),
                        "chat_history": itemgetter("chat_history"),
                    }
                    | prompt | self.llm | StrOutputParser())
            
                chain_with_history = RunnableWithMessageHistory(
                    qa_chain,
                    get_session_history,                    # 세션 기록을 가져오는 함수
                    input_messages_key="question",          # 사용자의 질문이 템플릿 변수에 들어갈 key
                    history_messages_key="chat_history",    # 기록 메시지의 키
                )
                
                response = chain_with_history.invoke(
                    {"question": "", "paper_content": paper_content},   # 질문, 유저에게 입력받은 정보
                    config={"configurable": {"session_id": session_id}}   # 세션 ID 기준으로 대화를 기록
                    )
                
                # 추가 대화까지 완료되었다면 대화기록을 삭제한다. 그래야 다음에 쓸데없이 불러오지않지..
                del(self.store[session_id])
                self.store = {}
                
            
            #print('result:', response)
            #print('---' * 20)
            #print(cb)
        
        return GraphState(answer=response)
    
    
    def build_workflow_write_paper_5(self):
        workflow = StateGraph(GraphState)

        # 노드 추가
        workflow.add_node("llm_answer", self.llm_answer_write_paper_5)
        
        # 노드 연결
        workflow.add_edge("llm_answer", END)
        
        # 시작 노드
        workflow.set_entry_point("llm_answer")

        memory = MemorySaver()
        self.app = workflow.compile(checkpointer=memory)
        
        self.config = RunnableConfig(
            recursion_limit=12, configurable={"thread_id": "search_precedents"}
        )
        
    
    def write_paper_5(self, is_post_conversation, \
        sender_name, receiver_name, \
        receiver_etc, purpose, crime_time, crime_history, damage, reason, evidence, \
        etc_accuse, station, add_info) -> dict:
        
        paper_content = {"sender_name": sender_name, "receiver_name": receiver_name, \
            "receiver_etc": receiver_etc, "purpose": purpose, "crime_time": crime_time, "crime_history": crime_history, "damage": damage, "reason": reason, "evidence": evidence, \
            "etc_accuse": etc_accuse, "station": station, "add_info": add_info}
        inputs = GraphState(paper_content=paper_content, post_conversation=is_post_conversation)
        
        answer, _, _, _ = self.run_workflow(inputs)
        
        result_dict = {"answer":answer}
        
        return result_dict
    
    
    # --- 서류작성 - (민사)소장 ---
    def llm_answer_write_paper_6(self, state: GraphState) -> GraphState:
        post_conversation = state["post_conversation"]
        paper_content = state["paper_content"]
        paper_content["post_conversation"] = post_conversation
        response = None
        session_id = "민사소장"
        
        with get_openai_callback() as cb:
            # context 로 넘기기위해 입력받은 정보를 합쳐 하나의 문자열로 만든다.
            def integrate_input_info(input_info):
                now = datetime.datetime.now()
                today = now.strftime("%Y년 %m월 %d일")
                
                input_info_list = []
                input_info_list.append(f"원고 이름: {input_info['sender_name']}")
                input_info_list.append(f"피고 이름: {input_info['receiver_name']}")
                
                input_info_list.append(f"사건명: {input_info['case_name']}")
                input_info_list.append(f"청구 취지: {input_info['purpose']}")
                input_info_list.append(f"청구 원인: {input_info['reason']}")
                input_info_list.append(f"입증 방법 및 증거: {input_info['evidence']}")                
                input_info_list.append(f"제출일: {today}")
                input_info_list.append(f"관할 법원: {input_info['court']}")
                
                # 추가 대화이면 정보도 추가된다.
                if input_info['post_conversation'] == True and input_info.get('add_info'):           
                    input_info_list.append(f"추가 질문에 대한 정보: {input_info['add_info']}")
                
                integrated_info = "\n".join(input_info_list)
                #print(f"user_input_info: \n{integrated_info}")
                
                return integrated_info
            
            
            # 대화 히스토리 저장
            # 세션 ID를 기반으로 세션 기록을 가져오는 함수
            def get_session_history(session_ids):
                #print(f"[대화 세션ID]: {session_ids}")
                if session_ids not in self.store:                    # 세션 ID가 store에 없는 경우                    
                    self.store[session_ids] = ChatMessageHistory()   # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
                return self.store[session_ids]                       # 해당 세션 ID에 대한 세션 기록 반환
            
            
            # 첫 대화
            if post_conversation == False:
                prompt = PromptTemplate.from_template(get_prompts_by_casetype("민사소장_1"))
                #print(f"llm_answer_write_paper_6-first:\n{prompt}")
            
                qa_chain = (
                    {
                        "context": itemgetter("paper_content") | RunnableLambda(integrate_input_info),
                        "question": itemgetter("question"),
                        "chat_history": itemgetter("chat_history"),
                    }
                    | prompt | self.llm | StrOutputParser())

                chain_with_history = RunnableWithMessageHistory(
                    qa_chain,
                    get_session_history,                    # 세션 기록을 가져오는 함수
                    input_messages_key="question",          # 사용자의 질문이 템플릿 변수에 들어갈 key
                    history_messages_key="chat_history",    # 기록 메시지의 키
                )
                        
                response = chain_with_history.invoke(
                    {"question": "", "paper_content": paper_content},   # 질문, 유저에게 입력받은 정보
                    config={"configurable": {"session_id": session_id}}   # 세션 ID 기준으로 대화를 기록
                    )
            
            else:   # 추가 대화
                prompt = PromptTemplate.from_template(get_prompts_by_casetype("민사소장_2"))
                #print(f"llm_answer_write_paper_6-post:\n{prompt}")
                
                qa_chain = (
                    {
                        "context": itemgetter("paper_content") | RunnableLambda(integrate_input_info),
                        "question": itemgetter("question"),
                        "chat_history": itemgetter("chat_history"),
                    }
                    | prompt | self.llm | StrOutputParser())
            
                chain_with_history = RunnableWithMessageHistory(
                    qa_chain,
                    get_session_history,                    # 세션 기록을 가져오는 함수
                    input_messages_key="question",          # 사용자의 질문이 템플릿 변수에 들어갈 key
                    history_messages_key="chat_history",    # 기록 메시지의 키
                )
                
                response = chain_with_history.invoke(
                    {"question": "", "paper_content": paper_content},   # 질문, 유저에게 입력받은 정보
                    config={"configurable": {"session_id": session_id}}   # 세션 ID 기준으로 대화를 기록
                    )
                
                # 추가 대화까지 완료되었다면 대화기록을 삭제한다. 그래야 다음에 쓸데없이 불러오지않지..
                del(self.store[session_id])
                self.store = {}
                
            
            #print('result:', response)
            #print('---' * 20)
            #print(cb)
        
        return GraphState(answer=response)
    
    
    def build_workflow_write_paper_6(self):
        workflow = StateGraph(GraphState)

        # 노드 추가
        workflow.add_node("llm_answer", self.llm_answer_write_paper_6)
        
        # 노드 연결
        workflow.add_edge("llm_answer", END)
        
        # 시작 노드
        workflow.set_entry_point("llm_answer")

        memory = MemorySaver()
        self.app = workflow.compile(checkpointer=memory)
        
        self.config = RunnableConfig(
            recursion_limit=12, configurable={"thread_id": "search_precedents"}
        )
        
    
    def write_paper_6(self, is_post_conversation, \
        sender_name, receiver_name, \
        case_name, purpose, reason, evidence, \
        court, add_info) -> dict:
        
        paper_content = {"sender_name": sender_name, "receiver_name": receiver_name, \
            "case_name": case_name, "purpose": purpose, "reason": reason, "evidence": evidence, \
            "court": court, "add_info": add_info}
        inputs = GraphState(paper_content=paper_content, post_conversation=is_post_conversation)
        
        answer, _, _, _ = self.run_workflow(inputs)
        
        result_dict = {"answer":answer}
        
        return result_dict
    
    