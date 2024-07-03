import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import faiss
import re
from typing import List
from langchain.docstore.document import Document
import pickle
import os
import json
import openai
import requests

def split_text_by_regulation(documents: List[Document]) -> List[Document]:
  """
  문서 리스트를 규정 제목과 조항을 기준으로 분할하고 Document 객체 리스트를 반환합니다.

  Args:
      documents (List[Document]): 분할할 Document 객체 리스트

  Returns:
      List[Document]: 분할된 Document 객체 리스트
  """
  pattern = r'제\s*\d+[조|항]\s*【.*】'  # 규정 제목과 조항 패턴
  # 첫 번째 문서의 텍스트 내용 추출 (여러 문서가 있는 경우 반복문으로 처리)
  text = documents[0].page_content
  chunks = re.split(pattern, text)
  chunks = [chunk.strip() for chunk in chunks if chunk.strip()]  # 빈 문자열 제거

  docs = []
  current_regulation = ""
  for chunk in chunks:
    if re.match(pattern, chunk):  # 규정 제목인 경우
      current_regulation = chunk  # 현재 규정 제목 업데이트
    else:  # 조항 내용인 경우
      metadata = {"regulation": current_regulation}  # 메타데이터에 규정 제목 추가
      doc = Document(page_content=chunk, metadata=metadata)  # Document 객체 생성
      docs.append(doc)  # 리스트에 추가

  return docs

def get_embeddings(docs, embedding_model, embedding_dir="./data", embedding_file_name="embeddings.pkl"):
  """
  문서 임베딩을 계산하고 저장합니다.

  Args:
      docs (List[Document]): Document 객체 리스트
      embedding_model (SentenceTransformer): 임베딩 모델
      embedding_dir (str): 임베딩 저장 디렉토리 (기본값: "./data")
      embedding_file_name (str): 임베딩 저장 파일 이름 (기본값: "embeddings.pkl")

  Returns:
      numpy.ndarray: 문서 임베딩
  """

  # 임베딩 저장 디렉토리 생성 (존재하지 않는 경우)
  os.makedirs(embedding_dir, exist_ok=True)

  embedding_file_path = os.path.join(embedding_dir, embedding_file_name)

  try:
    # 임베딩 파일 로드 시도
    with open(embedding_file_path, "rb") as f:
      embeddings = pickle.load(f)
    print("저장된 임베딩을 로드했습니다.")
  except (FileNotFoundError, EOFError):
    # 임베딩 파일이 없거나 손상된 경우 새로 계산
    print("임베딩 파일을 찾을 수 없거나 손상되었습니다. 새로 임베딩을 계산합니다.")
    embeddings = embedding_model.encode([doc.page_content for doc in docs])
    # 임베딩 파일 저장
    with open(embedding_file_path, "wb") as f:
      pickle.dump(embeddings, f)
    print("임베딩을 계산하고 저장했습니다.")

  return embeddings


# --- 1. 문서 로드 및 임베딩 생성 ---
loader = TextLoader("manual.txt")
documents = loader.load()


# RecursiveCharacterTextSplitter는 텍스트를 의미론적 단위로 분할하는 데 사용되는 도구
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

model = SentenceTransformer('jhgan/ko-/sroberta-multitask')
# model = SentenceTransformer('klue/roberta-base') 작동안됨
# model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# embeddings = model.encode([doc.page_content for doc in docs])

# 모델 및 임베딩 파일 경로 설정
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
embedding_dir = "./data"  # 임베딩 저장 디렉토리
embedding_file_name = "embeddings_ko-sroberta-multitask.pkl"  # 임베딩 저장 파일 이름

# 임베딩 계산 및 저장 (또는 로드)
embeddings = get_embeddings(docs, model, embedding_dir, embedding_file_name)
# --- 2. FAISS 인덱스 생성 ---
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# --- 3. FastAPI 앱 생성 ---
app = FastAPI(
    title="회사 매뉴얼 챗봇",
    description="회사 매뉴얼을 검색하는 챗봇 서비스입니다.",
    version="0.1.0"
)
# CORS 설정
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

class Query(BaseModel):
    message: str

class SearchResult(BaseModel):
    content: str
    context: str
    metadata: dict
    distance: float

class SearchResult_New(BaseModel):
    content: str
    
    def get_reference_str(self) -> str:
        return self.content
    
class ChatResponse(BaseModel):
    question: str
    reference : str
    answer: str

class ChatOnlyResponse(BaseModel):
    question: str    
    answer: str

# --- 4. 검색 함수 정의 ---
def search_manual(query, k=10, context_window=100):
    try:
        query_embedding = model.encode([query])
        D, I = index.search(query_embedding, k)

        results = []
        for i, distance in zip(I[0], D[0]):
            metadata = {"source": documents[0].metadata["source"]}  # 파일 이름을 메타데이터로 사용
            if hasattr(docs[i], "metadata") and "chunk" in docs[i].metadata:
                metadata["chunk"] = docs[i].metadata["chunk"]     # 청크 번호를 메타데이터로 사용 (선택 사항)
            results.append({
                "content": docs[i].page_content,
                "context": "",  # Adjust as needed, e.g., surrounding text context
                "metadata": metadata,
                "distance": distance
            })
        return results
    except Exception as e:
        print(f"Error during search: {e}")
        return []

def search_manual_content(query, k=10, context_window=100):
    
    try:
        query_embedding = model.encode([query])
        D, I = index.search(query_embedding, k)

        results = []
        is_first_result = True  # Flag to track the first result
        for i, distance in zip(I[0], D[0]):
            metadata = {"source": documents[0].metadata["source"]}
            if hasattr(docs[i], "metadata") and "chunk" in docs[i].metadata:
                metadata["chunk"] = docs[i].metadata["chunk"]

            content = docs[i].page_content
            
            results.append({
                "content": content,
                "context": "", 
                "metadata": metadata,
                "distance": distance
            })
      
        return results
    
    except Exception as e:
        print(f"Error during search: {e}")
        return []
    
def search_manual_content_new(query, k=10, context_window=100):
    
    try:
        query_embedding = model.encode([query])
        D, I = index.search(query_embedding, k)

        results = []
        is_first_result = True  # Flag to track the first result
        for i, distance in zip(I[0], D[0]):
            metadata = {"source": documents[0].metadata["source"]}
            if hasattr(docs[i], "metadata") and "chunk" in docs[i].metadata:
                metadata["chunk"] = docs[i].metadata["chunk"]

            content = docs[i].page_content
            
            results.append({
                "content": content,
                "context": "", 
                "metadata": metadata,
                "distance": distance
            })

        results_new = []
        contents = ""
        for result in results:
            content = result.get("content", "")  # "content" 키가 없으면 빈 문자열 처리
            contents = contents + content   
        results_new.append({"content": contents})

        return results_new
    
    except Exception as e:
        print(f"Error during search: {e}")
        return []

def get_chat_gpt(api_key, question_str, reference_str):
    """
    OpenAI의 GPT-3.5 모델을 사용하여 질문에 답변하는 함수.

    Args:
        api_key (str): OpenAI API 키
        question_str (str): 질문 내용
        reference_str (str): 참고 자료 (문서 내용)

    Returns:
        str: GPT 모델이 생성한 답변
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    messages = [
        # {"role": "system", "content": "You are a helpful assistant."},
        # {"role": "system", "content": "모든 응답은 한국어로 작성되어야 합니다."},
        {"role": "system", "content": "Your primary task is to provide accurate and detailed answers based on the company's standard regulations."},
        {"role": "system", "content": "Ensure that your answers are detailed and strictly based on the reference material provided."},
        {"role": "system", "content": "Use the provided reference material to formulate your responses and ensure they are comprehensive."},
        {"role": "system", "content": "Provide thorough and well-explained answers derived from the reference materials provided."},
        {"role": "system", "content": "If the Reference does not contain information about the question, respond with '회사 표준 규정집에 없는 내용인 것 같습니다.'."},
        {"role": "user", "content": f"Reference: {reference_str}"},
        {"role": "user", "content": f"Question: {question_str}"}
    ]

    data = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "max_tokens": 2000,  # 필요에 따라 조절 가능
        "n": 1,
        "stop": None,
        "temperature": 0.5  # 필요에 따라 조절 가능
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)

    if response.status_code == 200:
        completion = response.json()
        return completion['choices'][0]['message']['content']
    else:
        error_msg = f"Error: {response.status_code}\n{response.json()}"
        raise Exception(error_msg)  # API 호출 실패 시 예외 발생
    
def get_chat_gpt_4(api_key, question_str, reference_str):
    """
    OpenAI의 GPT-4 모델을 사용하여 질문에 답변하는 함수.

    Args:
        api_key (str): OpenAI API 키
        question_str (str): 질문 내용
        reference_str (str): 참고 자료 (문서 내용)

    Returns:
        str: GPT 모델이 생성한 답변
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "system", "content": "모든 응답은 한국어로 작성되어야 합니다."},
        {"role": "user", "content": f"Reference: {reference_str}"},
        {"role": "user", "content": f"Question: {question_str}"}
    ]

    data = {
        "model": "gpt-4",
        "messages": messages,
        "max_tokens": 2000,  # 필요에 따라 조절 가능
        "n": 1,
        "stop": None,
        "temperature": 0.5  # 필요에 따라 조절 가능
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)

    if response.status_code == 200:
        completion = response.json()
        return completion['choices'][0]['message']['content']
    else:
        error_msg = f"Error: {response.status_code}\n{response.json()}"
        raise Exception(error_msg)  # API 호출 실패 시 예외 발생
    
def get_chat_gpt_only(api_key, question_str):    

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "system", "content": "As a senior developer, I'll provide you with a detailed explanation."},
        {"role": "system", "content": "Let's dive into this programming concept together."},
        {"role": "user", "content": f"Question: {question_str}"}
    ]

    data = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "max_tokens": 4096,
        "n": 1,
        "stop": None,
        "temperature": 0.5
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)

    if response.status_code == 200:
        completion = response.json()
        return completion['choices'][0]['message']['content']
    else:
        error_msg = f"Error: {response.status_code}\n{response.json()}"
        raise Exception(error_msg)  # API 호출 실패 시 예외 발생


# --- 5. 검색 엔드포인트 정의 ---
@app.post("/search", response_model=list[SearchResult])
async def search(query: Query, k: int = 10, context_window: int = 100):
    query_str = query.message  # message 필드에서 쿼리 추출
    if not query_str:
        raise HTTPException(status_code=400, detail="쿼리를 입력해주세요.")
    elif len(query_str) < 2:
        raise HTTPException(status_code=400, detail="쿼리는 최소 2글자 이상이어야 합니다.")

    results = search_manual_content(query_str, k, context_window) 
    
    results_new = []
    for result in results:
        content = result.get("content", "")  # "content" 키가 없으면 빈 문자열 처리
        results_new.append({"contents": content})
           
    # print(results_new)
    
    if not results:
        raise HTTPException(status_code=404, detail="관련 내용을 찾지 못했습니다.")

    return results

@app.post("/content", response_model=list[SearchResult_New])
async def search(query: Query, k: int = 10, context_window: int = 100):
    query_str = query.message  # message 필드에서 쿼리 추출
    if not query_str:
        raise HTTPException(status_code=400, detail="쿼리를 입력해주세요.")
    elif len(query_str) < 2:
        raise HTTPException(status_code=400, detail="쿼리는 최소 2글자 이상이어야 합니다.")

    results = search_manual_content_new(query_str, k, context_window) 
  
    if not results:
        raise HTTPException(status_code=404, detail="관련 내용을 찾지 못했습니다.")

    return results

@app.post("/chatgpt", response_model=ChatResponse)
async def chat_to_gpt(query:Query):

    # CORS 헤더 설정 (필요한 경우)
    # response = JSONResponse(content=result)
    # response.headers["Access-Control-Allow-Origin"] = "*"  # 모든 Origin 허용 (또는 특정 Origin 지정)
       
    api_key = ""  # 실제 API 키로 대체   
    question_str = query.message
    # reference: SearchResult_New = search_manual_content_new(question, 10, 100) 
    reference  = search_manual_content_new(question_str, 10, 100)
    reference_str = " ".join([item["content"] for item in reference if "content" in item])

    try:
        answer = get_chat_gpt(api_key, question_str, reference_str)
        # print(answer)
    except Exception as e:
        print(e) 
        
    result = []
    result = {
            "question": question_str,
            "reference": reference_str,
            "answer": answer
        } 

    # list of dictionaries
    # result.append({
    #       "question": question_str,
    #       "reference": reference_str,
    #       "answer": answer
    # })
    
    return result

@app.post("/chatgpt4", response_model=ChatResponse)
async def chat_to_gpt(query: Query):
       
    api_key = ""  # 실제 API 키로 대체   
    question_str = query.message
    reference = search_manual_content_new(question_str, 10, 100)
    reference_str = " ".join([item["content"] for item in reference if "content" in item])

    try:
        answer = get_chat_gpt_4(api_key, question_str, reference_str)
    except Exception as e:
        print(e) 
        answer = str(e)
        
    result = {
        "question": question_str,
        "reference": reference_str,
        "answer": answer
    }
    
    return result

@app.post("/gptonly", response_model=ChatOnlyResponse)
async def gpt(query: Query):
    api_key = ""  # Replace with your actual API key
    question_str = query.message
    
    try:
        answer = get_chat_gpt_only (api_key, question_str)
    except Exception as e:
        print(e) 
        answer = str(e)
        
    result = {
        "question": question_str,
        "answer": answer
    }
    
    return result

# --- 6. 서버 실행 ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
