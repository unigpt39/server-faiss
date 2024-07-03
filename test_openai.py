import requests

api_key = ""
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# 질문할 데이터와 참고할 데이터 정의
question_data = "Can you provide an overview of OpenAI's GPT-4 model capabilities?"
reference_data = "OpenAI's GPT-4 model is the latest in a series of powerful language models. It has 175 billion parameters and can perform a wide range of natural language processing tasks."

# 메시지 형식으로 prompt 구성
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "system", "content": "모든 응답은 한국어로 작성되어야 합니다."},
    {"role": "user", "content": f"Reference: {reference_data}"},
    {"role": "user", "content": f"Question: {question_data}"}
]

data = {
    "model": "gpt-3.5-turbo",  # Ensure you are using the appropriate model
    "messages": messages,
    "max_tokens": 150,
    "n": 1,
    "stop": None,
    "temperature": 0.5
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
if response.status_code == 200:
    completion = response.json()
    print(completion['choices'][0]['message']['content'])
else:
    print(f"Error: {response.status_code}")
    print(response.json())
