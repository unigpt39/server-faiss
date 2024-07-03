import requests
import json

def get_chat_gpt():
    # Define API key and headers
    api_key = ""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Define question and reference data
    question_data = "Can you provide an overview of OpenAI's GPT-4 model capabilities?"
    reference_data = "OpenAI's GPT-4 model is the latest in a series of powerful language models. It has 175 billion parameters and can perform a wide range of natural language processing tasks."

    # Construct messages in the required format
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Reference: {reference_data}"},
        {"role": "user", "content": f"Question: {question_data}"}
    ]

    # Prepare data object for the POST request
    data = {
        "model": "gpt-3.5-turbo",  # Ensure you are using the appropriate model
        "messages": messages,
        "max_tokens": 150,
        "n": 1,
        "stop": None,
        "temperature": 0.5
    }

    # Make the POST request to OpenAI API
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        if response.status_code == 200:
            completion = response.json()['choices'][0]['message']['content']
            print(completion)
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
    except Exception as e:
        print(f"Error: {e}")

# Call the function to initiate the API request
get_chat_gpt()
