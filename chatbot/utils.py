from bs4 import BeautifulSoup
import requests
import os
from llama_index import GPTVectorStoreIndex, Document
from openai import OpenAI
import json
import time
import random

def crawl_content(url='https://www.presight.io/privacy-policy.html'):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Request to {url} failed with status code {response.status_code}")
    
    content = response.content
    soup = BeautifulSoup(content, 'html.parser')
    elements = soup.find_all(['p', 'h2'])
    data_list = [element.get_text() for element in elements]
    return data_list

def crawl_and_indexing(url='https://www.presight.io/privacy-policy.html', index_folder = './Index'):
    data_list = crawl_content(url)
    documents = [Document(text=". ".join(data_list))]
    index = GPTVectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=index_folder)
    
    return index_folder
    
def generate_questions(url='https://www.presight.io/privacy-policy.html', filepath='generated_question.json'):
    data_list = crawl_content(url)
    
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Generate json list of 40 questions that user will ask about this privacy policy.\n {'. '.join(data_list)}."},
        ],
        seed=42,
        response_format={'type': 'json_object'}
        )

    data = json.loads(completion.choices[0].message.content)
    with open(filepath, "w") as outfile: 
        json.dump(data, outfile) 

    return data

def evaluate(chatbot, questions, sleep_time=5):
    random.shuffle(questions)
    num_question = len(questions)
    start_time = time.time()
    for i, question in enumerate(questions):
        print(f"User: {question}")
        response = chatbot.generate_response(question)
        print(f"Bot: {response['content']}")
        time.sleep(sleep_time)
        end_time = time.time()
        total_time = end_time - start_time - (i+1)*sleep_time
        print(f"Evaluated on {i+1} questions, average response time {round(total_time/(i+1), 4)}")
    
if __name__ == "__main__":
    api_key = input("Enter api key:")
    os.environ["OPENAI_API_KEY"] = api_key
    
    url='https://www.presight.io/privacy-policy.html'
    try:
        crawl_and_indexing(url)
        generate_questions(url)
    except Exception as e:
        print(f"Error occurred while crawling or generate question from {url}: {str(e)}")
    