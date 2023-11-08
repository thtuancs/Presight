import os
import json
from openai import OpenAI
from glob import glob

def save_testset(func):
    def wrapper(*args, **kwargs):
        data_completion = func(*args, **kwargs)
        data = json.loads(data_completion.choices[0].message.content)

        folder = kwargs['folder']
        os.makedirs(folder, exist_ok=True)
        for k, v in data.items():
            with open(os.path.join(folder, f'{k}.json'), "w") as outfile: 
                json.dump(v, outfile)    
                
    return wrapper

@save_testset
def create_single_text_testset(client, folder='test_single_text'):
    data_completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are a helpful assistance."},
            {"role": "system", "content": "Generate list of json, about 50 elements, with format \
            {'Input':\
                {'text': <english word>,\
                'dest_language': 'vi'\
            }\
            'Expected output': <translated word>}"},
        ],
        seed=42,
        response_format={'type': 'json_object'}
    )

    return data_completion
    
@save_testset
def create_multi_text_testset(client, folder='test_multi_text'):
    data_completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are a helpful assistance."},
            {"role": "system", "content": "Generate list of json, total 10 elements, with format \
            {'Input':\
                {'text': <list 5 english short sentence>,\
                'dest_language': 'vi'\
            }\
            'Expected output': <list english translated sentence>}"},
        ],
        seed=42,
        response_format={'type': 'json_object'}
    )
    return data_completion

if __name__ == "__main__":
    api_key = input("Enter api key:")
    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI(api_key=api_key)
    create_single_text_testset(client, folder='test_single_text')
    create_multi_text_testset(client, folder='test_multi_text')
    