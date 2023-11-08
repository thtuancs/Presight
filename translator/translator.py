import os
import json
from openai import OpenAI

class TextTranslator():
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def translate(self, json_input: dict):
        text_to_translate = json_input['text']
        target_language = json_input['dest_language']
        
        promt = json.dumps(text_to_translate)
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": "You are a helpful translator."},
                {"role": "system", "content": f"Translate elements of the array to {target_language} language."},
                {"role": "user", "content":  promt},
            ],
            seed=42
            )
        
        result = completion.choices[0].message.content
        if isinstance(text_to_translate, list):
            result = json.loads(result)
            
        return result
        
    
if __name__ == "__main__":
    api_key = input("Input api key: ")
    os.environ["OPENAI_API_KEY"]
    translator = TextTranslator(api_key)
    
    # test single text
    single_text_input = {
        "text": "hello",
        "dest_language": "vi"
    }
    print("Test single text")
    print(translator.translate(single_text_input))
    
    multi_text_input = {
        "text": [
            "The sun is shining brightly.",
            "She is singing a beautiful song.",
            "He likes to play the guitar.",
            "We are having a picnic at the park.",
            "They are dancing at the party."
        ],
        "dest_language": "vi"
    }
    
    print("Test multiple texts")
    print(translator.translate(multi_text_input))
    
    