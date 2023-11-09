import os
import openai
import json
from llama_index import StorageContext, load_index_from_storage, ServiceContext
import utils
from llama_index.indices.postprocessor import SentenceEmbeddingOptimizer

class Chatbot:
    def __init__(self, api_key, index, use_optimizer=False, percentile_cutoff=0.7):
        self.index = index
        openai.api_key = api_key
        if use_optimizer:
            self.query_engine = self.index.as_query_engine(optimizer=SentenceEmbeddingOptimizer(percentile_cutoff=percentile_cutoff))
        else:
            self.query_engine = self.index.as_query_engine()
            
        self.chat_history = []

    def generate_response(self, user_input):
        prompt = "\n".join([f"{message['role']}: {message['content']}" 
                           for message in self.chat_history[-5:]])
        prompt += f"\nUser: {user_input}"
        response = self.query_engine.query(user_input)

        message = {"role": "assistant", "content": response.response}
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append(message)
        return message

    def load_chat_history(self, filename):
        try:
            with open(filename, 'r') as f:
                self.chat_history = json.load(f)
        except FileNotFoundError:
            pass

    def save_chat_history(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.chat_history, f)
            
if __name__ == "__main__":
    api_key = input("Enter api key:")
    os.environ["OPENAI_API_KEY"] = api_key
    
    storage_context = StorageContext.from_defaults(persist_dir='Index')
    index = load_index_from_storage(storage_context)
    bot = Chatbot(api_key, index=index, use_optimizer=True)
    bot.load_chat_history("chat_history.json")
    
    # # Evaluate response time
    with open('generated_question.json') as file:
        questions = json.load(file)["questions"]
    utils.evaluate(chatbot=bot, questions=questions)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["bye", "goodbye"]:
            print("Bot: Goodbye!")
            bot.save_chat_history("chat_history.json")
            break
        response = bot.generate_response(user_input)
        print(f"Bot: {response['content']}")