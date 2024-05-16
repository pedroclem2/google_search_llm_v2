import os
import openai
import base64
from ..configs.service_config import get_openai_api_key

class WineImageAnalyzer:
    def __init__(self):
        api_key = get_openai_api_key()
        self.client = openai.OpenAI(api_key=api_key)

    @staticmethod
    def open_file(filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()

    @staticmethod
    def save_file(filepath, content):
        with open(filepath, 'a', encoding='utf-8') as outfile:
            outfile.write(content)

    def gpt4o_chat(self, user_query):
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": user_query}
            ],
            temperature=0.4, 
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content

    @staticmethod
    def encode_image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def analyze_images_in_folder(self, folder_path):
        descriptions = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                base64_image = self.encode_image_to_base64(image_path)
                
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "You are going to be provided with either the image of a food or a wine. If somethig else is provided you should apoligise and state that you can only handle images of food or wine no matter what. Your job is to simply returnt he name of the food or the name of the wine."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ],
                        }
                    ],
                    max_tokens=1000
                )
                descriptions.append(f"{response.choices[0].message.content}")
        return descriptions[0]

def main():
    analyzer = WineImageAnalyzer()
    
    folder_path = "service/databases/images"
    image_desc = analyzer.analyze_images_in_folder(folder_path)
    # explainer = f"Image descriptions {image_desc}\n\nIn your answer, give me back just the wine details from the image description above in a structured format please"
    # final_answer = analyzer.gpt4o_chat(explainer)
    print(f"{image_desc}")

if __name__ == "__main__":
    main()
