import argparse

import base64
import requests
import os
import sys
import pandas as pd
import logging
    

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image', type=str,)
    parser.add_argument('-o', '--order', type=str)
    return parser.parse_args()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_vlm(api_key, origin_image_path, marked_image_path, order):

    # Getting the base64 string
    origin_base64_image = encode_image(origin_image_path)
    marked_base64_image = encode_image(marked_image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    text_prompt = f"You are given the first image, the original image, and the second image, with each object marked. The labels are represented as numbers placed above the objects. Your task is to identify the number based on a given instruction. Output only the number corresponding to the object that matches the instruction, with no additional text.\nInstruction: {order}\nNumber: "

    payload = {
        # "model": "gpt-4-vision-preview",
        "model": "gpt-4o",
        "messages": [
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": text_prompt,
              },
              {
                "type": "image_url",
                "image_url": {
                  "url": f"data:image/jpeg;base64,{origin_base64_image}"
                }
              },
              {
                "type": "image_url",
                "image_url": {
                  "url": f"data:image/jpeg;base64,{marked_base64_image}"
                }
              }
            ]
          }
        ],
        "max_tokens": 300,
        "temperature": 0
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print("*"*50)
    print(response.json())
    print(response.json()['choices'][0]['message']['content'])
    print("*"*50)
    return response.json()['choices'][0]['message']['content']
    
def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG)
    specific_columns = ['absolute_position_1', 'absolute_position_2', 'relative_position_1', 'relative_position_2']
    answer_columns = ['som_answer1', 'som_answer2', 'som_answer3', 'som_answer4']
    # answer_columns = ['yolo_answer1', 'yolo_answer2', 'yolo_answer3', 'yolo_answer4']
    marked_image_path_base = '../io/SoM/level1'
    origin_image_path_base = '../io/data/level1_image'
    csv_path = '../io/results/SII/level1_results_sii.csv'
    csv_data = pd.read_csv(csv_path)
    if os.getenv("OPENAI_API") is None:
        logging.error("OPENAI_API error")
        sys.exit()
    else:
        api_key = os.getenv("OPENAI_API")

    for col_q, col_an in zip(specific_columns, answer_columns):
        count = 0
        correct_count = 0
        answer = []
        if col_q in csv_data.columns and col_an in csv_data.columns:
            logging.info(f"Start: {col_q}")
            for order, graund_truth in zip(csv_data[col_q], csv_data[col_an]):
                count += 1
                marked_image_path = os.path.join(marked_image_path_base, f"image{count}.png")
                origin_image_path = os.path.join(origin_image_path_base, f"rgb_input{count}.jpg")
                # image_path = os.path.join(image_path_base, f"image{count}.png") # for som
                if isinstance(order, str):
                    responce = call_vlm(api_key, origin_image_path, marked_image_path, order)
                    answer.append(responce)
                    if str(responce) == str(graund_truth):
                        logging.info("correct!!")
                        correct_count += 1
                    else:
                        logging.info("incorrect!")
            csv_data[f"{col_q}_ans1"] = answer
            answer_rate = []
            answer_rate.append(f"{correct_count}")
            answer_rate.append(f"{float(correct_count // count)}")
            while len(answer_rate) != len(csv_data[f"{col_q}_ans1"]):
                answer_rate.append(None)
            csv_data[f"{col_q}_ans_rate1"] = answer_rate

    csv_data.to_csv('../io/results/updated_level1_results_som.csv')



if __name__ == "__main__":
    main(parse_args())