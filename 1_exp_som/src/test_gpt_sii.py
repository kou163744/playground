import argparse

import base64
import requests
import os
import sys
import pandas as pd
import logging
import yaml
import pathlib
import textwrap
import time
import google.generativeai as genai
# from google.colab import userdata
# from IPython.display import display
# from IPython.display import Markdown
import PIL.Image

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image', type=str,)
    parser.add_argument('-o', '--order', type=str)
    return parser.parse_args()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_openai(api_key=None, origin_image_path=None, marked_image_path=None, text=None):

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    if origin_image_path == None:
        # Getting the base64 string
        base64_image_marked = encode_image(marked_image_path)
        payload = {
            "model": "gpt-4-vision-preview",
            # "model": "gpt-4o",
            "messages": [
              {
                "role": "user",
                "content": [
                  {
                    "type": "text",
                    "text": text,
                  },
                  {
                    "type": "image_url",
                    "image_url": {
                      "url": f"data:image/jpeg;base64,{base64_image_marked}"
                    }
                  }
                ]
              }
            ],
            "max_tokens": 300,
            "temperature": 0
        }
    else:
        base64_image_origin = encode_image(origin_image_path)
        base64_image_marked = encode_image(marked_image_path)
        payload = {
            "model": "gpt-4-vision-preview",
            # "model": "gpt-4o",
            "messages": [
              {
                "role": "user",
                "content": [
                  {
                    "type": "text",
                    "text": text,
                  },
                  {
                    "type": "image_url",
                    "image_url": {
                      "url": f"data:image/jpeg;base64,{base64_image_origin}"
                    }
                  },
                  {
                    "type": "image_url",
                    "image_url": {
                      "url": f"data:image/jpeg;base64,{base64_image_marked}"
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

def call_gemini(api_key=None, origin_image_path=None, marked_image_path=None, text=None):

    genai.configure(api_key=api_key)
    # model = genai.GenerativeModel('gemini-1.5-pro-latest')
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    # model = genai.GenerativeModel('gemini-1.0-pro-latest')
    
    if origin_image_path == None:
        # Getting the base64 string
        marked_img = PIL.Image.open(marked_image_path)
        response = model.generate_content([
            text, 
            marked_img
        ], stream=False)
    else:
        # Getting the base64 string
        origin_img = PIL.Image.open(origin_image_path)
        marked_img = PIL.Image.open(marked_image_path)
        response = model.generate_content([
            text, 
            origin_img,
            marked_img
        ], stream=False)
    response.resolve()
    print("*"*50)
    print(response.text)
    print("*"*50)
    time.sleep(25)
    return response.text
    
def main(args: argparse.Namespace):
    with open('../io/data/SII/level1_sii.yaml', 'r') as file:
        label_list = yaml.safe_load(file)
    logging.basicConfig(level=logging.DEBUG)
    specific_columns = ['absolute_position_1', 'absolute_position_2', 'relative_position_1', 'relative_position_2']
    # answer_columns = ['som_answer1', 'som_answer2', 'som_answer3', 'som_answer4']
    answer_columns = ['yolo_answer1', 'yolo_answer2', 'yolo_answer3', 'yolo_answer4']
    marked_image_path_base = '../io/data/SII/level1_results_sii'
    image_path_base = '../io/data/level1_image'
    csv_path = '../io/results/SII/level1_results_sii.csv'
    csv_data = pd.read_csv(csv_path)
    
    # if os.getenv("OPENAI_API") is None:
    #     logging.error("OPENAI_API error")
    #     sys.exit()
    # else:
    #     openai_api_key = os.getenv("OPENAI_API")

    if os.getenv("GEMINI_API") is None:
        logging.error("GEMINI_API error")
        sys.exit()
    else:
        gemini_api_key = os.getenv("GEMINI_API")

    for col_q, col_an in zip(specific_columns, answer_columns):
        count = 0
        correct_count1 = 0
        correct_count2 = 0
        correct_count3 = 0
        correct_count4 = 0
        answer1 = []
        answer2 = []
        answer3 = []
        answer4 = []
        if col_q in csv_data.columns and col_an in csv_data.columns:
            logging.info(f"Start: {col_q}")
            for order, graund_truth in zip(csv_data[col_q], csv_data[col_an]):
                count += 1
                
                origin_image_path = os.path.join(image_path_base, f"rgb_input{count}.jpg")
                marked_image_path = os.path.join(marked_image_path_base, f"rgb_input{count}.jpg")
                label = label_list.get(f"image{count}")
                only_mark_wo_label = f"You are given an image depicting each object and its corresponding label. The labels are represented as numbers placed on the objects. Your task is to identify the number based on a given instruction. Output only the number corresponding to the object that matches the instruction, with no additional text.\nInstruction: {order}\nNumber: "
                origin_mark_wo_label = f"You are given the first image, the original image, and the second image, with each object marked. The labels are represented as numbers placed above the objects. Your task is to identify the number based on a given instruction. Output only the number corresponding to the object that matches the instruction, with no additional text.\nInstruction: {order}\nNumber: "
                only_mark_w_label = f"You are given an image depicting each object and its corresponding label. The labels are represented as numbers placed on the objects. Name of object corresponding to mark: {label} Your task is to identify the number based on a given instruction. Output only the number corresponding to the object that matches the instruction, with no additional text.\nInstruction: {order}\nNumber: "
                origin_mark_w_label = f"You are given the first image, the original image, and the second image, with each object marked. The labels are represented as numbers placed above the objects. Name of object corresponding to mark: {label} Your task is to identify the number based on a given instruction. Output only the number corresponding to the object that matches the instruction, with no additional text.\nInstruction: {order}\nNumber: "
                
                if isinstance(order, str):
                    # responce1 = call_openai(api_key=openai_api_key, origin_image_path=None, marked_image_path=marked_image_path, text=only_mark_wo_label)
                    # responce2 = call_openai(api_key=openai_api_key, origin_image_path=None, marked_image_path=marked_image_path, text=origin_mark_wo_label)
                    # responce3 = call_openai(api_key=openai_api_key, origin_image_path=origin_image_path, marked_image_path=marked_image_path, text=only_mark_w_label)
                    # responce4 = call_openai(api_key=openai_api_key, origin_image_path=origin_image_path, marked_image_path=marked_image_path, text=origin_mark_w_label)
                    responce1 = call_gemini(api_key=gemini_api_key, origin_image_path=None, marked_image_path=marked_image_path, text=only_mark_wo_label)
                    responce2 = call_gemini(api_key=gemini_api_key, origin_image_path=None, marked_image_path=marked_image_path, text=origin_mark_wo_label)
                    responce3 = call_gemini(api_key=gemini_api_key, origin_image_path=origin_image_path, marked_image_path=marked_image_path, text=only_mark_w_label)
                    responce4 = call_gemini(api_key=gemini_api_key, origin_image_path=origin_image_path, marked_image_path=marked_image_path, text=origin_mark_w_label)
                    answer1.append(responce1)
                    if str(graund_truth) in str(responce1):
                        logging.info("correct1!!")
                        correct_count1 += 1
                    answer2.append(responce2)
                    if str(graund_truth) in str(responce2):
                        logging.info("correct2!!")
                        correct_count2 += 1
                    answer3.append(responce3)
                    if str(graund_truth) in str(responce3):
                        logging.info("correct3!!")
                        correct_count3 += 1
                    answer4.append(responce4)
                    if str(graund_truth) in str(responce4):
                        logging.info("correct4!!")
                        correct_count4 += 1
            csv_data[f"{col_q}_ans1"] = answer1
            csv_data[f"{col_q}_ans2"] = answer2
            csv_data[f"{col_q}_ans3"] = answer3
            csv_data[f"{col_q}_ans4"] = answer4
            answer_rate1 = []
            answer_rate1.append(f"{correct_count1}")
            answer_rate1.append(f"{float(correct_count1 // count)}")
            while len(answer_rate1) != len(csv_data[f"{col_q}_ans1"]):
                answer_rate1.append(None)
            csv_data[f"{col_q}_ans_rate1"] = answer_rate1
            answer_rate2 = []
            answer_rate2.append(f"{correct_count2}")
            answer_rate2.append(f"{float(correct_count2 // count)}")
            while len(answer_rate2) != len(csv_data[f"{col_q}_ans2"]):
                answer_rate2.append(None)
            csv_data[f"{col_q}_ans_rate2"] = answer_rate2
            answer_rate3 = []
            answer_rate3.append(f"{correct_count3}")
            answer_rate3.append(f"{float(correct_count3 // count)}")
            while len(answer_rate3) != len(csv_data[f"{col_q}_ans3"]):
                answer_rate3.append(None)
            csv_data[f"{col_q}_ans_rate3"] = answer_rate3
            answer_rate4 = []
            answer_rate4.append(f"{correct_count4}")
            answer_rate4.append(f"{float(correct_count4 // count)}")
            while len(answer_rate4) != len(csv_data[f"{col_q}_ans4"]):
                answer_rate4.append(None)
            csv_data[f"{col_q}_ans_rate4"] = answer_rate4

    csv_data.to_csv('../io/results/SII/updated_level1_results_gemini_flash.csv')



if __name__ == "__main__":
    main(parse_args())