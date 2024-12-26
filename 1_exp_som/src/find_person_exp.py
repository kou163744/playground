import argparse
import base64
import requests
import os
import sys
import logging
import csv
import time
import pandas as pd
import google.generativeai as genai
import PIL.Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_folder', type=str, help="Folder containing images")
    parser.add_argument('-q', '--question_file', type=str, help="CSV file containing questions for each folder")
    parser.add_argument('-c', '--csv_output', type=str, default="output.csv", help="Path to output CSV file")
    parser.add_argument('--use_gemini', action='store_true', help="Use Gemini API instead of OpenAI GPT")
    return parser.parse_args()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_openai(api_key, target, image_paths):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    prompt = f"You given an image with each person and the corresponding number.\nYour task is to identify the number based on the given target. Output only the number corresponding to the persons matching the instructions, with no additional text.\nTarget: {target}\nNumber: "

    payload = {
        # "model": "gpt-4o-mini",
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "max_tokens": 30
    }

    for image_path in image_paths:
        encoded_image = encode_image(image_path)
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
            }
        }
        payload['messages'][0]['content'].append(image_content)

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        logging.info("API call successful.")
        return response.json()['choices'][0]['message']['content']
    else:
        logging.error(f"API call failed with status code {response.status_code}: {response.text}")
        return None

def call_gemini(api_key, target, image_paths):
    genai.configure(api_key=api_key)
    # model = genai.GenerativeModel('gemini-1.5-flash-latest')
    model = genai.GenerativeModel('gemini-1.5-pro-latest')

    prompt = f"You are given images where each person is labeled with a number.\nYour task is to identify the number based on the given target. Output only the number corresponding to the persons matching the instructions, with no additional text.\nTarget: {target}\nNumber: "

    images = [PIL.Image.open(image_path) for image_path in image_paths]

    response = model.generate_content([
        prompt,
        *images
    ], stream=False)
    response.resolve()
    return response.text

def process_folders(api_key, base_folder, question_file, output_csv, use_gemini):
    # Load questions from the CSV file
    questions_df = pd.read_csv(question_file)
    results = []

    # Sort folders numerically
    folder_names = sorted(os.listdir(base_folder), key=lambda x: int(x) if x.isdigit() else float('inf'))

    for folder_name in folder_names:
        folder_path = os.path.join(base_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        image_paths = sorted(
            [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith("marked_image")]
        )

        if not image_paths:
            logging.warning(f"No images found in folder: {folder_path}")
            continue

        logging.info(f"Processing folder: {folder_path}")

        # Filter questions for the current folder
        folder_questions = questions_df[questions_df['Level 1'].astype(str) == folder_name]

        if folder_questions.empty:
            logging.warning(f"No questions found for folder: {folder_name}")
            continue

        for index, row in folder_questions.iterrows():
            for question_col in ['question1', 'question2', 'question3']:
                question = row[question_col]
                logging.info(f"Processing question: {question}")
                if use_gemini:
                    response = call_gemini(api_key, question, image_paths)
                    time.sleep(5)
                else:
                    response = call_openai(api_key, question, image_paths)
                if response is None:
                    logging.warning(f"Empty response for question: {question} in folder: {folder_name}")
                results.append({"folder": folder_name, "question": question, "response": response})

    # Write results to CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["folder", "question", "response"])
        writer.writeheader()
        writer.writerows(results)

    logging.info(f"Results written to {output_csv}")

def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG)

    if args.use_gemini:
        if os.getenv("GEMINI_API") is None:
            logging.error("GEMINI_API environment variable is not set.")
            sys.exit()
        api_key = os.getenv("GEMINI_API")
    else:
        if os.getenv("OPENAI_API") is None:
            logging.error("OPENAI_API environment variable is not set.")
            sys.exit()
        api_key = os.getenv("OPENAI_API")

    if not args.image_folder:
        logging.error("Please provide a valid image folder.")
        sys.exit()

    if not args.question_file:
        logging.error("Please provide a valid CSV file containing questions.")
        sys.exit()

    try:
        process_folders(api_key, args.image_folder, args.question_file, args.csv_output, args.use_gemini)
    except Exception as e:
        logging.error(f"Error during processing: {e}")

if __name__ == "__main__":
    main(parse_args())
