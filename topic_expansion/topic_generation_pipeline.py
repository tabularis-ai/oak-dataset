import os
import csv
from openai import OpenAI
import time
from tqdm import tqdm  # Import tqdm for the progress bar

# Configuration 
OPEN_AI_KEY = "..."
CSV_FILE = 'categories_details_depth_3.csv'
OUTPUT_CSV_FILE = 'extended_categories.csv'
MAX_RETRIES = 20  # Maximum number of retries for API calls
RETRY_DELAY = 1  # Initial delay between retries in seconds, will double each retry
TOPICS_PER_CATEGORY = 20  # Number of unique topics to generate for each category

def read_csv(csv_file, start_category=None):
    """Read categories from a CSV file, optionally starting from a specific category."""
    skip = start_category is not None
    with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if skip and row['Category'] == start_category:
                skip = False
                continue
            yield row['Category'], int(row['Depth']), int(row['Page Count'])

def get_last_finished_category(output_csv_file):
    try:
        with open(output_csv_file, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            last_row = None
            for row in reader:
                last_row = row
            if last_row:
                return last_row['Category']
    except FileNotFoundError:
        return None

def generate_prompt(category, depth, page_count):
    prompt = f"""This is very important for my life and career. Given a category: {category} generate {TOPICS_PER_CATEGORY} unique related topics, separated by commas. 
    Do not include anything else, just the topics separated by commas, example: Topic1, Topic2, Topic3 ...  Please adhere strictly to this format without numbering the topics."""
    return prompt

def make_api_call(prompt):
    """Make an API call to OpenAI and return the response."""
    client = OpenAI(api_key=OPEN_AI_KEY)
    for attempt in range(MAX_RETRIES):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4o",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}. Retrying...")
            time.sleep(RETRY_DELAY)
    return None
    

def process_categories():
    """Process categories and generate topics, writing results to a CSV file."""
    last_finished_category = get_last_finished_category(OUTPUT_CSV_FILE)
    with open(OUTPUT_CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write header if the file is empty
        if file.tell() == 0:
            writer.writerow(['Category', 'Page Count', 'Generated Topics'])
        # Wrap the reading with tqdm for a progress bar
        for category, depth, page_count in tqdm(read_csv(CSV_FILE, start_category=last_finished_category), desc="Processing Categories"):
            prompt = generate_prompt(category, depth, page_count)
            response = make_api_call(prompt)
            if response:
                writer.writerow([category, page_count, response])
            else:
                print(f"Failed to generate topics for category: {category}")

if __name__ == "__main__":
    process_categories()