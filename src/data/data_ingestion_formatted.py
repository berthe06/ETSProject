
import sys
sys.path.append('/content/logparser')

import pandas as pd
import os
import re
import json
from logparser.Drain import LogParser
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml
import logging


# Logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler) 


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def split_logfile(input_file, train_file, valid_file, test_file, train_ratio=0.6, valid_ratio=0.2):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    train_split_index = int(len(lines) * train_ratio)
    valid_split_index = int(len(lines) * (train_ratio + valid_ratio))

    train_lines = lines[:train_split_index]
    valid_lines = lines[train_split_index:valid_split_index]
    test_lines = lines[valid_split_index:]

    with open(train_file, 'w') as file:
        file.writelines(train_lines)

    with open(valid_file, 'w') as file:
        file.writelines(valid_lines)

    with open(test_file, 'w') as file:
        file.writelines(test_lines)

    print(f"Split completed.")

def mapping(file_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file_name in file_names:
        log_templates_file = os.path.join(output_dir, file_name)
        log_temp = pd.read_csv(log_templates_file).sort_values(by="Occurrences", ascending=False)
        log_temp_dict = {event: f"E{idx + 1}" for idx, event in enumerate(log_temp["EventId"])}
        output_file = os.path.join(output_dir, f"{file_name.replace('.csv', '')}.json")
        with open(output_file, "w") as f:
            json.dump(log_temp_dict, f)

def process_log_files(input_dir, output_dir, json_filename, structured_log_filename, anomaly_label_filename, output_filename):
    json_file_path = os.path.join(output_dir, json_filename)
    anomaly_label_path = os.path.join(input_dir, anomaly_label_filename)
    structured_log_path = os.path.join(output_dir, structured_log_filename)

    df_structured = pd.read_csv(structured_log_path)
    with open(json_file_path, 'r') as json_file:
        event_mapping = json.load(json_file)

    df_labels = pd.read_csv(anomaly_label_path)
    df_labels['Label'] = df_labels['Label'].replace({'Normal': 'Success', 'Anomaly': 'Fail'})
    df_structured['BlockId'] = df_structured['Content'].apply(lambda x: re.search(r'blk_(|-)[0-9]+', x).group(0) if re.search(r'blk_(|-)[0-9]+', x) else None)
    df_structured = df_structured.dropna(subset=['BlockId'])
    df_structured['EventId'] = df_structured['EventId'].apply(lambda x: event_mapping.get(x, x))
    df_structured = pd.merge(df_structured, df_labels, on='BlockId', how='left')
    columns = ['BlockId', 'Label'] + [col for col in df_structured.columns if col not in ['BlockId', 'Label']]
    df_structured = df_structured[columns]
    df_structured.to_csv(output_filename, index=False)

