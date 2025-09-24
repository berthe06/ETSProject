import numpy as np
import pandas as pd
import json
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from tqdm import tqdm
import logging

# logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Download required NLTK data
nltk.download('wordnet')
nltk.download('stopwords')

# Chemin direct du fichier des labels
ANOMALY_LABEL_PATH = "/ETSProject/data/anomaly_label.csv"

def process_log_files(input_dir, output_dir, json_filename, structured_log_filename, output_filename):
    json_file_path = os.path.join(output_dir, json_filename)
    structured_log_path = os.path.join(output_dir, structured_log_filename)

    df_structured = pd.read_csv(structured_log_path)
    with open(json_file_path, 'r') as json_file:
        event_mapping = json.load(json_file)
    df_labels = pd.read_csv(ANOMALY_LABEL_PATH)
    df_labels['Label'] = df_labels['Label'].replace({'Normal': 'Success', 'Anomaly': 'Fail'})

    df_structured['BlockId'] = df_structured['Content'].apply(lambda x: re.search(r'blk_(|-)[0-9]+', x).group(0) if re.search(r'blk_(|-)[0-9]+', x) else None)
    df_structured = df_structured.dropna(subset=['BlockId'])
    df_structured['EventId'] = df_structured['EventId'].apply(lambda x: event_mapping.get(x, x))
    df_structured = pd.merge(df_structured, df_labels, on='BlockId', how='left')

    columns = ['BlockId', 'Label'] + [col for col in df_structured.columns if col not in ['BlockId', 'Label']]
    df_structured = df_structured[columns]

    df_structured.to_csv(output_filename, index=False)
    print(f" Fichier généré : {output_filename}")

def hdfs_sampling(file_names, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file_name in file_names:
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name.replace('.csv', '_sequence.csv'))

        struct_log = pd.read_csv(input_path, engine='c', na_filter=False, memory_map=True, dtype={'Time': str})
        struct_log['Time'] = struct_log['Time'].str.zfill(6)
        struct_log['Date'] = struct_log['Date'].astype(str).str.zfill(6)
        struct_log['BlockId'] = struct_log['Content'].str.extract(r'(blk_-?\d+)')
        struct_log['EventId'] = struct_log['EventId'].fillna('')
        struct_log['Label'] = struct_log['Label'].apply(lambda x: 1 if x == 'Fail' else 0)

        data_dict = defaultdict(list)
        time_dict = defaultdict(list)
        date_dict = defaultdict(list)
        type_count = defaultdict(int)

        grouped = struct_log.groupby('BlockId')
        for block_id, group in tqdm(grouped, total=len(grouped)):
            data_dict[block_id] = group['EventId'].tolist()
            time_dict[block_id] = pd.to_datetime(group['Time'], format='%H%M%S', errors='coerce').dropna()
            date_dict[block_id] = group['Date'].tolist()
            type_count[block_id] = group['Label'].sum()

        rows = []
        for block_id, events in tqdm(data_dict.items(), total=len(data_dict)):
            features = [event for event in events if event]
            times = time_dict[block_id]
            dates = date_dict[block_id]
            if len(times) > 1:
                time_intervals = [(times.iloc[i] - times.iloc[i - 1]).total_seconds() for i in range(1, len(times))]
                latency = (times.iloc[-1] - times.iloc[0]).total_seconds()
            else:
                time_intervals = []
                latency = 0
            label = 'Fail' if type_count[block_id] > 0 else 'Success'
            first_date = dates[0] if dates else ''
            first_time = times.iloc[0].strftime('%H%M%S') if not times.empty else ''
            rows.append({
                "BlockId": block_id,
                "Label": label,
                "Type": type_count[block_id],
                "Features": str(features),
                "Date": first_date,
                "Time": first_time,
                "TimeInterval": str(time_intervals),
                "Latency": latency
            })

        data_df = pd.DataFrame(rows)
        data_df.to_csv(output_path, index=False)
        print(f" HDFS sampling terminé : {output_path}")

def generate_event_occurrence_matrix(log_files, event_traces_files, input_dir, output_dir, event_columns=None):
    if event_columns is None:
        event_columns = [f"E{i}" for i in range(1, 30)]

    anomaly_labels = pd.read_csv(ANOMALY_LABEL_PATH)
    anomaly_labels['Label'] = anomaly_labels['Label'].apply(lambda x: 'Fail' if x == 'Anomaly' else 'Success')
    label_dict = anomaly_labels.set_index('BlockId')['Label'].to_dict()

    for log_file, event_traces_file in zip(log_files, event_traces_files):
        output_file = os.path.join(output_dir, f"Event_occurence_matrix_{log_file.replace('.log', '')}.csv")
        print(f"Processing {log_file}...")
        event_traces = pd.read_csv(event_traces_file)
        occurrence_matrix = []

        for _, row in event_traces.iterrows():
            block_id = row['BlockId']
            label = label_dict.get(block_id, 'Unknown')
            event_list = re.findall(r"E\d+", row['Features'])
            event_counts = {event: event_list.count(event) for event in event_columns}
            occurrence_matrix.append({
                "BlockId": block_id,
                "Label": label,
                "Type": int(row['Type']) if pd.notna(row['Type']) else 0,
                "Time": row.get('Time', ''),
                "Date": row.get('Date', ''),
                **event_counts
            })

        occurrence_matrix_df = pd.DataFrame(occurrence_matrix)
        occurrence_matrix_df.to_csv(output_file, index=False)
        print(f" Matrice d'occurrence sauvegardée : {output_file}")

# ========================
# Execution du pipeline
# ========================

input_dir = '/ETSProject/data/HDFS_results/'
output_dir = '/ETSProject/data/HDFS_results/'

process_log_files(input_dir, output_dir, 'HDFS_train.log_templates.json', 'HDFS_train.log_structured.csv', os.path.join(output_dir, 'HDFS_train.log_structured_blk.csv'))
process_log_files(input_dir, output_dir, 'HDFS_valid.log_templates.json', 'HDFS_valid.log_structured.csv', os.path.join(output_dir, 'HDFS_valid.log_structured_blk.csv'))
process_log_files(input_dir, output_dir, 'HDFS_test.log_templates.json',  'HDFS_test.log_structured.csv',  os.path.join(output_dir, 'HDFS_test.log_structured_blk.csv'))

hdfs_sampling([
    'HDFS_train.log_structured_blk.csv',
    'HDFS_valid.log_structured_blk.csv',
    'HDFS_test.log_structured_blk.csv'
], output_dir, output_dir)

generate_event_occurrence_matrix(
    ['HDFS_train.log', 'HDFS_valid.log', 'HDFS_test.log'],
    [
        os.path.join(output_dir, 'HDFS_train.log_structured_blk_sequence.csv'),
        os.path.join(output_dir, 'HDFS_valid.log_structured_blk_sequence.csv'),
        os.path.join(output_dir, 'HDFS_test.log_structured_blk_sequence.csv')
    ],
    output_dir,
    output_dir
)

def main():
    try:
        logger.debug("Démarrage du prétraitement des données...")
        train_data = pd.read_csv('/ETSProject/data/HDFS_results/HDFS_train.log_structured_blk.csv')
        test_data = pd.read_csv('/ETSProject/data/HDFS_results/HDFS_test.log_structured_blk.csv')

        def preprocess(df):
            df = df.drop(columns=["Time", "Date", "Type", "Component", "Pid", "Level"], errors='ignore')
            df['Label'] = df['Label'].map({'Fail': 1, 'Success': 0})
            return df

        train_processed_data = preprocess(train_data)
        test_processed_data = preprocess(test_data)

        output_dir = '/ETSProject/data/HDFS_results/preprocessed'
        os.makedirs(output_dir, exist_ok=True)

        train_processed_data.to_csv(os.path.join(output_dir, 'train_processed.csv'), index=False)
        test_processed_data.to_csv(os.path.join(output_dir, 'test_processed.csv'), index=False)
        logger.debug(" Données prétraitées sauvegardées dans : %s", output_dir)

    except Exception as e:
        logger.error(" Échec du prétraitement : %s", e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
