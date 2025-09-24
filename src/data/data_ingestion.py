import os
import re
import sys
import json
import pandas as pd
from logparser.Drain import LogParser
from tqdm import tqdm
import logging

# Configuration logger
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("errors.log", encoding="utf-8")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# === CONFIGURATION DES CHEMINS ===
BASE_DIR = "D:/ETSProject"
INPUT_FILE = os.path.join(BASE_DIR, "data", "HDFS.log")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "HDFS_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 1. SPLIT ===
def split_logfile(input_file, train_file, valid_file, test_file, train_ratio=0.6, valid_ratio=0.2):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    train_idx = int(len(lines) * train_ratio)
    valid_idx = int(len(lines) * (train_ratio + valid_ratio))

    with open(train_file, 'w') as f:
        f.writelines(lines[:train_idx])
    with open(valid_file, 'w') as f:
        f.writelines(lines[train_idx:valid_idx])
    with open(test_file, 'w') as f:
        f.writelines(lines[valid_idx:])

    print("✅ Split terminé")

# === 2. PARSING ===
def parse_logs(log_dir, files, log_format, regex, depth=4, st=0.5):
    parser = LogParser(log_format=log_format, indir=log_dir, outdir=log_dir, depth=depth, st=st, rex=regex)
    for file in files:
        parser.parse(file)
        print(f"✅ Parsed {file}")

# === 3. MAPPING JSON ===
def generate_json_mappings(csv_files, out_dir):
    for file in csv_files:
        csv_path = os.path.join(out_dir, file)
        df = pd.read_csv(csv_path)
        if "Occurrences" not in df.columns:
            logger.warning(f"'Occurrences' non trouvé dans {csv_path}, skip.")
            continue
        df = df.sort_values(by="Occurrences", ascending=False)
        mapping = {eid: f"E{idx + 1}" for idx, eid in enumerate(df["EventId"])}
        json_path = os.path.join(out_dir, file.replace(".csv", ".json"))
        with open(json_path, "w") as f:
            json.dump(mapping, f)
        print(f"✅ JSON mapping généré : {json_path}")

# === 4. STRUCTURATION + LABELS ===
def enrich_structured_logs(input_dir, output_dir, json_file, struct_file, label_file, output_file):
    df_struct = pd.read_csv(os.path.join(output_dir, struct_file))
    mapping = json.load(open(os.path.join(output_dir, json_file)))
    df_labels = pd.read_csv(os.path.join(input_dir, label_file))
    df_labels['Label'] = df_labels['Label'].replace({'Normal': 'Success', 'Anomaly': 'Fail'})

    df_struct['BlockId'] = df_struct['Content'].apply(lambda x: re.search(r'blk_(|-)[0-9]+', x).group(0) if re.search(r'blk_(|-)[0-9]+', x) else None)
    df_struct = df_struct.dropna(subset=['BlockId'])
    df_struct['EventId'] = df_struct['EventId'].apply(lambda x: mapping.get(x, x))
    df_struct = pd.merge(df_struct, df_labels, on='BlockId', how='left')

    cols = ['BlockId', 'Label'] + [c for c in df_struct.columns if c not in ['BlockId', 'Label']]
    df_struct = df_struct[cols]
    df_struct.to_csv(os.path.join(output_dir, output_file), index=False)
    print(f"✅ Fichier enrichi généré : {output_file}")

# === MAIN PIPELINE ===
def main():
    try:
        # Fichiers de sortie
        train_log = "HDFS_train.log"
        valid_log = "HDFS_valid.log"
        test_log  = "HDFS_test.log"

        train_path = os.path.join(OUTPUT_DIR, train_log)
        valid_path = os.path.join(OUTPUT_DIR, valid_log)
        test_path  = os.path.join(OUTPUT_DIR, test_log)

        # 1. Split
        split_logfile(INPUT_FILE, train_path, valid_path, test_path)

        # 2. Parsing
        log_format = "<Date> <Time> <Pid> <Level> <Component>: <Content>"
        regex = [
            r'blk_(|-)[0-9]+', 
            r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', 
            r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$'
        ]
        parse_logs(OUTPUT_DIR, [train_log, valid_log, test_log], log_format, regex)

        # 3. JSON Mapping
        generate_json_mappings([
            "HDFS_train.log_templates.csv",
            "HDFS_valid.log_templates.csv",
            "HDFS_test.log_templates.csv"
        ], OUTPUT_DIR)

        # 4. Enrich logs
        label_file = "anomaly_label.csv"
        enrich_structured_logs(BASE_DIR + "/data", OUTPUT_DIR,
                               "HDFS_train.log_templates.json", "HDFS_train.log_structured.csv", label_file, "HDFS_train_enriched.csv")
        enrich_structured_logs(BASE_DIR + "/data", OUTPUT_DIR,
                               "HDFS_valid.log_templates.json", "HDFS_valid.log_structured.csv", label_file, "HDFS_valid_enriched.csv")
        enrich_structured_logs(BASE_DIR + "/data", OUTPUT_DIR,
                               "HDFS_test.log_templates.json", "HDFS_test.log_structured.csv", label_file, "HDFS_test_enriched.csv")

        print("🎯 Pipeline complet exécuté avec succès.")
        
    except Exception as e:
        logger.error("Erreur critique : %s", e)
        print(f"❌ Pipeline échoué : {e}")

if __name__ == "__main__":
    main()
