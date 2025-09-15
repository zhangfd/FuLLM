import json
from sklearn.model_selection import KFold
from typing import Dict, List
import os

def process_data(file_path: str, output_dir: str, is_augmentation: bool = False):
    META_INFO_PATH = "data/meta_info.json"

    # Load meta_info
    if os.path.exists(META_INFO_PATH):
        with open(META_INFO_PATH, 'r', encoding='utf-8') as f:
            meta_info = json.load(f)
    else:
        meta_info = {}
    
    # Read JSONL file
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            dcap = item.get('dCap')
            if dcap:
                if dcap not in data:
                    data[dcap] = {'origin': [], 'augmentation': []}
                # Assuming 'augmentation' field in item distinguishes between origin and augmentation
                if item.get('augmentation'):
                    data[dcap]['augmentation'].append(item)
                else:
                    data[dcap]['origin'].append(item)

    all_keys = list(data.keys())

    # check for keys that need fold_id assignment
    new_keys = [key for key in all_keys if key not in meta_info or 'fold_id' not in meta_info.get(key, {})]
    
    meta_updated = False
    if new_keys:
        print(f"Found {len(new_keys)} new keys to assign to folds.")
        meta_updated = True
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold_idx, (_, test_index) in enumerate(kf.split(new_keys), 1):
            for i in test_index:
                key = new_keys[i]
                if key not in meta_info:
                    meta_info[key] = {}
                meta_info[key]['fold_id'] = fold_idx
    
    # Save meta_info if updated
    if meta_updated:
        os.makedirs(os.path.dirname(META_INFO_PATH), exist_ok=True)
        with open(META_INFO_PATH, 'w', encoding='utf-8') as f:
            json.dump(meta_info, f, ensure_ascii=False, indent=4)
        print(f"Updated meta_info saved to {META_INFO_PATH}")

    # Create folds based on meta_info
    folds = {i: [] for i in range(1, 6)}
    for key in all_keys:
        if key in meta_info and 'fold_id' in meta_info[key]:
            fold_id = meta_info[key]['fold_id']
            folds[fold_id].append(key)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    for fold in range(1, 6):
        test_keys = folds.get(fold, [])
        train_keys = []
        for i in range(1, 6):
            if i != fold:
                train_keys.extend(folds.get(i, []))

        # Output as JSON Lines format
        train_file = os.path.join(output_dir, f'train_fold_{fold}.jsonl')
        test_file = os.path.join(output_dir, f'test_fold_{fold}.jsonl')

        with open(train_file, 'w', encoding='utf-8') as f:
            for key in train_keys:
                if key in data:
                    for item in data[key]['origin']:
                        json.dump(item, f, ensure_ascii=False)
                        f.write('\n')
                    if is_augmentation:
                        for item in data[key]['augmentation']:
                            json.dump(item, f, ensure_ascii=False)
                            f.write('\n')

        with open(test_file, 'w', encoding='utf-8') as f:
            for key in test_keys:
                if key in data:
                    for item in data[key]['origin']:
                        json.dump(item, f, ensure_ascii=False)
                        f.write('\n')

    print("Data processing completed.")

if __name__ == "__main__":
    # Augmented data
    file_path = "data/wav_alpaca_dataset_dCap_all.jsonl"
    output_dir = 'data/follow_up_train_data_aug'
    process_data(file_path, output_dir, is_augmentation=True)
    # Original data
    file_path = "data/wav_alpaca_dataset_dCap_all.jsonl"
    output_dir = 'data/follow_up_train_data_wo_aug'
    process_data(file_path, output_dir, is_augmentation=False)
