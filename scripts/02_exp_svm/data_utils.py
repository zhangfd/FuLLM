import json
import numpy as np
from pathlib import Path
from datasets import load_dataset


def convert_data(data_list):
    """Convert raw data to texts and labels"""
    texts = []
    labels = []
    kept_raw_data = []
    expected_keys = {
        "随访信息来源": ["亲属", "本人", "未提及"],
        "受试者是否死亡": ["否", "是", "不确定", "未提及"],
        "受试者是否住院": ["否", "是", "不确定", "未提及"],
        "受试者是否手术": ["否", "是", "不确定", "未提及"],
        "受试者是否用药": ["否", "是", "不确定", "未提及"]
    }
    
    for item in data_list:
        try:
            input_text = item['input']
            output = item['output']
            lines = output.split('\n')
            
            item_labels = []
            processed_keys = set()
            
            for line in lines:
                parts = line.split('：', 1)
                if len(parts) != 2:
                    continue
                
                key, value = parts
                if key in expected_keys and key not in processed_keys:
                    possible_values = expected_keys[key]
                    if value in possible_values:
                        item_labels.append(possible_values.index(value))
                    else:
                        item_labels.append(len(possible_values) - 1)
                    processed_keys.add(key)
            
            for key in expected_keys:
                if key not in processed_keys:
                    item_labels.append(len(expected_keys[key]) - 1)
            
            if len(item_labels) == len(expected_keys):
                texts.append(input_text)
                labels.append(item_labels)
                kept_raw_data.append(item)
            else:
                print(f"Skipping data item due to unexpected number of labels: {len(item_labels)}")
                
        except Exception as e:
            print(f"Skipping data item due to error: {str(e)}")
            continue
    
    return texts, np.array(labels), kept_raw_data



def load_original_data(data_path):
    """Load original data"""
    data_path = Path(data_path)
    if data_path.suffix == '.jsonl':
        # Load JSONL format
        raw_data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                raw_data.append(json.loads(line))
    else:
        # Load JSON format
        raw_data = json.loads(data_path.read_text())
    
    texts, labels, kept_raw_data = convert_data(raw_data)
    return texts, labels, kept_raw_data



def load_fold_data(fold_data_dir, fold):
    """Load training and testing data for specified fold"""
    fold_data_dir = Path(fold_data_dir)
    train_data_path = fold_data_dir / f"train_fold_{fold}.jsonl"
    test_data_path = fold_data_dir / f"test_fold_{fold}.jsonl"
    
    train_data = [i for i in load_dataset("json", data_files=str(train_data_path), split="train")]
    test_data = [i for i in load_dataset("json", data_files=str(test_data_path), split="train")]
    
    train_texts, train_labels, kept_train_data = convert_data(train_data)
    test_texts, test_labels, kept_test_data = convert_data(test_data)
    
    train_dcaps = [i.get("dCap", "") for i in kept_train_data]
    test_dcaps = [i.get("dCap", "") for i in kept_test_data]
    
    return (train_texts, train_labels, train_dcaps), (test_texts, test_labels, test_dcaps) 