import os
from pathlib import Path

TEMPLATE_TRAIN = "train_template.yaml"
TEMPLATE_PREDICT = "predict_template.yaml"

# 你可以根据实际情况修改这些路径和数据集名格式
MODEL_PATH = "/path/to/model"
OUTPUT_BASE = "/path/to/output/dir"
ADAPTER_BASE = "/path/to/output/dir"
TRAIN_DATASET_FMT = "follow_up_data_v0_f{fold}"
EVAL_DATASET_FMT = "follow_up_data_test_v0_f{fold}"

FOLDS = [1, 2, 3, 4, 5]


def render_template(template_path, replacements):
    with open(template_path, 'r') as f:
        content = f.read()
    for k, v in replacements.items():
        content = content.replace(k, v)
    return content

def main():
    base_dir = Path(__file__).parent
    for fold in FOLDS:
        # 训练配置
        train_replacements = {
            '/path/to/model': MODEL_PATH,
            '/path/to/output/dir': f'{OUTPUT_BASE}/qwen2_7b_instruct_lora_sft_fwf{fold}_v0',
            '<fold_train_dataset>': TRAIN_DATASET_FMT.format(fold=fold),
        }
        train_yaml = render_template(base_dir / TEMPLATE_TRAIN, train_replacements)
        with open(base_dir / f'qwen2_lora_sft_fwf{fold}.yaml', 'w') as f:
            f.write(train_yaml)

        # 预测配置
        predict_replacements = {
            '/path/to/model': MODEL_PATH,
            '/path/to/adapter': f'{ADAPTER_BASE}/qwen2_7b_instruct_lora_sft_fwf{fold}_v0',
            '/path/to/output/dir': f'{OUTPUT_BASE}/qwen2_7b_instruct_lora_sft_fwf{fold}_v0',
            '<fold_eval_dataset>': EVAL_DATASET_FMT.format(fold=fold),
        }
        predict_yaml = render_template(base_dir / TEMPLATE_PREDICT, predict_replacements)
        with open(base_dir / f'qwen2_lora_sft_fwf{fold}_predict.yaml', 'w') as f:
            f.write(predict_yaml)

if __name__ == "__main__":
    main() 