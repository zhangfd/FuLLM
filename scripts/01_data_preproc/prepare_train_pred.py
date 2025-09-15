# TODO: 生成训练推理用脚本 主要是准备dataset info和template 而且还要准备2套数据集
import os
from pathlib import Path
import json

TEMPLATE_TRAIN = "config/train_predict/train_template.yaml"
TEMPLATE_PREDICT = "config/train_predict/predict_template.yaml"
DATASET_IDENTIFIERS = ["aug", "wo_aug"]
MODEL_PATH = "/mnt/gptdata/llm_weight/Qwen/Qwen2-7B-Instruct" # model path
OUTPUT_BASE = "data/trained_models" # trained result output dir (include adapter path)
TRAIN_DATASET_FMT = "follow_up_data_{DATASET_IDENTIFIER}_f{fold}" # train dataset name
PRED_DATASET_FMT = "follow_up_data_test_{DATASET_IDENTIFIER}_f{fold}" # predict dataset name
YAML_OUTPUT_ROOT = "config/train_predict"  # 生成yaml的根目录（相对repo根目录）
DATASET_INFO_OUTPUT = "data/dataset_info.json"  # dataset_info输出路径（相对repo根目录）
RUN_SCRIPT_PATH = "scripts/run_train_predict.sh"  # 生成的运行脚本路径（相对repo根目录）

FOLDS = [1, 2, 3, 4, 5]

def generate_dataset_info():
    dataset_info = {}
    for identifier in DATASET_IDENTIFIERS:
        for fold in FOLDS:
            train_key = TRAIN_DATASET_FMT.format(DATASET_IDENTIFIER=identifier, fold=fold)
            test_key = PRED_DATASET_FMT.format(DATASET_IDENTIFIER=identifier, fold=fold)
            dataset_info[train_key] = {
                "file_name": f"follow_up_train_data_{identifier}/train_fold_{fold}.jsonl",  # 相对dataset_dir
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output"
                },
                "description": f"Follow-up training data ({identifier}), fold {fold}"
            }
            dataset_info[test_key] = {
                "file_name": f"follow_up_train_data_{identifier}/test_fold_{fold}.jsonl",  # 相对dataset_dir
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output"
                },
                "description": f"Follow-up test data ({identifier}), fold {fold}"
            }
    return dataset_info
    


def render_template(template_path, replacements):
    with open(template_path, 'r') as f:
        content = f.read()
    for k, v in replacements.items():
        content = content.replace(k, v)
    return content


def generate_run_script(repo_root: Path, ordered_cfgs: list[Path]):
    """生成最简运行脚本：
    - 仅设置 CUDA_VISIBLE_DEVICES（可自行改为具体卡号）
    - 直接逐行写入 llamafactory-cli train <cfg> 与其 predict 对应项
    - 轻量日志：统一 tee 到一个 log 文件
    """
    run_path = repo_root / RUN_SCRIPT_PATH
    run_path.parent.mkdir(parents=True, exist_ok=True)

    cmd_lines = []
    for p in ordered_cfgs:
        cfg_abs = str(p)
        cmd_lines.append(f'echo "Run: {cfg_abs}" | tee -a "$LOG_FILE"')
        cmd_lines.append(f'llamafactory-cli train "{cfg_abs}" 2>&1 | tee -a "$LOG_FILE"')
        cmd_lines.append('echo "----------------------------" | tee -a "$LOG_FILE"')

    script = f"""#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2
REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
DATA_DIR="$REPO_ROOT/data"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/logs/train_predict_$TS"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run.log"

{os.linesep.join(cmd_lines)}
"""

    with open(run_path, 'w') as f:
        f.write(script)
    os.chmod(run_path, 0o755)


def main():
    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parent.parent  # scripts/01_data_preproc -> scripts -> repo root

    # 生成dataset_info.json（包含两套数据 + 5折的train/test）到repo根目录的data/
    dataset_info = generate_dataset_info()
    dataset_info_path = repo_root / DATASET_INFO_OUTPUT
    dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    # 收集生成的配置，按 trainN -> predictN 的顺序，便于生成运行脚本
    ordered_cfgs: list[Path] = []

    for identifier in DATASET_IDENTIFIERS:
        # 每个identifier一个子目录（位于repo根目录的config/train_predict/<identifier>）
        yaml_out_dir = repo_root / YAML_OUTPUT_ROOT / identifier
        yaml_out_dir.mkdir(parents=True, exist_ok=True)

        for fold in FOLDS:
            # 训练配置
            train_replacements = {
                '/path/to/model': MODEL_PATH,
                '/path/to/output/dir': f'{OUTPUT_BASE}/qwen2_7b_instruct_lora_sft_{identifier}_f{fold}_v0',
                '<fold_train_dataset>': TRAIN_DATASET_FMT.format(DATASET_IDENTIFIER=identifier, fold=fold),
            }
            train_yaml = render_template(repo_root / TEMPLATE_TRAIN, train_replacements)
            train_path = yaml_out_dir / f'qwen2_lora_sft_{identifier}_f{fold}.yaml'
            with open(train_path, 'w') as f:
                f.write(train_yaml)

            # 预测配置
            predict_replacements = {
                '/path/to/model': MODEL_PATH,
                '/path/to/adapter': f'{OUTPUT_BASE}/qwen2_7b_instruct_lora_sft_{identifier}_f{fold}_v0',
                '/path/to/output/dir': f'{OUTPUT_BASE}/qwen2_7b_instruct_lora_sft_{identifier}_f{fold}_v0',
                '<fold_eval_dataset>': PRED_DATASET_FMT.format(DATASET_IDENTIFIER=identifier, fold=fold),
            }
            predict_yaml = render_template(repo_root / TEMPLATE_PREDICT, predict_replacements)
            predict_path = yaml_out_dir / f'qwen2_lora_sft_{identifier}_f{fold}_predict.yaml'
            with open(predict_path, 'w') as f:
                f.write(predict_yaml)

            # 按要求的顺序追加
            ordered_cfgs.append(train_path)
            ordered_cfgs.append(predict_path)

    # 生成可直接运行的脚本
    generate_run_script(repo_root, ordered_cfgs)

if __name__ == "__main__":
    main() 