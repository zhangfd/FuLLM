from pathlib import Path
import json
import pandas as pd
from typing import List, Dict, Optional, Callable, Any
from functools import partial
# 准备成eval兼容的数据
# fullm = finetune_qwen2_7b_aug
# fig2 finetune_qwen2_7b_aug vs gt
# fig3 fullm + gpt 3 time stamp
# fig4 fullm + DeepSeek-v3 (2024_12_26) + GPT-3.5-turbo (2025_01_25) + GPT-4o (2024_11_20) + claude 3.5-sonnet (2024_10_22) + gemini-2.0-pro (2025_02_05)
# fig5 fullm + 4 svm exps
# fig6 fullm + 4 staff + all staff (fullm只用all staff的数据)
# table2 fullm + 7b_wo_aug + 7b_instruct
# table3 gpt4 with 3 prompts
# table4 (same as fig4)
# table5 (same as fig5)
# supp table1 3 center
# supp table2 2 des_code
# supp table3 (same as fig3)
# supp table4 (same as fig6)

# ==============================================================================
# 1. 配置中心 (Configuration)
# ==============================================================================

BASE_DATA_PATH = Path("data")
META_PATH = BASE_DATA_PATH / "meta_info.json"
OUTPUT_ROOT = BASE_DATA_PATH / "eval"
OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)
SOURCE_ROOT = BASE_DATA_PATH / "output" / "source"
MODELS = {
    "fullm": "finetune_qwen2_7b_aug.xlsx",
    "claude-zero_shot": "claude-zero_shot.xlsx",
    "gpt35-zero_shot": "gpt35-zero_shot.xlsx",
    "gpt4-zero_shot-timepoint1": "gpt4-zero_shot-timepoint1.xlsx",
    "gpt4-zero_shot-timepoint2": "gpt4-zero_shot-timepoint2.xlsx",
    "gpt4-zero_shot-timepoint3": "gpt4-zero_shot-timepoint3.xlsx",
    "deepseekv3-zero_shot": "deepseekv3-zero_shot.xlsx",
    "finetune_qwen2_7b_wo_aug": "finetune_qwen2_7b_wo_aug.xlsx",
    "svm_tfidf_wo_aug": "svm_tfidf_wo_aug.xlsx",
    "svm_w2v_aug": "svm_w2v_aug.xlsx",
    "gpt4o-zero_shot": "gpt4o-zero_shot.xlsx",
    "gpt4-zero_shot": "gpt4-zero_shot.xlsx",
    "gpt4-zero_shot_cot": "gpt4-zero_shot_cot.xlsx",
    "gpt4-one_shot": "gpt4-one_shot.xlsx",
    "qwen-zero_shot": "qwen-zero_shot.xlsx",
    "svm_w2v_wo_aug": "svm_w2v_wo_aug.xlsx",
    "gemini-zero_shot": "gemini-zero_shot.xlsx",
    "svm_tfidf_aug": "svm_tfidf_aug.xlsx",
}

COLUMN_MAPPING_EN_CN = {
    "follow_up_source": "Pred_随访信息来源",
    "is_deceased": "Pred_受试者是否死亡",
    "is_hospitalized": "Pred_受试者是否住院",
    "had_surgery": "Pred_受试者是否手术",
    "is_medicated": "Pred_受试者是否用药"
}
# 反向映射，用于将xlsx的列名转换为内部标准key
COLUMN_MAPPING_CN_EN = {v: k for k, v in COLUMN_MAPPING_EN_CN.items()}

VALUE_MAPPING = {"是": 1, "否": 2, "不确定": 3, "未提及": 4, "本人": 1, "亲属": 2}

# ==============================================================================
# 2. 过滤器注册中心 (Filter Registry)
# ==============================================================================

def get_all_dcaps(meta_info: Dict[str, Any]) -> List[str]:
    return list(meta_info.keys())

def filter_by_root_value(meta_info: Dict[str, Any], key: str, value_set: set) -> List[str]:
    """通用过滤器：根据meta_info根层级的某个键值对来筛选dCap。"""
    return [dcap for dcap, data in meta_info.items() if data.get(key) in value_set]

def filter_by_reader(meta_info: Dict[str, Any]) -> List[str]:
    return [dcap for dcap, data in meta_info.items() if "reader" in data and len(data["reader"]) == 4]

FILTER_REGISTRY: Dict[str, Callable[[Dict[str, Any]], List[str]]] = {
    "all": get_all_dcaps,
    "hosp_d01": partial(filter_by_root_value, key="hospital_code", value_set={"d01"}),
    "hosp_d02": partial(filter_by_root_value, key="hospital_code", value_set={"d02"}),
    "hosp_d03": partial(filter_by_root_value, key="hospital_code", value_set={"d03"}),
    "exam_type_1": partial(filter_by_root_value, key="exam_type_id", value_set={1}),
    "exam_type_2": partial(filter_by_root_value, key="exam_type_id", value_set={2}),
    "fold_1": partial(filter_by_root_value, key="fold_id", value_set={1}),
    "fold_2": partial(filter_by_root_value, key="fold_id", value_set={2}),
    "fold_3": partial(filter_by_root_value, key="fold_id", value_set={3}),
    "fold_4": partial(filter_by_root_value, key="fold_id", value_set={4}),
    "fold_5": partial(filter_by_root_value, key="fold_id", value_set={5}),
    "filter_by_reader": filter_by_reader,

}

# ==============================================================================
# 3. 任务定义 (采用统一的 pred_key/gt_key 结构)
# ==============================================================================
TASKS = [
    {
        "task_name": "fullm_vs_silver",
        "output_filename": "fig2.xlsx",
        "filter_key": "all",
        "sources": [
            {"pred_key": "fullm", "gt_key": "silver", "sheet_name": "fullm_vs_silver"},
        ]
    },
    {
        "task_name": "fullm_vs_gpt4-3timestamp",
        "output_filename": "fig3.xlsx",
        "filter_key": "hosp_d01",
        "sources":[
            {"pred_key": "fullm", "gt_key": "silver", "sheet_name": "fullm_vs_silver"},
            {"pred_key": "gpt4-zero_shot-timepoint1", "gt_key": "silver", "sheet_name": "gpt4_tp1_vs_silver"},
            {"pred_key": "gpt4-zero_shot-timepoint2", "gt_key": "silver", "sheet_name": "gpt4_tp2_vs_silver"},
            {"pred_key": "gpt4-zero_shot-timepoint3", "gt_key": "silver", "sheet_name": "gpt4_tp3_vs_silver"},
        ]
    },
    {
        "task_name": "fullm_vs_different_models",
        "output_filename": "fig4.xlsx",
        "filter_key": "all",
        "sources":[
            {"pred_key": "fullm", "gt_key": "silver", "sheet_name": "fullm_vs_silver"},
            {"pred_key": "deepseekv3-zero_shot", "gt_key": "silver", "sheet_name": "deepseekv3_vs_silver"},
            {"pred_key": "gpt35-zero_shot", "gt_key": "silver", "sheet_name": "gpt35_vs_silver"},
            {"pred_key": "gpt4o-zero_shot", "gt_key": "silver", "sheet_name": "gpt4o_vs_silver"},
            {"pred_key": "claude-zero_shot", "gt_key": "silver", "sheet_name": "claude_vs_silver"},
            {"pred_key": "gemini-zero_shot", "gt_key": "silver", "sheet_name": "gemini_vs_silver"},
        ]
    },
    {
        "task_name": "fullm_vs_svms",
        "output_filename": "fig5.xlsx",
        "filter_key": "all",
        "sources":[
            {"pred_key": "fullm", "gt_key": "silver", "sheet_name": "fullm_vs_silver"},
            {"pred_key": "svm_tfidf_wo_aug", "gt_key": "silver", "sheet_name": "svm_tfidf_wo_aug_vs_silver"},
            {"pred_key": "svm_w2v_wo_aug", "gt_key": "silver", "sheet_name": "svm_w2v_wo_aug_vs_silver"},
            {"pred_key": "svm_tfidf_aug", "gt_key": "silver", "sheet_name": "svm_tfidf_aug_vs_silver"},
            {"pred_key": "svm_w2v_aug", "gt_key": "silver", "sheet_name": "svm_w2v_aug_vs_silver"},
        ]
    },
    {
        "task_name": "fullm_vs_human",
        "output_filename": "fig6_part1.xlsx",
        "filter_key": "filter_by_reader",
        "post_process": "merge_readers",
        "sources":[
            {"pred_key": "fullm", "gt_key": "silver", "sheet_name": "fullm_vs_silver"},
            {"pred_key": "reader.wangyueyan", "gt_key": "silver", "sheet_name": "reader_wangyueyan_vs_silver"},
            {"pred_key": "reader.wanghan", "gt_key": "silver", "sheet_name": "reader_wanghan_vs_silver"},
            {"pred_key": "reader.zhaojinlong", "gt_key": "silver", "sheet_name": "reader_zhaojinlong_vs_silver"},
            {"pred_key": "reader.panghuimin", "gt_key": "silver", "sheet_name": "reader_panghuimin_vs_silver"},
        ]
    },
    {
        "task_name": "fullm_vs_qwen",
        "output_filename": "table2.xlsx",
        "filter_key": "all",
        "sources":[
            {"pred_key": "fullm", "gt_key": "silver", "sheet_name": "fullm_vs_silver"},
            {"pred_key": "qwen-zero_shot", "gt_key": "silver", "sheet_name": "qwen_7b_instruct_vs_silver"},
            {"pred_key": "finetune_qwen2_7b_wo_aug", "gt_key": "silver", "sheet_name": "finetune_qwen_wo_aug_vs_silver"},
        ]
    },
    {
        "task_name": "gpt4_different_prompts",
        "output_filename": "table3.xlsx",
        "filter_key": "all",
        "sources":[
            {"pred_key": "gpt4-zero_shot", "gt_key": "silver", "sheet_name": "gpt4_zero_shot_vs_silver"},
            {"pred_key": "gpt4-zero_shot_cot", "gt_key": "silver", "sheet_name": "gpt4_zero_shot_cot_vs_silver"},
            {"pred_key": "gpt4-one_shot", "gt_key": "silver", "sheet_name": "gpt4_one_shot_vs_silver"},
        ]
    },
    {
        "task_name": "different_models",
        "output_filename": "table4.xlsx",
        "filter_key": "all",
        "sources":[
            {"pred_key": "deepseekv3-zero_shot", "gt_key": "silver", "sheet_name": "deepseekv3_vs_silver"},
            {"pred_key": "gpt35-zero_shot", "gt_key": "silver", "sheet_name": "gpt35_vs_silver"},
            {"pred_key": "gpt4o-zero_shot", "gt_key": "silver", "sheet_name": "gpt4o_vs_silver"},
            {"pred_key": "claude-zero_shot", "gt_key": "silver", "sheet_name": "claude_vs_silver"},
            {"pred_key": "gemini-zero_shot", "gt_key": "silver", "sheet_name": "gemini_vs_silver"},
        ]
    },
    {
        "task_name": "svms",
        "output_filename": "table5.xlsx",
        "filter_key": "all",
        "sources":[
            {"pred_key": "svm_tfidf_wo_aug", "gt_key": "silver", "sheet_name": "svm_tfidf_wo_aug_vs_silver"},
            {"pred_key": "svm_w2v_wo_aug", "gt_key": "silver", "sheet_name": "svm_w2v_wo_aug_vs_silver"},
            {"pred_key": "svm_tfidf_aug", "gt_key": "silver", "sheet_name": "svm_tfidf_aug_vs_silver"},
            {"pred_key": "svm_w2v_aug", "gt_key": "silver", "sheet_name": "svm_w2v_aug_vs_silver"},
        ]
    },
    {
        "task_name": "supp_table1_part1",
        "output_filename": "supp_table1_part1.xlsx",
        "filter_key": "hosp_d01",
        "sources":[
                {"pred_key": "fullm", "gt_key": "silver", "sheet_name": "fullm_vs_silver"},
        ]
    },
    {
        "task_name": "supp_table1_part2",
        "output_filename": "supp_table1_part2.xlsx",
        "filter_key": "hosp_d02",
        "sources":[
                {"pred_key": "fullm", "gt_key": "silver", "sheet_name": "fullm_vs_silver"},
        ]
    },
    {
        "task_name": "supp_table1_part3",
        "output_filename": "supp_table1_part3.xlsx",
        "filter_key": "hosp_d03",
        "sources":[
                {"pred_key": "fullm", "gt_key": "silver", "sheet_name": "fullm_vs_silver"},
        ]
    },
    {
        "task_name": "supp_table2_part1",
        "output_filename": "supp_table2_part1.xlsx",
        "filter_key": "exam_type_1",
        "sources":[
                {"pred_key": "fullm", "gt_key": "silver", "sheet_name": "fullm_vs_silver"},
        ]
    },
    {
        "task_name": "supp_table2_part2",
        "output_filename": "supp_table2_part2.xlsx",
        "filter_key": "exam_type_2",
        "sources":[
                {"pred_key": "fullm", "gt_key": "silver", "sheet_name": "fullm_vs_silver"},
        ]
    },
    {
        "task_name": "gpt4-3timestamp",
        "output_filename": "supp_table3.xlsx",
        "filter_key": "hosp_d01",
        "sources":[
            {"pred_key": "gpt4-zero_shot-timepoint1", "gt_key": "silver", "sheet_name": "gpt4_tp1_vs_silver"},
            {"pred_key": "gpt4-zero_shot-timepoint2", "gt_key": "silver", "sheet_name": "gpt4_tp2_vs_silver"},
            {"pred_key": "gpt4-zero_shot-timepoint3", "gt_key": "silver", "sheet_name": "gpt4_tp3_vs_silver"},
        ]
    },
    {
        "task_name": "fullm_vs_human_supp",
        "output_filename": "supp_table4_part1.xlsx",
        "filter_key": "filter_by_reader",
        "post_process": "merge_readers",
        "sources":[
            {"pred_key": "fullm", "gt_key": "silver", "sheet_name": "fullm_vs_silver"},
            {"pred_key": "reader.wangyueyan", "gt_key": "silver", "sheet_name": "reader_wangyueyan_vs_silver"},
            {"pred_key": "reader.wanghan", "gt_key": "silver", "sheet_name": "reader_wanghan_vs_silver"},
            {"pred_key": "reader.zhaojinlong", "gt_key": "silver", "sheet_name": "reader_zhaojinlong_vs_silver"},
            {"pred_key": "reader.panghuimin", "gt_key": "silver", "sheet_name": "reader_panghuimin_vs_silver"},
        ]
    },
]

# ==============================================================================
# 4. 核心处理器 (Processor) - 采用统一数据源解析器
# ==============================================================================

class EvaluationDataProcessor:
    def __init__(self, meta_info_path: Path, output_root: Path):
        self.output_root = output_root
        try:
            with open(meta_info_path, 'r', encoding='utf-8') as f:
                self.meta_info = json.load(f)
            print("Successfully loaded meta info.")
        except Exception as e:
            print(f"Error loading or parsing meta info: {e}")
            self.meta_info = {}

    def _resolve_path(self, data: Dict, path: str) -> Optional[Any]:
        """安全地解析字典中的嵌套路径，如 'reader.wangyueyan'"""
        keys = path.split('.')
        current = data
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def _resolve_data_source(self, key: str) -> Dict[str, Dict[str, Any]]:
        """【核心】统一数据源解析器。接收任意key，返回标准化数据字典。"""
        print(f"    - Resolving data source for key: '{key}'...")
        data_dict = {}
        if key in MODELS:
            # --- Case 1: Key points to an Excel model output ---
            file_path = SOURCE_ROOT / MODELS[key]
            try:
                df = pd.read_excel(file_path)
                for _, row in df.iterrows():
                    dcap_id = row['dCap']
                    data_dict[dcap_id] = {
                        COLUMN_MAPPING_CN_EN.get(col): VALUE_MAPPING.get(row[col], -1)
                        for col in COLUMN_MAPPING_CN_EN if col in row
                    }
            except FileNotFoundError:
                print(f"      - Warning: File not found for model '{key}' at {file_path}")
        else:
            # --- Case 2: Key is a path within meta_info.json ---
            for dcap_id, dcap_data in self.meta_info.items():
                resolved_data = self._resolve_path(dcap_data, key)
                if isinstance(resolved_data, dict):
                    data_dict[dcap_id] = resolved_data

        if not data_dict:
            print(f"      - Warning: No data resolved for key '{key}'.")
        
        return data_dict

    def _generate_comparison_df(self, pred_key: str, gt_key: str, dcap_list: List[str]) -> pd.DataFrame:
        """【统一方法】基于任意两个数据源key生成对比DataFrame。"""
        pred_data_source = self._resolve_data_source(pred_key)
        gt_data_source = self._resolve_data_source(gt_key)
        
        processed_rows = []
        for dcap_id in dcap_list:
            pred_data = pred_data_source.get(dcap_id, {})
            gt_data = gt_data_source.get(dcap_id, {})
            
            output_row = {"dCap": dcap_id}
            # only include data with gt
            for internal_key in gt_data:
                short_name = COLUMN_MAPPING_EN_CN.get(internal_key, internal_key).split("_", 1)[1]
                output_row[f"Pred_{short_name}"] = pred_data.get(internal_key, -1)
                output_row[f"GT_{short_name}"] = gt_data.get(internal_key, -1)
            processed_rows.append(output_row)
        return pd.DataFrame(processed_rows)

    def run_task(self, task_config: Dict[str, Any]):
        task_name = task_config['task_name']
        print(f"\n===== Running task: {task_name} =====")

        filter_key = task_config.get("filter_key", "all")
        filter_func = FILTER_REGISTRY[filter_key]
        print(f"  - Applying filter: '{filter_key}'...")
        dcap_list = filter_func(self.meta_info)
        print(f"  - Filter selected {len(dcap_list)} dCaps.")

        output_path = self.output_root / task_config["output_filename"]
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            print(f"  - Writing to {output_path}")
            for source in task_config["sources"]:
                sheet_name = source["sheet_name"]
                print(f"  - Generating sheet: '{sheet_name}' (Pred: {source['pred_key']}, GT: {source['gt_key']})")
                df = self._generate_comparison_df(source['pred_key'], source['gt_key'], dcap_list)
                if not df.empty:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"    - Success: Wrote {len(df)} rows.")
                else:
                    print(f"    - Warning: No data generated for this sheet.")
        print(f"===== Task '{task_name}' finished. =====")

    def run_merge_readers_post_process(self, task_config: Dict[str, Any]):
        """
        专用于处理"fig6"和"supp_table4"的后处理任务。
        合并多个reader的标注结果，并将dCap与reader_name拼接。
        """
        task_name = task_config['task_name']
        print(f"\n===== Running Post-processing (Merge Readers) for task: {task_name} =====")

        # 1. 获取 dCap 列表
        filter_key = task_config.get("filter_key", "all")
        filter_func = FILTER_REGISTRY[filter_key]
        dcap_list = filter_func(self.meta_info)
        print(f"  - Using filter '{filter_key}' which selected {len(dcap_list)} dCaps.")

        # 2. 识别 reader 数据源并准备合并
        reader_dfs = []
        reader_sources = [s for s in task_config.get("sources", []) if s.get("pred_key", "").startswith("reader.")]
        
        if not reader_sources:
            print("  - Warning: No reader sources found in task config. Skipping merge.")
            print(f"===== Post-processing for '{task_name}' finished. =====")
            return

        for source in reader_sources:
            pred_key = source["pred_key"]
            gt_key = source.get("gt_key", "silver")
            reader_name = pred_key.split('.')[-1]
            
            print(f"  - Processing data for reader: '{reader_name}'...")
            
            df = self._generate_comparison_df(pred_key, gt_key, dcap_list)
            
            if not df.empty:
                # 拼接 dCap 和 reader_name
                df["dCap"] = df["dCap"].astype(str) + "_" + reader_name
                reader_dfs.append(df)
            else:
                print(f"    - Warning: No data generated for reader '{reader_name}'.")

        if not reader_dfs:
            print("  - No reader data was processed successfully. Skipping file generation.")
            print(f"===== Post-processing for '{task_name}' finished. =====")
            return

        # 3. 合并 DataFrame
        merged_df = pd.concat(reader_dfs, ignore_index=True)
        
        # 4. 确定输出文件名 (part1 -> part2)
        original_filename = Path(task_config["output_filename"])
        new_stem = original_filename.stem.replace("part1", "part2")
        output_filename = new_stem + original_filename.suffix
        
        output_path = self.output_root / output_filename
        sheet_name = "human_vs_silver"
        
        # 5. 写入 Excel
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            print(f"  - Writing merged data to '{output_path}' in sheet '{sheet_name}'...")
            merged_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"    - Success: Wrote {len(merged_df)} rows.")

        print(f"===== Post-processing for '{task_name}' finished. =====")

# ==============================================================================
# 5. 执行器 (Orchestrator)
# ==============================================================================

if __name__ == "__main__":
    processor = EvaluationDataProcessor(meta_info_path=META_PATH, output_root=OUTPUT_ROOT)

    if processor.meta_info:
        for task in TASKS:
            if task["task_name"] == "gpt4_different_prompts":
                processor.run_task(task)
            if task.get("post_process") == "merge_readers":
                processor.run_merge_readers_post_process(task)