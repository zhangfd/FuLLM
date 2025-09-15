# merge all data into one jsonl file
from pathlib import Path
from datasets import load_dataset
from datasets import exceptions
from collections import defaultdict
import json

def main(): 
    composed_output = Path("data/wav_alpaca_dataset_dCap_all.jsonl")
    all_data = [Path("data/wav_alpaca_dataset_dCap.jsonl"),
                Path("data/wav_alpaca_dataset_dCap_rewritten.jsonl"),
                Path("data/wav_alpaca_dataset_dCap_synthesized.jsonl")]
    unique_inputs = set()
    output = defaultdict(dict)
    pass_del_data = 0
    pass_dup_data = 0
    for pth in all_data:
        print(str(pth))
        # load data path i by datasets.load_dataset
        try:
            dataset = load_dataset("json", data_files=str(pth), split="train")
        except exceptions.DatasetGenerationError:
            print(f"Error loading dataset from {pth}")
            continue
        is_origin = pth == all_data[0]
        for item in dataset:
            if item["input"] in unique_inputs:
                pass_dup_data += 1
                continue
            unique_inputs.add(item["input"])
            if not output[item["dCap"]]:
                output[item["dCap"]]["origin"] = []
                output[item["dCap"]]["augmentation"] = []
            if is_origin:
                output[item["dCap"]]["origin"].append(item)
            else:
                if "删除" in item["input"] or "省略" in item["input"]: #  or "\n\n" in item["input"]:
                    pass_del_data += 1
                    continue
                output[item["dCap"]]["augmentation"].append(item)
    print("all del pass data", pass_del_data)
    print("all dup pass data", pass_dup_data)
    with composed_output.open("w") as file:
        json.dump(output, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()