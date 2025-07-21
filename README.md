## Project Structure

```
./
├── prompt/                 # Chinese prompts for experiments
├── prompt_en/             # English translations of prompts for readability
├── data_example.jsonl     # Data format specification
└── config/
    └── train_predict/     # Training and prediction configuration files
```

## Components

### Prompts
- **`prompt/`**: Contains Chinese prompt templates used in our experiments
  - `zero_shot_prompt.txt`: Zero-shot prompting template
  - `zero_shot_cot_prompt.txt`: Zero-shot chain-of-thought prompting template
  - `one_shot_prompt.txt`: One-shot prompting template with examples

- **`prompt_en/`**: English translations of the Chinese prompts for better accessibility
  - Corresponding English versions of all prompt templates

### Data Format
- **`data_example.jsonl`**: Defines the standard data format for all experiments
  ```json
  {
    "instruction": "prompt",
    "input": "chat_content", 
    "output": "parsed_result",
    "id": "id"
  }
  ```
  All data should be prepared according to this format.

### Configuration
- **`config/train_predict/`**: Contains training and prediction configuration files
  - `train_template.yaml`: Template for training configuration
  - `predict_template.yaml`: Template for prediction configuration
  - `gen_5fold_config.py`: Script for generating 5-fold cross-validation configurations

## Installation

1. Clone the LLaMA-Factory repository:
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
```

2. Install dependencies:
```bash
pip install -e ".[torch,metrics]" --no-build-isolation
```

## Usage

### Training

To start training with the provided configuration:

```bash
llamafactory-cli train examples/train_lora/follow_up_data_v0_fx.yaml
```
Remember copy config/train_predict to LLaMA-Factory/examples/train_lora

### Data Preparation

1. Prepare your data according to the format specified in `data_example.jsonl`
2. Configure training parameters in `config/train_predict/train_template.yaml`

## Citation


## License


## Contact

