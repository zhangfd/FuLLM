# different models 
python scripts/prompt_experiment_runner.py --prompt-type 'zero_shot' --model 'qwen' --output-dir data/output/qwen-zero_shot
python scripts/prompt_experiment_runner.py --prompt-type 'zero_shot' --model 'gpt35' --output-dir data/output/gpt35-zero_shot
python scripts/prompt_experiment_runner.py --prompt-type 'zero_shot' --model 'gpt4o' --output-dir data/output/gpt4o-zero_shot
python scripts/prompt_experiment_runner.py --prompt-type 'zero_shot' --model 'deepseekv3' --output-dir data/output/deepseekv3-zero_shot
python scripts/prompt_experiment_runner.py --prompt-type 'zero_shot' --model 'claude' --output-dir data/output/claude-zero_shot
python scripts/prompt_experiment_runner.py --prompt-type 'zero_shot' --model 'gemini' --output-dir data/output/gemini-zero_shot

# gpt with different timestamp
python scripts/prompt_experiment_runner.py --prompt-type 'zero_shot' --model 'gpt4' --output-dir data/output/gpt4-zero_shot
# python scripts/prompt_experiment_runner.py --prompt-type 'zero_shot' --model 'gpt4' --output-dir data/output/gpt4-zero_shot-timepoint1
# python scripts/prompt_experiment_runner.py --prompt-type 'zero_shot' --model 'gpt4' --output-dir data/output/gpt4-zero_shot-timepoint2
# python scripts/prompt_experiment_runner.py --prompt-type 'zero_shot' --model 'gpt4' --output-dir data/output/gpt4-zero_shot-timepoint3

# different type
python scripts/prompt_experiment_runner.py --prompt-type 'zero_shot_cot' --model 'gpt4' --output-dir data/output/gpt4-zero_shot_cot
python scripts/prompt_experiment_runner.py --prompt-type 'one_shot' --model 'gpt4' --output-dir data/output/gpt4-one_shot