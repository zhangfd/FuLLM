import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
from llmtoolkit.aioutils import BaseGenerator
from llmtoolkit.utils import Config, parsing_json
from loguru import logger


class PromptExperimentGenerator(BaseGenerator):
    """
    Generator for running experiments with different prompt types on the wav_alpaca dataset.
    Supports 3 experiment types:
    1. zero_shot
    2. zero_shot_cot
    3. one_shot

    """
    
    def __init__(self, config: Config, model_name: str = None):
        """
        Initialize the generator with config and optional model name.
        
        Args:
            config: Configuration object
            model_name: Model name to use (if None, will use default from config)
        """
        super().__init__(config)
        self.custom_model_name = model_name
    
    def init_models(self):
        """Initialize the LLM models to be used for generation."""
        # Use custom model name if provided, otherwise use default
        if self.custom_model_name:
            self.model_name = self.custom_model_name
        else:

            self.model_name = self.cfg.LOCAL_QWEN_MODEL

        # Register all available models from .env.example

        # Qwen Model
        self.llm_manager.register_model(
            self.cfg.LOCAL_QWEN_MODEL,
            self.cfg.LOCAL_QWEN_KEY,
            self.cfg.LOCAL_QWEN_BASE
        )
        # GPT-4 Model
        self.llm_manager.register_model(
            self.cfg.GPT4_MODEL,
            self.cfg.GPT4_KEY,
            self.cfg.GPT4_BASE
        )
        # GPT-3.5 Model
        self.llm_manager.register_model(
            self.cfg.GPT35_MODEL,
            self.cfg.GPT35_KEY,
            self.cfg.GPT35_BASE
        )
        # GPT-4o Model
        self.llm_manager.register_model(
            self.cfg.GPT4O_MODEL,
            self.cfg.GPT4O_KEY,
            self.cfg.GPT4O_BASE
        )
        # DeepSeek V3 Model
        self.llm_manager.register_model(
            self.cfg.DEEPSEEKV3_MODEL,
            self.cfg.DEEPSEEKV3_KEY,
            self.cfg.DEEPSEEKV3_BASE
        )
        # Claude Model
        self.llm_manager.register_model(
            self.cfg.CLAUDE_MODEL,
            self.cfg.CLAUDE_KEY,
            self.cfg.CLAUDE_BASE
        )
        # Gemini Model
        self.llm_manager.register_model(
            self.cfg.GEMINI_MODEL,
            self.cfg.GEMINI_KEY,
            self.cfg.GEMINI_BASE
        )
        # FuLLM Model
        self.llm_manager.register_model(
            self.cfg.FULLM_MODEL,
            self.cfg.FULLM_KEY,
            self.cfg.FULLM_BASE
        )

    def init_data(self, src_pth: Path):
        """
        Initialize the dataset from the source path.
        
        Args:
            src_pth: Path to the source JSON file
        """
        # Load the JSON data
        with open(src_pth, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        
        # Create a mapping from dCap to index for quick lookup
        self.dcap_to_idx = {item['dCap']: idx for idx, item in enumerate(self.data)}

    def init_prompts(self, prompt_dir: Path):
        """
        Initialize the prompts for different experiment types.
        
        Args:
            prompt_dir: Path to the directory containing prompt files
        """
        self.prompt_dir = prompt_dir
        self.prompt_types = [
            "zero_shot",
            "zero_shot_cot",
            "one_shot"
        ]
        
        # Load all prompt files
        self.prompts = {}
        for prompt_file in prompt_dir.glob("*.txt"):
            self.prompts[prompt_file.stem] = prompt_file.read_text(encoding='utf-8')
        # Validate that we have the basic prompt types
        for prompt_type in self.prompt_types:
            prompt_name = f"{prompt_type}_prompt"
            if prompt_name not in self.prompts:
                logger.error(f"Missing prompt file: {prompt_name}.txt")
                raise ValueError(f"Missing prompt file: {prompt_name}.txt")

    def init_workflow(self, output_dir: Path, prompt_type: str = None):
        """
        Initialize the workflow for the experiments.
        
        Args:
            output_dir: Path to the output directory
            prompt_type: Type of prompt to use (if None, all types will be used)
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set the current prompt type
        self.current_prompt_type = prompt_type
        
        # Create subdirectories for each experiment type
        if prompt_type:
            exp_dir = self.output_dir / prompt_type
            exp_dir.mkdir(exist_ok=True)
        else:
            for pt in self.prompt_types:
                exp_dir = self.output_dir / pt
                exp_dir.mkdir(exist_ok=True)
        
        # Set data indices for processing
        self.data_indices = list(range(len(self.data)))

    def get_prompt_for_item(self, item: Dict[str, Any], prompt_type: str) -> str:
        """
        Get the appropriate prompt for the given item and prompt type.
        
        Args:
            item: The data item to process
            prompt_type: The type of prompt to use
            
        Returns:
            The formatted prompt text
        """
        dcap = item['dCap']
        prompt_name = f"{prompt_type}_prompt"
        
        # Check if there's a specific prompt for this dCap
        specific_prompt_name = f"{prompt_name}_{dcap}"
        if specific_prompt_name in self.prompts:
            prompt_template = self.prompts[specific_prompt_name]
        else:
            prompt_template = self.prompts[prompt_name]
        
        # Format the prompt with the item's input
        input_text = item['input']
        
        # Combine instruction and input for the prompt
        content = f"## 对话内容\n{input_text}"
        prompt_template = prompt_template.strip() + "\n" + "## 对话内容\n{content}"
        return prompt_template.format(content=content.strip()) # Need to call format; otherwise the '{{' and '}}' in our prompt will not be parsed

    async def process_single(self, idx: int):
        """
        Process a single data item with the current prompt type.
        
        Args:
            idx: Index of the data item
        """
        item = self.data[idx]
        dcap = item['dCap']
        
        # If no prompt type is specified, skip processing
        if not hasattr(self, 'current_prompt_type') or not self.current_prompt_type:
            logger.error("No prompt type specified")
            return
            
        prompt_type = self.current_prompt_type
        
        # Check if output file exists and has valid parsed response
        output_path = self.output_dir / prompt_type / f"{dcap}.json"
        if output_path.exists():
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_result = json.load(f)
                    if existing_result.get('parsed_response') is not None:
                        logger.info(f"Item {dcap} already processed successfully for {prompt_type}")
                        return
            except Exception as e:
                logger.warning(f"Error reading existing result for {dcap}: {e}")
        
        # Get the appropriate prompt
        prompt = self.get_prompt_for_item(item, prompt_type)
        
        # Generate response from LLM
        try:
            response = await self.llm_manager.generate(prompt, self.model_name)
            
            # Parse the JSON response
            parsed_response = None
            try:
                # Try to extract JSON from the response
                parsed_response = parsing_json(response)
                if not parsed_response:
                    norm_prompt = self.prompts["normalize_prompt"] + response
                    norm_response = await self.llm_manager.generate(norm_prompt, self.cfg.LOCAL_API_MODEL)
                    parsed_response = parsing_json(norm_response)
            except Exception as e:
                logger.error(f"Failed to parse response for {dcap} with {prompt_type}: {e}")
            
            # Save the result
            result = {
                **item,  # Include all original data
                "prompt": prompt,  # The prompt sent to LLM
                "raw_response": response,  # Raw LLM response
                "parsed_response": parsed_response,  # Parsed response (or None if parsing failed)
                "prompt_type": prompt_type  # Record which prompt type was used
            }
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Processed {dcap} with {prompt_type}")
            
        except Exception as e:
            logger.error(f"Error processing {dcap} with {prompt_type}: {e}")


    async def complete_callback(self, results):
        """
        Callback after all data is processed.
        
        Args:
            results: All results
        """
        logger.info(f"Completed processing all {len(results)} items with {self.current_prompt_type}")
        # Compile results after each prompt type is completed
        self.compile_results()

    def compile_results(self):
        """
        Compile results from all experiments into a single DataFrame.
        
        Returns:
            DataFrame containing all results
        """
        all_results = []
        
        for prompt_type in self.prompt_types:
            exp_dir = self.output_dir / prompt_type
            if not exp_dir.exists():
                continue
                
            for result_file in exp_dir.glob("*.json"):
                with open(result_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    
                # Add a summary row
                summary = {
                    "dCap": result["dCap"],
                    "prompt_type": prompt_type,
                    "parsed_successfully": result["parsed_response"] is not None,
                }
                
                # Add the parsed response fields if available
                if result["parsed_response"]:
                    for key, value in result["parsed_response"].items():
                        summary[f"Pred_{key}"] = value
                
                all_results.append(summary)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        df.to_excel(self.output_dir / "all_results.xlsx", index=False)
        
        return df


async def run_single_experiment(prompt_type, model_name=None, output_dir=None, batch_size=1700, concurrency=64):
    """
    Run a single experiment for a given prompt type.
    
    Args:
        prompt_type: Prompt type (zero_shot, zero_shot_cot, one_shot)
        model_name: Model name (if None, use default from config)
        output_dir: Output directory (if None, use default path)
        batch_size: Batch size
        concurrency: Concurrency
    """
    # Path settings
    src_pth = Path("data/wav_alpaca_dataset_dCap.jsonl")
    prompt_dir = Path("prompt")
    if output_dir is None:
        output_dir = Path("data/output/experiment_results")
    else:
        output_dir = Path(output_dir)
    
    # Initialize generator
    g = PromptExperimentGenerator(Config(), model_name=model_name)
    g.init_models()
    g.init_data(src_pth)
    g.init_prompts(prompt_dir)
    g.init_workflow(output_dir, prompt_type)
    
    # Run experiment
    await g.run(batch_size=batch_size, concurrency=concurrency)
    
    logger.info(f"Experiment {prompt_type} completed")


async def run_all_experiments(model_name=None, output_dir=None, batch_size=1700, concurrency=64):
    """
    Run experiments for all prompt types.
    
    Args:
        model_name: Model name (if None, use default from config)
        output_dir: Output directory (if None, use default path)
        batch_size: Batch size
        concurrency: Concurrency
    """
    # Path settings
    src_pth = Path("data/wav_alpaca_dataset_dCap.jsonl")
    prompt_dir = Path("prompt")
    if output_dir is None:
        output_dir = Path("data/output/experiment_results")
    else:
        output_dir = Path(output_dir)
    
    # Initialize generator
    g = PromptExperimentGenerator(Config(), model_name=model_name)
    g.init_models()
    g.init_data(src_pth)
    g.init_prompts(prompt_dir)
    
    # Run experiments for each prompt type
    for prompt_type in g.prompt_types:
        logger.info(f"Starting {prompt_type} experiment")
        g.init_workflow(output_dir, prompt_type)
        await g.run(batch_size=batch_size, concurrency=concurrency)
    
    # Compile final results
    g.compile_results()
    
    logger.info("All experiments completed")


async def reprocess_failed_items(prompt_type=None, model_name=None, output_dir=None, batch_size=1700, concurrency=64):
    """
    Reprocess failed parsing items
    
    Args:
        prompt_type: Prompt type (None means all types)
        model_name: Model name (if None, use default from config)
        output_dir: Output directory (if None, use default path)
        batch_size: Batch size
        concurrency: Concurrency
    """
    # Path settings
    src_pth = Path("data/wav_alpaca_dataset_dCap.jsonl")
    prompt_dir = Path("prompt")
    if output_dir is None:
        output_dir = Path("data/output/experiment_results")
    else:
        output_dir = Path(output_dir)
    
    # Initialize generator
    g = PromptExperimentGenerator(Config(), model_name=model_name)
    g.init_models()
    g.init_data(src_pth)
    g.init_prompts(prompt_dir)
    
    # Get all prompt types (if not specified)
    prompt_types = [prompt_type] if prompt_type else g.prompt_types
    
    # For each prompt type, find and reprocess failed items
    for pt in prompt_types:
        logger.info(f"Reprocessing {pt} failed items for")
        
        # Find all processed data items
        exp_dir = output_dir / pt
        if not exp_dir.exists():
            logger.warning(f"Directory does not exist: {exp_dir}")
            continue
            
        processed_files = list(exp_dir.glob("*.json"))
        
        # Check which parsing failed
        failed_indices = []
        for file_path in processed_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    if result.get("parsed_response") is None:
                        dcap = result["dCap"]
                        if dcap in g.dcap_to_idx:
                            idx = g.dcap_to_idx[dcap]
                            failed_indices.append(idx)
                        # Delete the file to allow reprocessing
                        file_path.unlink()
                        logger.info(f"Marked {dcap} for reprocessing with {pt} Reprocessing")
            except Exception as e:
                logger.error(f"Error checking {file_path} : {e}")
                # If the file cannot be read, mark for reprocessing
                dcap = file_path.stem
                if dcap in g.dcap_to_idx:
                    idx = g.dcap_to_idx[dcap]
                    failed_indices.append(idx)
                # Delete the file to allow reprocessing
                file_path.unlink()
                logger.info(f"Due to error, Marked {dcap} for reprocessing with {pt} Reprocessing")
        
        # Reprocess failed data items
        if failed_indices:
            g.init_workflow(output_dir, pt)
            g.data_indices = failed_indices
            await g.run(batch_size=batch_size, concurrency=concurrency)
        else:
            logger.info(f"No {pt} failed items for")
    
    # Compile results
    g.compile_results()
    


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run prompt experiments on wav_alpaca dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single experiment with specific prompt type
  python prompt_experiment_runner.py --prompt-type zero_shot --model gpt-4o --output-dir ./results
  
  # Run all experiments
  python prompt_experiment_runner.py --prompt-type all --model claude-3-5-sonnet
  
  # Reprocess failed items
  python prompt_experiment_runner.py --prompt-type zero_shot --reprocess --model gpt-4o
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--prompt-type", 
        choices=["zero_shot", "zero_shot_cot", "one_shot", "all"],
        required=True,
        help="Type of prompt experiment to run"
    )
    
    # Optional arguments
    parser.add_argument(
        "--model", 
        type=str,
        help="Model name to use (if not specified, uses default from config)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        help="Output directory for results (default: data/output/experiment_results)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1700,
        help="Batch size for processing (default: 1700)"
    )
    
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent requests (default: 1)"
    )
    
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Reprocess failed items instead of running new experiment"
    )
    
    return parser.parse_args()


async def main():
    """Main function - parse arguments and run experiments."""
    args = parse_args()
    
    if args.reprocess:
        # Reprocess failed items
        prompt_type = None if args.prompt_type == "all" else args.prompt_type
        await reprocess_failed_items(
            prompt_type=prompt_type,
            model_name=args.model,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            concurrency=args.concurrency
        )
    elif args.prompt_type == "all":
        # Run all experiments
        await run_all_experiments(
            model_name=args.model,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            concurrency=args.concurrency
        )
    else:
        # Run single experiment
        await run_single_experiment(
            prompt_type=args.prompt_type,
            model_name=args.model,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            concurrency=args.concurrency
        )


if __name__ == "__main__":
    asyncio.run(main()) 