"""
Data synthesis script - Generate new follow-up dialogue data based on follow-up dialogue content
"""
import json
import random
from pathlib import Path
from llmtoolkit.aioutils import BaseGenerator
from llmtoolkit.utils import Config, parsing_json
from loguru import logger


class DataSynthesisGenerator(BaseGenerator):
    """Data synthesis generator - Generate new follow-up dialogue content"""
    
    def init_models(self):
        """Initialize models and prompts"""
        self.qwen_model_name = self.cfg.LOCAL_QWEN_MODEL
        # Register model
        self.llm_manager.register_model(
            self.cfg.LOCAL_QWEN_MODEL,
            self.cfg.LOCAL_QWEN_KEY,
            self.cfg.LOCAL_QWEN_BASE,
            self.qwen_model_name,
        )
        # Register prompt template
        self.llm_manager.register_prompt(
            "data_synthesis",
            self.llm_manager.load_template("prompt/data_synthesis_prompt.txt")
        )
    
    def init_data(self, src_path: Path):
        """Initialize data"""
        self.src_path = src_path
        self.data = []
        
        # Read JSONL data
        with open(src_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line.strip()))
        
        # Set data indices to be processed
        self.data_indices = list(range(len(self.data)))
        logger.info(f"Loaded {len(self.data)} records from {src_path}")
    
    def init_workflow(self, output_dir: Path, dst_path: Path, num_iterations: int = 20):
        """Initialize workflow"""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dst_path = dst_path
        self.num_iterations = num_iterations
        self.current_iteration = 1
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Final output: {dst_path}")
        logger.info(f"Number of iterations: {num_iterations}")
    
    def generate_random_results(self):
        """Generate random follow-up report results"""
        oneself = random.choices(
            ["本人", "亲属"],
            weights=[0.6, 0.4],
            k=1
        )[0]

        death = random.choices(
            ["是", "否"],
            weights=[0.1, 0.9],
            k=1
        )[0]

        hospitalization = random.choices(
            ["是", "否", "未提及", "不确定"],
            weights=[0.2, 0.5, 0.2, 0.1],
            k=1
        )[0]

        surgery = random.choices(
            ["是", "否", "未提及", "不确定"],
            weights=[0.2, 0.5, 0.2, 0.1],
            k=1
        )[0]

        medication = random.choices(
            ["是", "否", "未提及", "不确定"],
            weights=[0.7, 0.1, 0.1, 0.1],
            k=1
        )[0]

        return hospitalization, surgery, medication, death, oneself
    
    def sample_input_text(self, input_text: str) -> str:
        """Randomly sample segments from input text"""
        lines = input_text.split("\n")
        indices = list(range(len(lines)))
        
        # Try 3 times to find suitable segments
        for _ in range(3):
            start = random.choice(indices)
            end_candidates = [i for i in indices if i > start + 5]
            
            if end_candidates:
                end = random.choice(end_candidates)
                return "\n".join(lines[start:end])
        
        # If no suitable segment found, return original text
        return input_text
    
    async def process_single(self, idx):
        """Process single data item"""
        iteration_dir = self.output_dir / f"iteration_{self.current_iteration}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        output_path = iteration_dir / f"{idx}.json"
        if output_path.exists():
            logger.info(f"Chunk {idx} already processed for iteration {self.current_iteration}")
            return
        
        item = self.data[idx]
        
        # Check data integrity
        if not item.get("input") or not item.get("output"):
            logger.error(f"Missing input or output in item {idx}: {item}")
            return
        
        # Generate random results
        hospitalization, surgery, medication, death, oneself = self.generate_random_results()
        
        # Build new output text
        if death == "是":
            new_output_text = f"随访信息来源: {oneself}\n受试者是否死亡: {death}"
        else:
            new_output_text = (
                f"随访信息来源: {oneself}\n"
                f"受试者是否死亡: {death}\n"
                f"受试者是否住院: {hospitalization}\n"
                f"受试者是否手术: {surgery}\n"
                f"受试者是否用药: {medication}"
            )
        
        # Sample segments from input text
        sampled_input = self.sample_input_text(item["input"])
        
        # Format prompt
        query = self.llm_manager.format_prompt(
            "data_synthesis",
            input_text=sampled_input,
            output_text=new_output_text
        )
        
        # Generate response
        response = await self.llm_manager.generate(
            query, 
            self.qwen_model_name,
            max_tokens=8192,
            temperature=1,
            top_p=0.4,
            frequency_penalty=0.5
        )
        
        if not response:
            logger.error(f"Failed to generate response for item {idx}")
            return
        
        # Create new data item
        processed_item = {
            "instruction": item.get("instruction", ""),
            "input": response,
            "output": new_output_text,
            "dCap": item.get("dCap", "")
        }
        
        # Save to cache file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_item, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Processed item {idx}")
    
    async def complete_callback(self, results):
        """Completion callback - Merge files from current iteration"""
        iteration_dir = self.output_dir / f"iteration_{self.current_iteration}"
        logger.info(f"Merging files from iteration {self.current_iteration}...")
        
        # Get all output files from current iteration and sort them
        all_output_files = list(iteration_dir.glob("*.json"))
        all_output_files.sort(key=lambda x: int(x.stem))
        
        iteration_data = []
        for output_file in all_output_files:
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    iteration_data.append(data)
            except Exception as e:
                logger.error(f"Failed to load {output_file}: {e}")
                continue
        
        # Save iteration results to JSONL format
        iteration_output = self.output_dir / f"iteration_{self.current_iteration}.jsonl"
        with open(iteration_output, 'w', encoding='utf-8') as f:
            for item in iteration_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(iteration_data)} synthesized records from iteration {self.current_iteration}")
    
    async def merge_all_iterations(self):
        """Merge all iterations into final output file"""
        logger.info("Merging all iterations into final output...")
        
        all_data = []
        for i in range(1, self.num_iterations + 1):
            iteration_file = self.output_dir / f"iteration_{i}.jsonl"
            if iteration_file.exists():
                with open(iteration_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            all_data.append(json.loads(line.strip()))
                logger.info(f"Loaded iteration {i} with {sum(1 for _ in open(iteration_file, 'r'))} records")
            else:
                logger.warning(f"Iteration file {iteration_file} not found")
        
        # Save final merged results
        with open(self.dst_path, 'w', encoding='utf-8') as f:
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Final merge complete: {len(all_data)} total synthesized records saved to {self.dst_path}")
    
    async def run_multiple_iterations(self, batch_size: int = 50, concurrency: int = 80):
        """Run data synthesis for multiple iterations"""
        for iteration in range(1, self.num_iterations + 1):
            self.current_iteration = iteration
            logger.info(f"Starting iteration {iteration}/{self.num_iterations}")
            
            # Run single iteration
            await self.run(batch_size=batch_size, concurrency=concurrency)
            
            logger.info(f"Completed iteration {iteration}/{self.num_iterations}")
        
        # Merge all iterations
        await self.merge_all_iterations()


async def main():
    """Main function"""
    # Configure paths
    src_path = Path("data/wav_alpaca_dataset_dCap.jsonl")
    dst_path = Path("data/wav_alpaca_dataset_dCap_synthesized.jsonl")
    output_dir = Path("data/cache/synthesis")
    
    # Configuration parameters
    num_iterations = 20
    batch_size = 50
    concurrency = 80
    
    # Initialize generator
    generator = DataSynthesisGenerator(Config())
    generator.init_models()
    generator.init_data(src_path)
    generator.init_workflow(output_dir, dst_path, num_iterations)
    
    # Run processing for multiple iterations
    await generator.run_multiple_iterations(batch_size=batch_size, concurrency=concurrency)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 