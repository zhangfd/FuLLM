"""
Data rewrite script - Optimize follow-up dialogue content based on follow-up report results
"""
import json
from pathlib import Path
from llmtoolkit.aioutils import BaseGenerator
from llmtoolkit.utils import Config, parsing_json
from loguru import logger


class DataRewriteGenerator(BaseGenerator):
    """Data rewrite generator - Optimize follow-up dialogue content"""
    
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
            "data_rewrite",
            self.llm_manager.load_template("prompt/data_rewrite_prompt.txt")
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
        
        # Format prompt
        query = self.llm_manager.format_prompt(
            "data_rewrite",
            input_text=item["input"],
            output_text=item["output"]
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
        
        # Update data
        processed_item = item.copy()
        processed_item["input"] = response
        
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
        
        logger.info(f"Saved {len(iteration_data)} rewritten records from iteration {self.current_iteration}")
    
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
        
        logger.info(f"Final merge complete: {len(all_data)} total rewritten records saved to {self.dst_path}")
    
    async def run_multiple_iterations(self, batch_size: int = 50, concurrency: int = 30):
        """Run data rewrite for multiple iterations"""
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
    dst_path = Path("data/wav_alpaca_dataset_dCap_rewritten.jsonl")
    output_dir = Path("data/cache/rewrite")
    
    # Configuration parameters
    num_iterations = 20
    batch_size = 50
    concurrency = 30
    
    # Initialize generator
    generator = DataRewriteGenerator(Config())
    generator.init_models()
    generator.init_data(src_path)
    generator.init_workflow(output_dir, dst_path, num_iterations)
    
    # Run processing for multiple iterations
    await generator.run_multiple_iterations(batch_size=batch_size, concurrency=concurrency)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 