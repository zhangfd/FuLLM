# A Large Language Model For Clinical Outcomes Adjudication from Telephone Follow-up Interviews: A Secondary Analysis of a Multicenter Randomized Clinical Trial

This repository contains the official source code and experimental setup for the paper "A Large Language Model For Clinical Outcomes Adjudication from Telephone Follow-up Interviews: A Secondary Analysis of a Multicenter Randomized Clinical Trial".

## Project Structure

```
./
├── config/                 # Configuration files for training and inference
├── data/                   # Directory for input and generated data
├── llmtoolkit/             # Local dependency for asynchronous model inference
├── prompt/                 # Chinese prompts for experiments
├── prompt_en/              # English translations of prompts
└── scripts/                # Shell scripts to run the entire workflow
```

## Environment and Installation

### Environment

This project is developed and tested under Python 3.10 on Ubuntu 22.04.

### H20 GPU Bug Fix

If you are using NVIDIA H20 GPUs, you may encounter a "Floating point exception" error. This is a known issue that can be resolved by installing a specific version of CUDA libraries. This hardware was used in our paper.
```bash
pip install nvidia-cublas-cu12==12.4.5.8
```

### Installation

1.  **Install LLaMA-Factory**
    The training and prediction scripts depend on `LLaMA-Factory`. Please install it first:
    ```bash
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory
    pip install -e ".[torch,metrics]" --no-build-isolation
    cd ..
    ```

2.  **Install Project Dependencies**
    Clone this repository and install the necessary packages:
    ```bash
    git clone [repository-url]
    cd [repository-name]
    pip install -e .
    ```
3.  **Install Git LFS**
    This project uses Git LFS to manage large model files. Please make sure you have Git LFS installed.
    ```bash
    git lfs install
    ```
    After cloning the repository, you may need to pull the LFS files separately:
    ```bash
    git lfs pull
    ```

## Data Preparation

The project requires two main input files to be placed in the `data/` directory:

1.  `data/wav_alpaca_dataset_dCap.jsonl`: This file should contain the primary dataset in JSON Lines format. Each line represents a single data entry and should follow a structure similar to this:
    ```json
    {
      "instruction": "prompt",
      "input": "chat_content", 
      "output": "parsed_result",
      "dCap": "id"
    }
    ```

2.  `data/meta_info.json`: This file contains metadata associated with the dataset. The structure should be as follows:
    ```json
      "dCap_id": {
          "silver": { ... },
          "gold": { ... },
          "exam_type_id": 1,
          "reader": { ... },
          "fold_id": int,
          "hospital_code": "str"
      }
    ```

Please format your custom data accordingly.

## Configuration

Before running the workflow, you need to configure your environment. Create a `.env` file in the root of the project (you can copy `.env.example`) and specify your model paths and any other necessary environment variables.

## Usage

The entire experimental workflow is orchestrated through a series of shell scripts. Please run them in the following order to reproduce the results.

1.  **Data Preprocessing**
    This script preprocesses the raw data. It performs rewriting, synthesis, filtering, and splits the data into folds for cross-validation.
    ```bash
    bash scripts/run_data_preproc.sh
    ```

2.  **SVM Experiments**
    This script runs the baseline SVM models with TF-IDF and Word2Vec features.
    ```bash
    bash scripts/run_svm.sh
    ```

3.  **Train and Predict with LLM**
    This script fine-tunes the Large Language Model using 5-fold cross-validation. It handles both training and prediction phases.
    ```bash
    bash scripts/run_train_predict.sh
    ```

4.  **Model Inference**
    This script runs inference with various pre-trained models using different prompting strategies (e.g., zero-shot, few-shot).
    ```bash
    bash scripts/run_model_inference.sh
    ```

5.  **Data Postprocessing and Evaluation**
    Finally, this script collects all the results, performs evaluation, and generates the figures and tables presented in the paper.
    ```bash
    bash scripts/run_data_postproc.sh
    ```

After running all the steps, you will find all the generated data, including figures and tables, in the project directory.

## Standalone Inference

Here are instructions for running standalone inference, either through an interactive web UI or in a batch process.

### Using the Web UI for Interactive Chat

For interactive testing and demonstration, you can launch a web-based chat interface.

1.  **Start the Web UI**

    Run the following command to start the Gradio web server:
    ```bash
    llamafactory-cli webchat config/train_predict/fullm_api.yaml
    ```

2.  **Using the Chat Interface**

    Once the server is running, open the provided URL in your browser. For optimal results, you should prepend the content of the `prompt/zero_shot_prompt.txt` file to your message in the chat input box.

### Batch Inference with the OpenAI-style API

For processing multiple data points programmatically, you can use the following command.
```bash
llamafactory-cli train config/train_predict/fullm_predict.yaml
```
The results will be saved in the specified output directory.

## Citation

Our paper is currently under review. We will update this section with the full citation once it is published.

## License


## Contact

