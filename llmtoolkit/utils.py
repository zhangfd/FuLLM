import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import demjson3
import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger


class Config:
    """
    配置管理类，支持：
    1. 直接指定 .env 路径初始化
    2. 自动逐层向上查找 .env
    3. 通过属性方式访问配置
    """

    def __init__(self, env_path: Optional[Union[str, Path]] = None):
        """
        初始化配置

        Args:
            env_path: .env 文件路径，如果不指定则自动查找
        """
        self._config = {}

        if env_path:
            self._load_env(Path(env_path))
        else:
            self._find_and_load_env()

    def _load_env(self, env_path: Path) -> None:
        """加载指定的 .env 文件"""
        logger.debug(f"loading env file from: {env_path!s}")
        if not env_path.exists():
            raise FileNotFoundError(f"找不到配置文件：{env_path}")
        load_dotenv(env_path, override=True)
        # 将环境变量加载到实例字典中
        self._config = {key: value for key, value in os.environ.items()}

    def _find_and_load_env(self) -> None:
        """逐层向上查找并加载 .env 文件"""
        current = Path.cwd()

        while current != current.parent:
            env_path = current / ".env"
            if env_path.exists():
                self._load_env(env_path)
                return
            current = current.parent

        raise FileNotFoundError("在当前目录及其父级目录中找不到 .env 文件")

    def __getattr__(self, name: str) -> Any:
        """允许通过属性方式访问配置"""
        env_name = name.upper()  # 约定：环境变量使用大写

        if env_name not in self._config:
            raise AttributeError(f"配置项 '{name}' 不存在")

        return self._config[env_name]

    def get(self, name: str, default: Any = None) -> Any:
        """通过方法获取配置，支持默认值"""
        return self._config.get(name.upper(), default)

    def require(self, name: str) -> Any:
        """获取必需的配置项，如果不存在则抛出异常"""
        value = self.get(name)
        if value is None:
            raise ValueError(f"必需的配置项 '{name}' 未设置")
        return value

    @property
    def all(self) -> dict:
        """获取所有配置项"""
        return self._config.copy()

def split_sentences(para: str) -> List[str]:
    """
    将段落按句子分隔符拆分为句子列表。
    该函数处理中文和英文的标点符号，将段落中的句子拆分为单独的句子，并返回一个句子列表。
    Args:
        para (str): 输入的段落文本。
    Returns:
        List[str]: 拆分后的句子列表。
    """
    # 单字符断句符
    para = re.sub(r"([。！？\?])([^”’])", r"\1\n\2", para)
    # 英文省略号
    para = re.sub(r"(\.{6})([^”’])", r"\1\n\2", para)
    # 中文省略号
    para = re.sub(r"(\…{2})([^”’])", r"\1\n\2", para)
    # 处理中文标点符号后的引号
    para = re.sub(r"([。！？\?][”’])([^，。！？\?])", r"\1\n\2", para)
    # 处理英文标点符号
    para = re.sub(r"([;:](?!$))([^”’])", r"\1\n\2", para)  # 分号和冒号后面如果是句子开头则换行
    para = re.sub(r"([。！？\?][”’])([a-zA-Z])", r"\1\n\2", para)  # 双引号后跟英文字符则换行
    # 去掉段尾多余的空格或换行符
    para = para.rstrip()
    return [line for line in para.split("\n") if line.strip() != ""]


def parsing_json(text, multi_match=False):
    """从文本中提取并解析 JSON，支持嵌套结构

    Args:
        text: 包含 JSON 的文本字符串

    Returns:
        解析后的 JSON 对象，如果解析失败返回 None
    """
    if not isinstance(text, str):
        return None
    # 首先尝试匹配 ```json ... ``` 格式
    code_block_pattern = r"```json(.*?)```"
    code_matches = re.findall(code_block_pattern, text, re.DOTALL)

    # 尝试解析代码块中的 JSON
    parsed_results = []
    for match in code_matches:
        try:
            matched_json = demjson3.decode(match.strip())
            parsed_results.append(matched_json)
        except demjson3.JSONDecodeError:
            continue

    if parsed_results:
        return parsed_results if multi_match else parsed_results[0]

    # 如果代码块解析失败，尝试匹配一般的 JSON 格式
    def find_matching_bracket(s, start_pos, open_char, close_char):
        """查找匹配的括号位置"""
        count = 1
        pos = start_pos
        while pos < len(s) and count > 0:
            if s[pos] == open_char:
                count += 1
            elif s[pos] == close_char:
                count -= 1
            pos += 1
        return pos if count == 0 else -1

    def extract_json_candidates(text):
        """提取可能的 JSON 字符串"""
        candidates = []
        i = 0
        while i < len(text):
            if text[i] == "{":
                end_pos = find_matching_bracket(text, i + 1, "{", "}")
                if end_pos != -1:
                    candidates.append(text[i:end_pos])
                    i = end_pos
                else:
                    i += 1
            elif text[i] == "[":
                end_pos = find_matching_bracket(text, i + 1, "[", "]")
                if end_pos != -1:
                    candidates.append(text[i:end_pos])
                    i = end_pos
                else:
                    i += 1
            else:
                i += 1
        return candidates

    # 获取所有可能的 JSON 字符串
    candidates = extract_json_candidates(text)

    # 尝试解析每个候选字符串
    for candidate in candidates:
        # 首先尝试用标准 json 库解析
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # 如果标准解析失败，尝试用 demjson3
            try:
                return demjson3.decode(candidate)
            except Exception:
                continue

    return None


def format_messages(content: Union[str, List[str]]) -> List[dict]:
    """
    Format the input content into a list of message dictionaries.

    Args:
        content (Union[str, List[str]]): A string or a list of strings.

    Returns:
        List[dict]: A list of message dictionaries.

    Raises:
        ValueError: If the content is not a string or a list of strings, or if the list length is even.
    """
    if isinstance(content, str):
        return [{"role": "user", "content": content}]

    if not isinstance(content, list):
        raise ValueError("Content must be either a string or a list of strings.")

    if len(content) % 2 == 0:
        raise ValueError("When providing a list, its length should be odd. The last message should be from the user.")

    messages = []
    for i, message in enumerate(content):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": message})

    return messages


def call_llm_api(
    messages: Union[str, List[str]],
    url: str,
    model_name: str,
    authorization: str,
    return_str=True,
    **kwargs,
):
    """
    Call the LLM API with the given parameters.

    Args:
        messages (Union[str, List[str]]): A string or a list of strings representing the conversation.
        url (str): API endpoint URL.
        model_name (str): Name of the model to use.
        authorization (str): Authorization token.
        **kwargs: Additional parameters to customize the API call.

    Returns:
        dict: API response as a Python object.

    Raises:
        requests.exceptions.RequestException: If the API call fails.
    """
    # Format messages
    formatted_messages = format_messages(messages)

    # Default parameters
    default_params = {
        "stream": False,
        "max_tokens": 4000,
        "temperature": 0.7,
        "top_p": 0.4,
        "frequency_penalty": 0.5,
        "n": 1,
    }

    # Update default parameters with any provided kwargs
    default_params.update(kwargs)

    # Construct the payload
    payload = {"model": model_name, "messages": formatted_messages, **default_params}

    # Set up headers
    headers = {
        "Authorization": f"Bearer {authorization}",
        "Content-Type": "application/json",
    }

    # Make the API call
    response = requests.post(url, json=payload, headers=headers)

    # Check for successful response
    response.raise_for_status()
    res_json = response.json()
    if return_str:
        try:
            return res_json["choices"][0]["message"]["content"]
        except KeyError:
            return res_json
    # Parse and return the JSON response
    return res_json


def save_to_jsonl(data: List[Dict], output_path: str | Path, encoding: str = "utf-8") -> None:
    """
    Save a list of dictionaries to a JSONL file.

    Args:
        data: List of dictionaries to save
        output_path: Path to save the JSONL file
        encoding: File encoding (default: utf-8)
    """
    # Create directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write to JSONL file
    with open(output_path, "w", encoding=encoding) as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")


if __name__ == "__main__":
    ans = call_llm_api("query", "url", "model_name", "authorization")
