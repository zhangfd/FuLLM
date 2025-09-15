import asyncio
import json
import random
import re
import time
from abc import ABC, abstractmethod
from builtins import ValueError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import aiofiles
import aiohttp
from loguru import logger
from openai import AsyncOpenAI

from llmtoolkit.utils import Config, format_messages, parsing_json  # 注意不要循环导入了


class LLMManager:
    """
    LLM客户端管理器，用于管理多个语言模型客户端、提示词模板和配置。
    支持异步操作、提示词模板管理、工具调用和灵活的配置调整。
    """

    # 工具描述模板（英文）
    TOOL_DESCRIPTION_TEMPLATE_EN = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
"""

    # 工具描述模板（中文）
    TOOL_DESCRIPTION_TEMPLATE_CN = """# 工具

你可以调用一个或多个函数来协助回答用户的问题。

在<tools></tools> XML标签内提供了可用的函数签名：
<tools>
{tools}
</tools>

对于每个函数调用，请在<tool_call></tool_call> XML标签内返回一个包含函数名和参数的json对象：
<tool_call>
{{"name": <函数名>, "arguments": <参数json对象>}}
</tool_call>
"""

    def __init__(
        self,
        clients: Optional[Dict[str, AsyncOpenAI]] = None,
        timeout: float = 600.0,
        **kwargs,
    ):
        """
        初始化LLM管理器

        参数:
            clients: 模型名称到AsyncOpenAI客户端的映射字典
            timeout: API请求超时时间（秒），默认600秒
            **kwargs: 用于覆盖基础配置的参数
        """
        self.clients = clients if clients is not None else {}
        self.model_names = {}  # 新增：模型名称列表
        self.prompts = {}
        self.system_prompt = None  # 全局系统提示词
        self.tools = {}  # 工具注册表 {name: (description, function)}
        self.timeout = timeout  # API请求超时时间
        self.base_config = {
            # "max_tokens": 1024,
            # "top_p": 0.5,
            # "temperature": 0.2,
            # "frequency_penalty": 1.0,
            # "presence_penalty": 0.1
        }
        self.update_config(**kwargs)

    def set_timeout(self, timeout: float):
        """设置API请求超时时间"""
        self.timeout = timeout

    def register_prompt(self, name: str, template: str, **kwargs):
        """
        注册提示词模板及其可选的采样参数

        参数:
            name: 模板唯一标识符
            template: 包含格式化占位符的模板字符串
            **kwargs: 模板参数的可选值列表 加载准备好的json文件，然后解包式的输入
        """
        self.prompts[name] = {"template": template}
        self.prompts[name].update(kwargs)

    def register_model(self, model, api_key, api_base, model_alias=None, timeout=600):
        """
        注册新的LLM客户端
        注意： `model` 或者 `model_alias` 不能重复  重复的话会报错
        参数:
            model: 模型标识符
            api_key: API认证密钥
            api_base: API基础URL
            model_alias: 模型别名（可选）
            timeout: 请求超时时间（默认600秒）,reasoning 模型建议1500以上
        """

        def check_model_name(model_name):
            if model_name in self.model_names:
                logger.error(f"Model {model_name} already registered")
                return False
            return True

        def check_clients(model_name):
            if model_name in self.clients:
                logger.error(f"client {model_name} already registered")
                return False
            return True

        if model_alias is not None:
            # model_alias = model  # TODO: 保存model_name 在generate的时候调用 否则传的还是旧的model_name
            # model = model_alias
            check_status = check_model_name(model_alias) or check_clients(model_alias)
            if not check_status:
                return
            self.model_names.update({model_alias: model})
            self.clients[model_alias] = AsyncOpenAI(api_key=api_key, base_url=api_base, timeout=timeout)
            logger.info(f"Registered model {model} with alias {model_alias} successfully")
        else:
            check_status = check_model_name(model_alias) or check_clients(model_alias)
            if not check_status:
                return
            self.model_names.update({model: model})
            self.clients[model] = AsyncOpenAI(api_key=api_key, base_url=api_base, timeout=timeout)

            logger.info(f"Registered model {model} successfully")

    def load_template(self, template_path: str | Path):
        """
        从文件加载提示词模板

        参数:
            template_path: 模板文件路径

        返回:
            str: 模板文件内容

        异常:
            FileNotFoundError: 模板文件不存在时抛出
        """
        if isinstance(template_path, str):
            template_path = Path(template_path)
        if not isinstance(template_path, Path) or not template_path.exists():
            logger.error("no template file")
            raise FileNotFoundError(f"prompt文件路径不存在:{template_path!s}")
        return template_path.read_text()

    def format_prompt(self, name: str, **kwargs):
        """
        格式化已注册的提示词模板

        参数:
            name: 已注册的模板名称
            **kwargs: 模板参数值

        返回:
            str: 格式化后的提示词字符串
        """
        prompt_data = self.prompts[name]
        template = prompt_data["template"]

        format_kwargs = {}
        for key, value in prompt_data.items():
            if key != "template" and isinstance(value, list) and key not in kwargs:
                format_kwargs[key] = random.choice(value)

        format_kwargs.update(kwargs)
        return template.format(**format_kwargs)

    def get_prompt_template(self, prompt_name):
        """返回指定名称的提示词模板"""
        return self.prompts[prompt_name]["template"]

    def get_all_prompts(self):
        """返回所有已注册的提示词模板名称列表"""
        return list(self.prompts)

    def get_all_models(self):
        """返回所有已注册的模型和名称映射列表"""
        return self.model_names

    def update_config(self, **kwargs):
        """使用提供的参数更新基础配置"""
        for k, v in kwargs.items():
            self.base_config[k] = v

    def set_system_prompt(self, system_prompt: str):
        """设置全局系统提示词"""
        self.system_prompt = system_prompt

    def register_tool(self, name: str, description: Dict, function: Callable):
        """注册工具

        Args:
            name: 工具名称
            description: 工具描述（JSON格式）
            function: 工具函数
        """
        self.tools[name] = (description, function)

    def get_tool_description(self, language: str = "zh") -> str:
        """获取工具描述

        Args:
            language: 语言选择 ("en" 或 "cn")

        Returns:
            str: 格式化后的工具描述
        """
        # 将所有工具描述合并为一个JSON数组
        tool_descriptions = [desc for desc, _ in self.tools.values()]
        tools_json = json.dumps(tool_descriptions, ensure_ascii=False)

        # 选择模板
        template = self.TOOL_DESCRIPTION_TEMPLATE_EN if language == "en" else self.TOOL_DESCRIPTION_TEMPLATE_CN

        return template.format(tools=tools_json)

    async def _parse_tool_calls(self, content: str) -> List[Dict]:
        """解析工具调用

        Args:
            content: 模型输出内容

        Returns:
            List[Dict]: 解析后的工具调用列表，每个元素包含 name 和 arguments
        """
        tool_calls = []
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            try:
                tool_call = json.loads(match.group(1))
                if isinstance(tool_call, dict) and "name" in tool_call and "arguments" in tool_call:
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call: {match.group(1)}")
                continue

        return tool_calls

    async def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """执行工具调用

        Args:
            tool_calls: 工具调用列表，每个元素包含 name 和 arguments

        Returns:
            List[Dict]: 工具执行结果列表
        """
        results = []
        for tool_call in tool_calls:
            try:
                name = tool_call["name"]
                arguments = tool_call["arguments"]

                if name in self.tools:
                    _, function = self.tools[name]
                    result = (
                        await function(**arguments) if asyncio.iscoroutinefunction(function) else function(**arguments)
                    )
                    results.append({"name": name, "arguments": arguments, "result": result})
            except Exception as e:
                logger.error(f"Error executing tool {name}: {e}")
                continue

        return results

    @abstractmethod
    def init_workflow(self):
        """初始化工作流
        在这里实现:
        - 输出路径设置
        - 处理参数设置
        - data_indices设置等 我们的数据要按照List[Any]准备，建议是初始化成List[int]，这样方便我们重启任务，或者是就是一条条的处理数据
        """
        pass

    # ruff: noqa: PLR0915, PLR0912
    async def generate(
        self,
        query: str,
        model_name: str,
        force_json: bool = False,
        default_on_timeout: str = "",
        default_on_error: Optional[str] = None,
        retry_count: int = 0,
        retry_sleep_time: float = 15.0,
        return_dict: bool = False,
        execute_tools: bool = True,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Union[str, Dict[str, str]]:
        """
        生成LLM响应，支持工具调用。仅支持单次对话

        参数:
            query: 用户查询/提示词
            model_name: 使用的模型名称 or 模型别名
            force_json: 是否强制返回JSON格式
            default_on_timeout: 超时时返回的默认值
            default_on_error: 错误时返回的默认值
            retry_count: 重试次数
            retry_sleep_time: 重试间隔时间（秒）
            return_dict: 是否返回字典格式（包含content和messages）
            execute_tools: 是否执行检测到的工具调用
            timeout: 本次请求的超时时间（秒），如果不指定则使用模型注册时的超时时间
            **kwargs: 其他参数，将传递给OpenAI API
        """
        if model_name not in self.clients:
            raise ValueError(f"Model {model_name} not registered")

        if default_on_error is None:
            default_on_error = default_on_timeout

        real_model_name = self.model_names.get(model_name, "")

        # 如果指定了本次请求的超时时间，创建一个临时客户端
        client = self.clients[model_name]
        if timeout is not None and timeout != self.timeout:
            # 获取原始客户端的配置
            api_key = client.api_key
            base_url = client.base_url
            # 创建临时客户端
            temp_client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
            client = temp_client

        async def _make_request() -> Optional[Union[str, Dict[str, str]]]:
            try:
                # 构建消息列表
                messages = []

                # 处理系统提示词
                if self.system_prompt:
                    system_content = self.system_prompt
                    # 如果有工具且需要执行，将工具描述添加到系统提示词
                    if execute_tools and self.tools:
                        tool_description = self.get_tool_description()
                        system_content = f"{system_content}\n\n{tool_description}"
                    messages.append({"role": "system", "content": system_content})

                # 添加用户消息
                messages.extend(format_messages(query))

                # 获取响应
                response = await client.chat.completions.create(
                    model=real_model_name,
                    messages=messages,
                    response_format={"type": "json_object"} if force_json else None,
                    **{**self.base_config, **kwargs},
                )

                # 解析响应
                raw_result = response.choices[0].message
                content = raw_result.content
                tool_results = []

                if execute_tools and self.tools:
                    # 检查是否有服务端解析的工具调用
                    server_tool_calls = raw_result.tool_calls if hasattr(raw_result, "tool_calls") else []

                    if server_tool_calls:
                        # 转换服务端解析的工具调用格式
                        tool_calls = []
                        for tool_call in server_tool_calls:
                            function_call = tool_call.get("function", {})
                            tool_calls.append(
                                {
                                    "name": function_call.get("name"),
                                    "arguments": function_call.get("arguments", {}),
                                }
                            )
                    else:
                        # 从content中解析工具调用
                        tool_calls = await self._parse_tool_calls(content)

                    if tool_calls:
                        # 执行工具调用
                        tool_results = await self._execute_tool_calls(tool_calls)

                        # 格式化工具调用为消息内容，不能使用f-string，否则会因为双重转义出错
                        formatted_calls = [
                            "<tool_call>" + json.dumps(call, ensure_ascii=False) + "</tool_call>" for call in tool_calls
                        ]

                        # 更新content
                        if content:
                            joined_calls = "\n".join(formatted_calls)
                            content = f"{content}\n\n{joined_calls}"
                        else:
                            content = "\n".join(formatted_calls)

                # 如果有工具调用结果，进行第二轮对话
                if tool_results:
                    # 更新消息历史：添加原始助手回复
                    messages.append({"role": "assistant", "content": content})

                    # 将所有工具结果合并为一个observation消息，此处不应该再dumps了，否则会被多次dumps，数据就无法loads了
                    tool_responses = [
                        "<tool_response>" + result["result"] + "</tool_response>" for result in tool_results
                    ]

                    messages.append({"role": "observation", "content": "\n".join(tool_responses)})

                    # 进行第二轮对话
                    second_response = await client.chat.completions.create(
                        model=real_model_name,
                        messages=messages,
                        response_format={"type": "json_object"} if force_json else None,
                        **{**self.base_config, **kwargs},
                    )
                    second_result = second_response.choices[0].message
                    content = second_result.content

                    # 将第二轮对话的回复添加到消息历史
                    messages.append({"role": "assistant", "content": content})

                # 根据return_dict参数决定返回格式
                if return_dict:
                    response_dict = {"content": content, "messages": messages}
                    if hasattr(raw_result, "reasoning_content") and raw_result.reasoning_content:
                        response_dict["reasoning_content"] = raw_result.reasoning_content
                    return response_dict
                else:
                    return content

            except Exception as e:
                logger.error(f"API Error: {e!s}")
                return None

        # 执行重试逻辑
        for attempt in range(retry_count + 1):
            try:
                result = await _make_request()
                if result:
                    logger.debug(f"Query success on attempt {attempt + 1} for query: {query[:min(30, len(query))]}...")
                    return result

                if attempt < retry_count:
                    logger.warning(f"Empty response, retrying... (attempt {attempt + 1})")
                    await asyncio.sleep(retry_sleep_time)
                    continue

            except Exception as e:
                logger.error(f"Retry error: {e!s}")
                if attempt < retry_count:
                    await asyncio.sleep(retry_sleep_time)
                    continue

        logger.error(f"All {retry_count + 1} attempts failed for query: {query[:100]}...")
        return {"content": default_on_error} if return_dict else default_on_error


class BatchProcessor:
    """异步批处理器，支持批量处理数据并控制并发

    功能:
    - 对数据列表进行批量异步处理
    - 控制并发数量
    - 支持批处理和完成时的回调函数
    示例：
    async def process_single(item):
        await asyncio.sleep(0.1)  # 模拟处理
        return f"Processed {item}"

    async def batch_callback(results):
        print(f"Batch done: {len(results)} items")

    async def complete_callback(results):
        print(f"All done: {len(results)} items")

    processor = BatchProcessor(
        data_list=list(range(100)),
        process_single_fn=process_single,
        batch_callback=batch_callback,
        complete_callback=complete_callback
    )

    results = await processor.run(batch_size=10, concurrency=3)
    """

    def __init__(
        self,
        data_list: List[Any],
        process_single_fn: Callable,
        batch_callback: Optional[Callable] = None,
        complete_callback: Optional[Callable] = None,
    ):
        """
        Args:
            data_list: 待处理的数据列表
            process_single_fn: 处理单条数据的异步函数
            batch_callback: 每批数据处理完成后的回调函数，接收batch_results作为参数
            complete_callback: 所有数据处理完成后的回调函数，接收所有results作为参数
        """
        self.data_list = data_list
        self.process_single = process_single_fn
        self.batch_callback = batch_callback
        self.complete_callback = complete_callback

    async def process_batch(self, batch: List[Any], semaphore: asyncio.Semaphore) -> List[Any]:
        """处理单批数据，使用信号量控制并发

        Args:
            batch: 单批待处理数据
            semaphore: 控制并发的信号量

        Returns:
            批处理结果列表，顺序与输入batch对应
        """

        async def _process_item(item: Any):
            async with semaphore:
                return await self.process_single(item)

        return await asyncio.gather(*[_process_item(item) for item in batch])

    async def run(self, batch_size: int = 32, concurrency: int = 10) -> List[Any]:
        """运行批处理

        Args:
            batch_size: 每批处理的数据量
            concurrency: 最大并发数

        Returns:
            所有数据的处理结果列表，顺序与输入data_list对应

        Note:
            - 错误处理由process_single_fn负责
            - batch_callback在每批数据处理完成后调用
            - complete_callback在所有数据处理完成后调用
        """
        semaphore = asyncio.Semaphore(concurrency)
        results = []

        for i in range(0, len(self.data_list), batch_size):
            batch = self.data_list[i : i + batch_size]
            batch_results = await self.process_batch(batch, semaphore)
            results.extend(batch_results)

            if self.batch_callback:
                await self.batch_callback(batch_results)

        if self.complete_callback:
            await self.complete_callback(results)

        return results


class BaseGenerator(ABC):
    """生成器基类

    定义标准的初始化和处理流程:
    1. 数据初始化
    2. 模型和提示词初始化
    3. 工作流(路径等)初始化
    4. 数据处理流程
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.llm_manager = LLMManager()

        # 三阶段初始化 这个还是得留给子类自己去调用
        # self.init_data()
        # self.init_models()
        # self.init_workflow()

    @abstractmethod
    def init_data(self):
        """初始化数据
        在这里实现:
        - 数据加载
        - 数据预处理
        - 数据集初始化等
        """
        pass

    @abstractmethod
    def init_models(self):
        """初始化模型和提示词
        在这里实现:
        - 模型注册
        - 提示词加载
        - 提示词模板注册等
        """
        pass

    @abstractmethod
    def init_workflow(self):
        """初始化工作流
        在这里实现:
        - 输出路径设置
        - 处理参数设置
        - data_indices设置等 我们的数据要按照List[Any]准备，建议是初始化成List[int]，这样方便我们重启任务，或者是就是一条条的处理数据
        """
        pass

    @abstractmethod
    async def process_single(self, idx: int):
        """处理单条数据
        在这里实现具体的生成逻辑
        """
        pass

    async def batch_callback(self, batch: List[Any]):
        """批处理回调
        每个batch处理完成后调用
        """
        return None

    async def complete_callback(self, results: List[Any]):
        """完成回调
        所有数据处理完成后调用
        """
        return None

    async def run(self, batch_size: int = 1, concurrency: int = 1):
        """运行生成流程"""
        if not hasattr(self, "data_indices"):
            raise ValueError("请在init_workflow中设置data_indices")

        processor = BatchProcessor(
            data_list=self.data_indices,
            process_single_fn=self.process_single,
            batch_callback=self.batch_callback,
            complete_callback=self.complete_callback,
        )
        start = time.perf_counter()
        await processor.run(batch_size=batch_size, concurrency=concurrency)
        elapsed = time.perf_counter() - start
        logger.info(f"All done, total time: {elapsed:.2f}s")


async def save_json(data: dict, filepath: str):
    async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
        json_str = json.dumps(data, ensure_ascii=False, indent=4)
        await f.write(json_str)


async def load_json(filepath):
    async with aiofiles.open(filepath, encoding="utf-8", mode="r") as file:
        content = await file.read()
        if not content:
            return {}
        else:
            return json.loads(content)


async def call_llm_api_async(
    session: aiohttp.ClientSession,
    messages: Union[str, List[str]],
    url: str,
    model_name: str,
    authorization: str,
    return_str=True,
    timeout: float = 600.0,
    **kwargs,
):
    """异步调用LLM API

    参数:
        session: aiohttp客户端会话
        messages: 消息列表或单条消息
        url: API端点URL
        model_name: 模型名称
        authorization: 认证令牌
        return_str: 是否返回字符串（否则返回完整响应）
        timeout: 请求超时时间（秒），默认600秒
        **kwargs: 其他参数，将传递给API

    返回:
        根据return_str参数返回字符串或完整响应
    """
    formatted_messages = format_messages(messages)

    default_params = {
        "stream": False,
        "max_tokens": 4000,
        "temperature": 0.7,
        "top_p": 0.4,
        "frequency_penalty": 0.5,
        "n": 1,
    }
    default_params.update(kwargs)

    payload = {"model": model_name, "messages": formatted_messages, **default_params}

    headers = {
        "Authorization": f"Bearer {authorization}" if authorization else "",
        "Content-Type": "application/json",
    }

    try:
        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            response.raise_for_status()
            res_json = await response.json()
            if return_str:
                try:
                    return res_json["choices"][0]["message"]["content"]
                except KeyError:
                    return res_json
            return res_json
    except (asyncio.TimeoutError, aiohttp.ClientError, Exception):
        return None


async def parsing_and_fix_json(query, desired_type, llm_generate, prompt=None):
    if not prompt:
        prompt = """你是一个专业的JSON格式化和修复专家。请严格按照以下要求处理：

1. 输入的JSON字符串可能存在以下常见错误：
   - 缺少引号
   - 多余的逗号
   - 不一致的引号类型
   - 转义字符错误
   - 缩进或换行问题
   - 多余的注释
   - 错误的截断

2. 修复规则：
   - 使用英文双引号("")
   - 确保所有键都被双引号包裹
   - 删除多余的逗号
   - 正确转义特殊字符
   - 保持原始数据结构和内容不变
   - 生成标准的、可解析的JSON
   - 为错误的截断补充必要的、空值数据

3. 如果原始字符串无法通过轻微修改转换为有效JSON，请返回空字符串

原始JSON字符串:
{query}

请仅返回修复后的、完全正确的JSON字符串，不要添加任何额外解释或注释。
"""
    parsed_json = parsing_json(query)
    if not isinstance(parsed_json, desired_type):
        fix_query = await llm_generate(prompt.format(query=query), max_tokens=2048)
        return parsing_json(fix_query)
    else:
        return parsed_json
