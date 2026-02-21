"""Core AI Agent for CFST Data Extraction."""

from pathlib import Path

from pydantic_ai import Agent, RunContext

from cfst_extractor.agent.models import PaperExtraction
from cfst_extractor.agent.tools import (
    execute_python_calc,
    inspect_image,
    list_directory_files,
    read_markdown,
)

# 读取 System Prompt
# 动态指向到根目录外层的 config 文件夹下的 System_Prompt.md
PROMPT_PATH = Path(__file__).resolve().parents[4] / "config" / "System_Prompt.md"
if PROMPT_PATH.exists():
    SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")
else:
    print(f"WARNING: 找不到 Prompt 文件 {PROMPT_PATH}")
    SYSTEM_PROMPT = "你是一个专门从钢管混凝土（CFST）科学论文中提取试验数据的专家。"

import json
import os

import httpx
from pydantic_ai import Agent, RunContext

# --- 针对中转 API 的 $defs 解析补丁 ---
# 很多 OpenAI 到 Gemini 的中转 API 不支持 JSON Schema 中的 $defs / $ref 嵌套引用
# 我们在此拦截底层 HTTP 请求，将嵌套的 Schema 在发送前就地“展平” (inline)。
original_send = httpx.AsyncClient.send

async def _patched_send(self, request: httpx.Request, **kwargs):
    if request.url.path.endswith("/chat/completions"):
        content = request.content
        if b'"$defs"' in content:
            body = json.loads(content)
            
            def resolve_refs(node, root_defs):
                if isinstance(node, dict):
                    if "$ref" in node:
                        ref_key = node["$ref"].split("/")[-1]
                        if ref_key in root_defs:
                            resolved = root_defs[ref_key].copy()
                            return resolve_refs(resolved, root_defs)
                    return {k: resolve_refs(v, root_defs) for k, v in node.items()}
                elif isinstance(node, list):
                    return [resolve_refs(x, root_defs) for x in node]
                return node
                
            def fix_anyof(node):
                if isinstance(node, dict):
                    if "anyOf" in node:
                        types = []
                        for item in node["anyOf"]:
                            if isinstance(item, dict) and "type" in item:
                                types.append(item["type"])
                        if types:
                            if "null" in types:
                                types.remove("null")
                            if len(types) >= 1:
                                node["type"] = types[0]
                        del node["anyOf"]
                    for k, v in node.items():
                        fix_anyof(v)
                elif isinstance(node, list):
                    for x in node:
                        fix_anyof(x)

            if "tools" in body:
                for tool in body.get("tools", []):
                    params = tool.get("function", {}).get("parameters", {})
                    if "$defs" in params:
                        defs = params.pop("$defs")
                        tool["function"]["parameters"] = resolve_refs(params, defs)
                    fix_anyof(tool["function"]["parameters"])
                        
            new_content = json.dumps(body).encode("utf-8")
            # Create a completely new Request object to avoid state corruption
            headers = dict(request.headers)
            headers["content-length"] = str(len(new_content))
            request = httpx.Request(
                method=request.method,
                url=request.url,
                headers=headers,
                content=new_content
            )
            
    return await original_send(self, request, **kwargs)

httpx.AsyncClient.send = _patched_send
# -----------------------------------

# 配置用户提供的中转 API (OpenAI 兼容格式)
os.environ["OPENAI_API_KEY"] = "sk-L6MznlRXdfQHBxWvELC9TOi3QDTe5a4JLdzZslyGP2qhAs9e"
os.environ["OPENAI_BASE_URL"] = "https://api.mttieeo.com/v1"

# 初始化 Pydantic AI Agent
# 注意：使用 openai: 前缀来触发 OpenAI兼容的 API 调用
cfst_agent = Agent(
    "openai:[f]gemini-3-flash-preview",
    output_type=PaperExtraction,
    instructions=SYSTEM_PROMPT,
    retries=3,  # 结构化输出校验失败时的重试次数
)

# 注册 Dependency Type 为 Path (paper_dir)
@cfst_agent.tool
def tool_list_directory_files(ctx: RunContext[Path]) -> list[str]:
    """列出当前论文解析目录中的所有可用文件列表。"""
    return list_directory_files(ctx.deps)

@cfst_agent.tool
def tool_read_markdown(ctx: RunContext[Path]) -> str:
    """一次性读取论文解析出的 Markdown 正文内容。"""
    return read_markdown(ctx.deps)

@cfst_agent.tool
def tool_execute_python_calc(ctx: RunContext[Path], expression: str) -> float:
    """
    一个 Python 计算器。当你需要进行单位转换或尺寸计算时，传入有效的单行 Python 算术表达式。
    """
    return execute_python_calc(expression)

@cfst_agent.tool
def tool_inspect_image(ctx: RunContext[Path], image_path: str) -> bytes:
    """
    视觉读取工具。传入相对于论文目录的图片路径（如 'auto/images/img_1.jpg'）。
    如果 Markdown 表格损坏或不确定，使用此工具查看原始图片。
    """
    return inspect_image(ctx.deps, image_path)
