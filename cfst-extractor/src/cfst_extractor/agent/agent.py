"""Core AI Agent for CFST Data Extraction."""

import json
import os
from pathlib import Path

import httpx
import yaml
from pydantic_ai import Agent, RunContext

from cfst_extractor.agent.models import PaperExtraction
from cfst_extractor.agent.tools import (
    execute_python_calc,
    inspect_image,
    list_directory_files,
    read_markdown,
)

# 读取 System Prompt
_PROMPT_PATH = Path(__file__).resolve().parents[4] / "config" / "System_Prompt.md"
if _PROMPT_PATH.exists():
    SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")
else:
    print(f"WARNING: 找不到 Prompt 文件 {_PROMPT_PATH}")
    SYSTEM_PROMPT = "你是一个专门从钢管混凝土（CFST）科学论文中提取试验数据的专家。"

# ---------------------------------------------------------------------------
# 配置加载: YAML 文件 → 环境变量覆盖 → 代码默认值
# ---------------------------------------------------------------------------
_SETTINGS_PATH = Path(__file__).resolve().parents[4] / "config" / "settings.yaml"

# 平台预设: 每个平台对 OpenAI 兼容协议的"方言"不同，预设自动选择正确的补丁组合
_PLATFORM_PRESETS: dict[str, dict[str, bool]] = {
    "dashscope":   {"flatten_defs": True,  "fix_tool_choice": True,  "fix_anyof": True,  "xhigh": False},
    "openai":      {"flatten_defs": False, "fix_tool_choice": False, "fix_anyof": False, "xhigh": False},
    "local_proxy": {"flatten_defs": True,  "fix_tool_choice": False, "fix_anyof": True,  "xhigh": True},
    "custom":      {"flatten_defs": False, "fix_tool_choice": False, "fix_anyof": False, "xhigh": False},
}

_DEFAULTS = {
    "api": {"api_key": "", "base_url": ""},
    "model": {"name": "google-gla:gemini-2.5-pro"},
    "agent": {"retries": 3, "platform": "openai"},
}


def _load_settings() -> dict:
    """加载 settings.yaml，缺失时回退到内置默认值。"""
    cfg = _DEFAULTS.copy()
    if _SETTINGS_PATH.exists():
        with open(_SETTINGS_PATH, encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
        # 逐层合并 (YAML 覆盖默认)
        for section in ("api", "model", "agent"):
            if section in file_cfg:
                cfg[section] = {**cfg[section], **file_cfg[section]}
    else:
        print(f"WARNING: 配置文件不存在 {_SETTINGS_PATH}，使用内置默认值")
    return cfg


def _resolve_patches(agent_cfg: dict) -> dict[str, bool]:
    """从 platform 预设 + 手动 patches 覆盖解析最终的补丁开关。"""
    platform = agent_cfg.get("platform", "openai")
    base = _PLATFORM_PRESETS.get(platform, _PLATFORM_PRESETS["custom"]).copy()
    # agent.patches 中的手动配置覆盖预设
    manual = agent_cfg.get("patches", {})
    if manual:
        base.update(manual)
    return base


_settings = _load_settings()
_patches = _resolve_patches(_settings["agent"])

# 环境变量优先级最高
_api_key = os.environ.get("OPENAI_API_KEY") or _settings["api"].get("api_key", "")
_base_url = os.environ.get("OPENAI_BASE_URL") or _settings["api"].get("base_url", "")
_model_name = os.environ.get("CFST_MODEL") or _settings["model"]["name"]
_retries = int(os.environ.get("CFST_RETRIES", _settings["agent"]["retries"]))

# 仅在有值时设置环境变量 (供 pydantic-ai 的 OpenAI provider 读取)
if _api_key:
    os.environ["OPENAI_API_KEY"] = _api_key
if _base_url:
    os.environ["OPENAI_BASE_URL"] = _base_url

# ---------------------------------------------------------------------------
# HTTP 请求拦截补丁 (按平台预设启用/禁用各项修补)
# ---------------------------------------------------------------------------
original_send = httpx.AsyncClient.send


async def _patched_send(self, request: httpx.Request, **kwargs):
    if request.url.path.endswith("/chat/completions"):
        content = request.content
        if b'"$defs"' in content or b'"model"' in content:
            body = json.loads(content)
            modified = False

            # 补丁 1: 注入 xhigh (思考强调) 参数
            if _patches["xhigh"]:
                body["xhigh"] = True
                modified = True

            # 补丁 2: thinking mode 兼容 — tool_choice=required → auto
            if _patches["fix_tool_choice"]:
                tc = body.get("tool_choice")
                if tc == "required" or (isinstance(tc, dict) and tc.get("type") == "required"):
                    body["tool_choice"] = "auto"
                    modified = True

            # 补丁 3: 展平 $defs/$ref 嵌套引用
            if _patches["flatten_defs"] and "tools" in body:
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

                for tool in body.get("tools", []):
                    params = tool.get("function", {}).get("parameters", {})
                    if "$defs" in params:
                        defs = params.pop("$defs")
                        tool["function"]["parameters"] = resolve_refs(params, defs)
                        modified = True

            # 补丁 4: 将 anyOf 简化为单一 type (部分平台不支持)
            if _patches["fix_anyof"] and "tools" in body:
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

                for tool in body.get("tools", []):
                    fix_anyof(tool.get("function", {}).get("parameters", {}))
                modified = True

            if modified:
                new_content = json.dumps(body).encode("utf-8")
                headers = dict(request.headers)
                headers["content-length"] = str(len(new_content))
                request = httpx.Request(
                    method=request.method,
                    url=request.url,
                    headers=headers,
                    content=new_content,
                )

    return await original_send(self, request, **kwargs)


httpx.AsyncClient.send = _patched_send
# ---------------------------------------------------------------------------

# 初始化 Pydantic AI Agent
cfst_agent = Agent(
    _model_name,
    output_type=PaperExtraction,
    instructions=SYSTEM_PROMPT,
    retries=_retries,
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
