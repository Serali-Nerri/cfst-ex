# CFST 数据提取 Agent 设计文档

## 1. 概述

本文档描述了一个基于 LLM 的自动化数据提取 Agent，用于从 CFST（钢管混凝土）科学论文中提取结构化试验数据。

### 1.1 设计目标

- **通用性**：一个 Agent 处理所有论文，无需为每篇论文编写特定代码
- **可扩展性**：支持 300+ 篇论文的批量处理
- **多模型适配**：支持 Claude、GPT-4o、Gemini 等主流 LLM
- **高质量输出**：结构化 JSON，符合预定义 Schema

### 1.2 核心思路

放弃传统的规则匹配方案，构建具备 **工具调用能力** 的 LLM Agent：
- Agent 自主决定何时读取图片、计算公式、查看文件列表
- 多步推理，按需获取信息，而非一次性投喂所有内容
- 输出：结构化 JSON（试件数据）

**为什么需要工具调用？**
- 公式计算：论文中 `L = 3D` 需要准确计算，LLM 直接算可能出错
- 按需读图：Agent 发现需要某张表格图片时再读取，而非投喂所有图片
- 文件探索：Agent 可以查看工作目录有哪些文件可用
- 充分发挥模型能力：Gemini 2.5 Pro、Claude 等模型的 tool use、多步推理能力需要 Agent 框架才能体现

**技术栈**：
- **Pydantic AI**：Agent 框架，支持工具调用 + 结构化输出
- **Pydantic**：数据 Schema 定义 + 校验

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CFST Extraction Agent 架构                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────────────────────────────────────────┐   │
│  │   输入层    │     │                  调度层 (Orchestrator)           │   │
│  ├─────────────┤     ├─────────────────────────────────────────────────┤   │
│  │ - 解析目录  │────▶│  - 任务队列管理                                  │   │
│  │   /paper_1/ │     │  - 并发控制 (asyncio)                           │   │
│  │   /paper_2/ │     │  - 进度追踪 & 断点续传                           │   │
│  │   ...       │     │  - 错误重试 & 降级策略                           │   │
│  └─────────────┘     └──────────────────┬──────────────────────────────┘   │
│                                         │                                   │
│                                         ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Agent 核心 (Pydantic AI)                         │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │   Agent = LLM + Tools + 结构化输出                                  │   │
│  │                                                                     │   │
│  │   ┌─────────────────────────────────────────────────────────────┐  │   │
│  │   │  工具集 (Tools)                                              │  │   │
│  │   ├─────────────────────────────────────────────────────────────┤  │   │
│  │   │  list_files()     - 列出工作目录中的文件                     │  │   │
│  │   │  read_image()     - 读取指定图片                             │  │   │
│  │   │  calculate()      - 计算数学表达式                           │  │   │
│  │   │  read_markdown()  - 读取 Markdown 内容                       │  │   │
│  │   │  unit_convert()   - 单位换算                                 │  │   │
│  │   └─────────────────────────────────────────────────────────────┘  │   │
│  │                                                                     │   │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│  │   │ Claude   │  │ GPT-4o   │  │ Gemini   │  │ 本地模型  │          │   │
│  │   │ API      │  │ API      │  │ 2.5 Pro  │  │ (Ollama) │          │   │
│  │   └──────────┘  └──────────┘  └──────────┘  └──────────┘          │   │
│  │                                                                     │   │
│  │   执行流程:                                                         │   │
│  │   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐        │   │
│  │   │ 读取MD  │───▶│ 分析    │───▶│ 调用    │───▶│ 综合    │        │   │
│  │   │ 文件    │    │ 需要什么│    │ 工具    │    │ 输出    │        │   │
│  │   └─────────┘    └─────────┘    └────┬────┘    └─────────┘        │   │
│  │                                      │ 循环                        │   │
│  │                                      ▼                             │   │
│  │                              ┌─────────────┐                       │   │
│  │                              │ 工具返回结果│                       │   │
│  │                              │ 继续推理... │                       │   │
│  │                              └─────────────┘                       │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                         │                                   │
│                                         ▼                                   │
│  ┌─────────────┐                                                           │
│  │   输出层    │                                                           │
│  ├─────────────┤                                                           │
│  │ - JSON 文件 │  (每篇论文一个 JSON)                                      │
│  │ - 汇总 CSV  │  (所有试件合并)                                           │
│  │ - 提取日志  │  (来源、置信度、问题标注)                                  │
│  └─────────────┘                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3. 核心组件设计

### 3.1 调度层 (Orchestrator)

负责批量任务的调度和管理。

```python
# orchestrator.py
import asyncio
from pathlib import Path
from typing import List
from dataclasses import dataclass

@dataclass
class ExtractionTask:
    paper_id: str
    input_dir: Path
    output_path: Path
    status: str = "pending"  # pending, running, completed, failed
    retry_count: int = 0

class Orchestrator:
    def __init__(self, config: dict):
        self.concurrency = config.get("concurrency", 5)
        self.max_retries = config.get("max_retries", 3)
        self.semaphore = asyncio.Semaphore(self.concurrency)

    async def process_batch(self, tasks: List[ExtractionTask]) -> List[dict]:
        """并发处理一批任务"""
        async def process_one(task: ExtractionTask):
            async with self.semaphore:
                try:
                    task.status = "running"
                    result = await self.extractor.extract(task.input_dir)
                    task.status = "completed"
                    return {"task": task, "result": result, "error": None}
                except Exception as e:
                    task.retry_count += 1
                    if task.retry_count < self.max_retries:
                        return await process_one(task)  # 重试
                    task.status = "failed"
                    return {"task": task, "result": None, "error": str(e)}

        results = await asyncio.gather(
            *[process_one(t) for t in tasks],
            return_exceptions=True
        )
        return results

    def save_progress(self, tasks: List[ExtractionTask]):
        """保存进度，支持断点续传"""
        # 保存到 JSON 文件
        pass

    def load_progress(self, progress_file: Path) -> List[ExtractionTask]:
        """加载进度"""
        pass
```

### 3.2 Agent 核心 (Pydantic AI)

使用 **Pydantic AI** 构建具备工具调用能力的 Agent。

**为什么用 Pydantic AI？**
- 原生支持工具调用（function calling）
- 结构化输出（`result_type`）
- 多模型支持（Claude/GPT/Gemini/Ollama）
- 与 Pydantic Schema 无缝集成
- 充分发挥 LLM 的多步推理能力

```python
# agent.py
from pydantic_ai import Agent
from pathlib import Path
from .models import PaperExtraction
from .tools import list_files, read_image, calculate, read_markdown, unit_convert

# 创建 Agent
agent = Agent(
    'google-gla:gemini-2.5-pro',  # 或 'claude-sonnet-4-20250514', 'openai:gpt-4o'
    result_type=PaperExtraction,   # 结构化输出
    system_prompt=SYSTEM_PROMPT,
    retries=3,
)

# 注册工具
@agent.tool
def list_files(ctx) -> list[str]:
    """列出当前论文目录中的所有文件"""
    paper_dir: Path = ctx.deps
    files = []
    for f in paper_dir.rglob('*'):
        if f.is_file():
            files.append(str(f.relative_to(paper_dir)))
    return files

@agent.tool
def read_image(ctx, image_name: str) -> bytes:
    """读取指定图片，如 'images/img_5.jpg'"""
    paper_dir: Path = ctx.deps
    img_path = paper_dir / image_name
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_name}")
    return img_path.read_bytes()

@agent.tool
def calculate(expression: str) -> float:
    """
    计算数学表达式。
    示例: calculate("3 * 150") → 450.0
    示例: calculate("D / t") 需要先获取 D 和 t 的值
    """
    import ast
    import operator

    # 安全的表达式计算
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }

    def eval_expr(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = eval_expr(node.left)
            right = eval_expr(node.right)
            return allowed_operators[type(node.op)](left, right)
        else:
            raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    tree = ast.parse(expression, mode='eval')
    return eval_expr(tree.body)

@agent.tool
def read_markdown(ctx) -> str:
    """读取论文的 Markdown 内容"""
    paper_dir: Path = ctx.deps
    md_files = list(paper_dir.glob("**/*.md"))
    if not md_files:
        raise FileNotFoundError("No markdown file found")
    return md_files[0].read_text(encoding="utf-8")

@agent.tool
def unit_convert(value: float, from_unit: str, to_unit: str) -> float:
    """
    单位换算。
    示例: unit_convert(1000, "N", "kN") → 1.0
    示例: unit_convert(10, "cm", "mm") → 100.0
    """
    conversions = {
        ("N", "kN"): 0.001,
        ("kN", "N"): 1000,
        ("MN", "kN"): 1000,
        ("kN", "MN"): 0.001,
        ("cm", "mm"): 10,
        ("mm", "cm"): 0.1,
        ("m", "mm"): 1000,
        ("mm", "m"): 0.001,
        ("GPa", "MPa"): 1000,
        ("MPa", "GPa"): 0.001,
        ("kPa", "MPa"): 0.001,
        ("MPa", "kPa"): 1000,
    }
    key = (from_unit, to_unit)
    if key not in conversions:
        raise ValueError(f"Unknown conversion: {from_unit} → {to_unit}")
    return value * conversions[key]
```

**Agent 执行流程示例：**

```
用户: 提取论文 paper_001 的试件数据

Agent 思考: 我需要先了解这个目录有什么文件
  → 调用 list_files()
  → 返回: ["auto/paper.md", "auto/images/img_1.jpg", "auto/images/img_2.jpg", ...]

Agent 思考: 先读取 Markdown 内容
  → 调用 read_markdown()
  → 返回: "# 钢管混凝土柱轴压试验研究..."

Agent 思考: 表格中 L = 3D，D = 150mm，我需要计算 L
  → 调用 calculate("3 * 150")
  → 返回: 450.0

Agent 思考: 表格 OCR 不清晰，需要看原图确认
  → 调用 read_image("auto/images/img_5.jpg")
  → 返回: <图片数据>

Agent 思考: 论文中力的单位是 N，需要转换为 kN
  → 调用 unit_convert(8810, "N", "kN")
  → 返回: 8.81

Agent: 综合所有信息，输出结构化 JSON
  → 返回: PaperExtraction(specimens=[...])
```

**支持的模型配置：**

```yaml
# config.yaml
agent:
  # Pydantic AI 模型格式
  model: google-gla:gemini-2.5-pro
  # 或: claude-sonnet-4-20250514
  # 或: openai:gpt-4o
  # 或: ollama:llama3.1

  retries: 3  # 结构化输出校验失败时的重试次数

extraction:
  concurrency: 5
  timeout_seconds: 300  # Agent 可能需要多轮工具调用，超时设长一些
```

**安装依赖：**
```bash
pip install pydantic-ai
```

### 3.3 提取入口 (Extractor)

封装 Agent 调用，提供简洁的接口。

```python
# extractor.py
from pathlib import Path
from .agent import agent
from .models import PaperExtraction

class Extractor:
    async def extract(self, paper_dir: Path) -> PaperExtraction:
        """从单篇论文提取数据"""
        # Agent 会自主调用工具获取所需信息
        # paper_dir 作为依赖注入，工具通过 ctx.deps 访问
        result = await agent.run(
            f"请从目录 {paper_dir.name} 中提取所有 CFST 试件数据",
            deps=paper_dir,
        )
        return result.data
```

### 3.4 输出 Schema (Pydantic)

Pydantic Schema 用于：
1. **Agent 结构化输出**：Pydantic AI 的 `result_type` 约束 Agent 最终输出
2. **自动重试**：输出不符合 Schema 时，Agent 会自动修正并重试

```python
# models.py
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum

class SectionType(str, Enum):
    CIRCULAR = "circular"
    SQUARE = "square"
    RECTANGULAR = "rectangular"
    CIRCULAR_END = "circular_end"

class LoadingType(str, Enum):
    AXIAL = "axial"
    ECCENTRIC = "eccentric"
    BIAXIAL = "biaxial"

class ConcreteType(BaseModel):
    """混凝土强度类型"""
    value: float = Field(..., description="强度值 (MPa)")
    specimen_type: str = Field(..., description="试件类型，如 'Cube 150', 'Cylinder 150x300'")

class Specimen(BaseModel):
    """单个试件数据"""
    # 标识
    specimen_id: str = Field(..., description="试件编号")

    # 截面类型
    section_type: SectionType

    # 几何参数 (mm)
    D: Optional[float] = Field(None, description="外径/宽度", ge=10, le=2000)
    B: Optional[float] = Field(None, description="高度（矩形截面）", ge=10, le=2000)
    t: Optional[float] = Field(None, description="壁厚", ge=0.5, le=50)
    L: Optional[float] = Field(None, description="长度", ge=50, le=10000)
    r0: Optional[float] = Field(None, description="内圆角半径")

    # 钢材属性
    fy: Optional[float] = Field(None, description="屈服强度 (MPa)", ge=100, le=1000)
    Es: Optional[float] = Field(None, description="弹性模量 (MPa)", ge=150000, le=250000)

    # 混凝土属性
    fc: Optional[ConcreteType] = Field(None, description="混凝土强度")
    Ec: Optional[float] = Field(None, description="弹性模量 (MPa)")

    # 加载条件
    loading_type: LoadingType = LoadingType.AXIAL
    e1: Optional[float] = Field(None, description="偏心距1 (mm)")
    e2: Optional[float] = Field(None, description="偏心距2 (mm)")

    # 试验结果
    Nu: Optional[float] = Field(None, description="极限承载力 (kN)", ge=0)

    # 元数据
    confidence: float = Field(1.0, description="提取置信度", ge=0, le=1)
    issues: List[str] = Field(default_factory=list, description="问题标注")

    # Pydantic 校验器：校验失败时 Agent 会自动重试
    @field_validator('Nu')
    @classmethod
    def check_nu_range(cls, v):
        if v is not None and (v < 10 or v > 50000):
            raise ValueError(f"Nu={v} kN 超出合理范围 [10, 50000]，请使用 calculate() 工具重新计算或检查单位")
        return v

class PaperExtraction(BaseModel):
    """单篇论文提取结果"""
    # 论文元数据
    paper_id: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    journal: Optional[str] = None

    # 试件数据
    specimens: List[Specimen]

    # 提取元数据
    extraction_model: str = Field(..., description="使用的 LLM 模型")
    extraction_time: str = Field(..., description="提取时间 ISO 格式")
    total_specimens: int = Field(..., description="试件总数")

    # 备注
    notes: Optional[str] = Field(None, description="提取过程中的备注")
```

### 3.5 工具集 (Tools)

Agent 可调用的工具，封装在独立模块中。

```python
# tools.py
"""
Agent 工具集

工具设计原则：
1. 每个工具做一件事，职责单一
2. 返回值清晰，便于 Agent 理解
3. 错误信息明确，帮助 Agent 自我纠正
"""

from pathlib import Path
import ast
import operator

def list_files(paper_dir: Path) -> list[str]:
    """列出论文目录中的所有文件"""
    files = []
    for f in paper_dir.rglob('*'):
        if f.is_file():
            files.append(str(f.relative_to(paper_dir)))
    return sorted(files)

def read_image(paper_dir: Path, image_name: str) -> bytes:
    """读取指定图片"""
    img_path = paper_dir / image_name
    if not img_path.exists():
        available = [f for f in list_files(paper_dir) if f.endswith(('.jpg', '.png'))]
        raise FileNotFoundError(
            f"Image not found: {image_name}. Available images: {available[:5]}"
        )
    return img_path.read_bytes()

def calculate(expression: str) -> float:
    """安全计算数学表达式"""
    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }

    def eval_node(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return allowed_ops[type(node.op)](
                eval_node(node.left),
                eval_node(node.right)
            )
        raise ValueError(f"Unsupported: {ast.dump(node)}")

    tree = ast.parse(expression, mode='eval')
    return float(eval_node(tree.body))

def read_markdown(paper_dir: Path) -> str:
    """读取 Markdown 内容"""
    md_files = list(paper_dir.glob("**/*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown in {paper_dir}")
    return md_files[0].read_text(encoding="utf-8")

# 单位换算表
UNIT_CONVERSIONS = {
    ("N", "kN"): 0.001,
    ("kN", "N"): 1000,
    ("MN", "kN"): 1000,
    ("cm", "mm"): 10,
    ("m", "mm"): 1000,
    ("GPa", "MPa"): 1000,
    ("kPa", "MPa"): 0.001,
}

def unit_convert(value: float, from_unit: str, to_unit: str) -> float:
    """单位换算"""
    if from_unit == to_unit:
        return value
    key = (from_unit, to_unit)
    if key not in UNIT_CONVERSIONS:
        raise ValueError(f"Unknown: {from_unit} → {to_unit}")
    return value * UNIT_CONVERSIONS[key]
```

## 4. System Prompt 设计

System Prompt 是 Agent 的核心，包含领域知识和提取规则。

```markdown
# CFST 试验数据提取专家

你是一个专门从钢管混凝土（CFST）科学论文中提取试验数据的专家。

## 任务

从提供的论文内容中提取所有试件的试验数据，输出结构化 JSON。

## 领域知识

### 截面类型分组
- **Group A (方形/矩形)**: b ≠ h 或 b = h 且 r0 = 0
- **Group B (圆形)**: b = h = D, r0 = D/2
- **Group C (圆端形)**: b ≥ h, r0 = h/2

### 关键字段定义
| 字段 | 含义 | 单位 | 常见范围 |
|------|------|------|----------|
| D | 外径/宽度 | mm | 50-1000 |
| B | 高度（矩形） | mm | 50-1000 |
| t | 壁厚 | mm | 1-30 |
| L | 长度 | mm | 100-5000 |
| fy | 钢材屈服强度 | MPa | 200-500 |
| fc | 混凝土抗压强度 | MPa | 20-100 |
| Nu | 极限承载力 | kN | 100-10000 |

### 混凝土强度类型
- **立方体强度 (fcu)**: Cube 150mm, Cube 100mm
- **圆柱体强度 (fc')**: Cylinder 150×300mm, Cylinder 100×200mm

### 单位标准
- 力: **kN** (注意 N→kN, MN→kN 换算)
- 应力: **MPa** (注意 kPa→MPa, GPa→MPa 换算)
- 长度: **mm** (注意 cm→mm, m→mm 换算)

## 提取规则

1. **只提取试验数据**：排除 FEA/有限元分析结果、理论计算值
2. **识别试件编号**：通常在表格第一列，如 C1, S2, R3, SC-1, 试件A 等
3. **处理多表融合**：几何参数和试验结果可能在不同表中，按试件ID关联
4. **处理 OCR 错误**：
   - `3.426` 可能是 `3,426`（千位分隔符）
   - 检查数值是否在合理范围内
5. **缺失值处理**：
   - 如果某字段在论文中未提供，设为 null
   - 如果可以从其他字段推算（如 D = D/t × t），进行计算并标注

## 输出格式

```json
{
  "paper_id": "论文标识",
  "title": "论文标题",
  "specimens": [
    {
      "specimen_id": "C1",
      "section_type": "circular",
      "D": 140.0,
      "t": 3.0,
      "L": 600.0,
      "fy": 285.0,
      "fc": {"value": 28.2, "specimen_type": "Cylinder"},
      "Nu": 881.0,
      "loading_type": "axial",
      "confidence": 0.95,
      "issues": []
    }
  ],
  "notes": "提取过程中的备注"
}
```

## 注意事项

1. 仔细阅读表格标题和脚注，确定单位和试件类型
2. 如果表格图片比 HTML 更清晰，优先参考图片
3. 对于不确定的值，降低 confidence 并在 issues 中说明
4. 如果论文不是 CFST 相关（如钢节点、纯钢柱），返回空 specimens 列表
```

## 5. 输入内容策略

### 5.1 输入组成

| 输入内容 | 必要性 | 说明 |
|---------|--------|------|
| Markdown 全文 | **必须** | 包含所有文本、表格（HTML）、公式（LaTeX）、图片引用 |
| 所有图片 | **推荐** | 表格截图可帮助修正 OCR 错误，加载装置图帮助判断试验类型 |
| content_list.json | 可选 | 提供 bbox、页码等元数据，一般不需要 |

### 5.2 为什么 Markdown 足够

MinerU 解析出的 Markdown 包含：
- 完整的文本内容（OCR 质量较高）
- 表格的 HTML 结构（`<table>` 标签）
- 公式的 LaTeX 格式（`$...$`）
- 图片引用（`![](images/xxx.jpg)`）

LLM 能够理解 HTML 表格结构，直接从中提取数据。

### 5.3 图片处理策略

**全部投喂**：将 `images/` 目录下所有图片作为输入，让 LLM 自己判断哪些有用。

理由：
1. 现代多模态 LLM 对无关图片的干扰容忍度较高
2. 避免漏掉关键信息（如表格截图、加载装置图）
3. 自动筛选图片的成本高于直接投喂

### 5.4 Token 成本估算

| 内容 | Token 数 |
|------|----------|
| System Prompt | ~2K |
| Markdown 文本 | ~10-20K |
| 图片（15张 × ~1K/张） | ~15K |
| **单篇总计** | **~25-35K** |

**300 篇论文总成本**：
- Claude Sonnet: ~$30-50
- GPT-4o: ~$40-60
- Claude Opus: ~$150-200

## 6. 项目结构

```
cfst-extractor/
├── agent/                          # Agent 核心代码
│   ├── __init__.py
│   ├── orchestrator.py             # 调度层
│   ├── agent.py                    # Pydantic AI Agent 定义
│   ├── tools.py                    # 工具集
│   ├── extractor.py                # 提取入口
│   └── models.py                   # Pydantic Schema
├── prompts/                        # Prompt 模板
│   └── system_prompt.md            # System Prompt
├── config/                         # 配置文件
│   └── config.yaml                 # 主配置
├── scripts/                        # 脚本
│   ├── run_extraction.py           # 批量提取入口
│   └── validate_output.py          # 输出校验
├── output/                         # 输出目录
│   ├── json/                       # 单篇 JSON
│   └── merged.csv                  # 汇总 CSV
└── tests/
```

## 7. 使用流程

### 7.1 准备工作

1. **PDF 解析**（Colab）：
   ```bash
   # 在 Colab 中运行 cfst_mineru_test.ipynb
   # 输出: /content/parsed_output/<paper_name>/
```

2. **下载解析结果**：
   ```bash
   # 下载到本地
   scp -r colab:/content/parsed_output ./parsed/
   ```

3. **配置 API Key**：
   ```bash
   export ANTHROPIC_API_KEY=sk-xxx
   # 或在 config.yaml 中配置
   ```

### 7.2 运行提取

```bash
# 单篇测试
python scripts/run_extraction.py --input ./parsed/paper_1 --output ./output/paper_1.json

# 批量处理
python scripts/run_extraction.py --input ./parsed --output ./output --concurrency 5

# 断点续传
python scripts/run_extraction.py --input ./parsed --output ./output --resume ./output/progress.json
```

### 7.3 输出校验

```bash
# 校验输出格式
python scripts/validate_output.py ./output/json/

# 与金标准对比
python scripts/validate_output.py ./output/json/ --gold ./testdata/gold/
```

## 8. 参考项目

| 项目 | 特点 | 参考价值 |
|------|------|----------|
| [Pydantic AI](https://github.com/pydantic/pydantic-ai) | Agent 框架 | **核心依赖**，工具调用 + 结构化输出 |
| [Instructor](https://github.com/567-labs/instructor) | 结构化输出 | 简单场景备选 |
| [LLM-IE](https://github.com/daviden1013/llm-ie) | 信息提取 | 批量处理参考 |
| [Awesome-LLM4IE-Papers](https://github.com/quqxui/Awesome-LLM4IE-Papers) | 论文合集 | 学术前沿 |

## 9. 后续优化方向

1. **Prompt 优化**：根据提取效果迭代 System Prompt
2. **Few-shot 示例**：在 Prompt 中加入高质量提取示例
3. **工具扩展**：根据实际需求增加新工具
4. **自动校验**：与金标准对比，评估提取质量
5. **人工复核**：对低置信度结果进行人工复核
