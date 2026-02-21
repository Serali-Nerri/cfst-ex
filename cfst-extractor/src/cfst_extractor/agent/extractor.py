"""Extractor wrapper to run the CFST Agent."""

from datetime import datetime
from pathlib import Path

from cfst_extractor.agent.agent import cfst_agent
from cfst_extractor.agent.models import PaperExtraction


class Extractor:
    """封装 Agent 调用以提供简洁的接口。"""
    
    def __init__(self, model: str | None = None):
        """
        初始化提取器。
        
        Args:
            model: 如果提供，将覆盖默认的 'google-gla:gemini-2.5-pro' 模型。
                  支持通过 model_settings 设置。
        """
        self.model = model

    async def extract(self, paper_dir: Path) -> PaperExtraction:
        """
        从单篇论文（MinerU 解析目录）提取数据。
        Agent 会自主调用工具获取所需信息。
        
        Args:
            paper_dir: 包含解析结果 (MD和Images) 的目录路径。
            
        Returns:
            符合 PaperExtraction schema 的结构化数据。
        """
        paper_id = paper_dir.name
        prompt = (
            f"目标：请从当前分配给你的文献解析目录中提取出结构化的 CFST 试验数据，"
            f"严格遵循我们在 System Prompt 中定义的 JSON 格式进行输出。\n"
            f"操作指南与校验机制：\n"
            f"1. 【基础阅读】调用 read_markdown 读取 Markdown 正文。由于解析质量很高，多数数据可直接提取。\n"
            f"2. 【表格错位排查】检查 Markdown 中的表格数据是否有空缺、单格多值或错位（特别是复杂论文可能出现此类解析错误）。\n"
            f"   - **若表格清晰无误**：可直接提取数据，无需查阅该表原图。\n"
            f"   - **⚠️ 绝对禁止猜测与脑补**：如发现单格多值（如 `140.8 141.4`）、行列名字不对齐（如 `S5 R1` 挤在一行）、或者重要参数空缺，**你绝对不能尝试自己用逻辑分配或者切割这些值**！必须找到对应表格的原图路径，调用 inspect_image(image_path, reason='...') 去看真实的截图并以此为准进行更正！\n"
            f"3. 【加载方式校验】对于关键参数（截面形状、是否轴压偏压），若正文未明确交代，必须定位说明装置的图片路径进行查阅。\n"
            f"4. 【强制纠错循环】提取时若遇到自己都觉得很可能是排版错误的地方，说明你提取出了脏数据，必须调用看图工具来自我纠正，可以多次查阅不同图片直至无误。\n"
            f"5. 【运算强制使用工具】任何单位换算、几何计算必须使用 execute_python_calc 工具，绝对禁止心算。\n"
            f"6. 综合文本与图片信息，输出完整的结构化数据。\n"
            f"当前文献目录：{paper_id}\n"
        )
        
        try:
            # 运行 Agent，将 paper_dir 作为依赖注入给工具
            # 因为我们在 tools.py 的具体工具实现中增加了 typer.secho，所以此处不需要特殊 stream 处理也会有原生日志输出
            import typer
            
            if self.model:
                typer.secho("  [Agent] 🚀 初始化推理核心...", fg=typer.colors.MAGENTA)
                result = await cfst_agent.run(prompt, deps=paper_dir, model=self.model)
            else:
                typer.secho("  [Agent] 🚀 初始化推理核心...", fg=typer.colors.MAGENTA)
                result = await cfst_agent.run(prompt, deps=paper_dir)
                
            extraction = result.output
            
            # 后期补全部分系统元数据
            extraction.extraction_model = self.model or "default"
            extraction.extraction_time = datetime.now().isoformat()
            
            return extraction
            
        except Exception as e:
            from cfst_extractor.agent.models import RefInfo
            return PaperExtraction(
                is_valid=False,
                reason=f"Extraction Failed: {str(e)}",
                ref_info=RefInfo(title="", authors=[], journal="", year=0),
                Group_A=[],
                Group_B=[],
                Group_C=[],
                extraction_model=self.model or "default",
                extraction_time=datetime.now().isoformat(),
            )
