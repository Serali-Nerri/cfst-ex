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
            f"操作指南：\n"
            f"1. 先调用 tool_list_directory_files 查看目录中可用的文件。\n"
            f"2. 调用 read_markdown 读取全文和表格数据。\n"
            f"3. **必须**调用 inspect_image 查看关键图片（试验装置图、截面示意图、"
            f"数据表格图），以确认试件截面形状、加载方式（轴压/偏压）、以及校验 OCR 表格数据。"
            f"不能跳过此步骤，但无需逐张查看所有图片，优先选择与试验配置和数据相关的图片。\n"
            f"4. 综合文本与图片信息，输出完整的结构化数据。\n"
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
