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
            f"操作指南与校验机制（请严格按以下核心工作流按顺序执行）：\n"
            f"1. 【基础阅读】首要步骤：调用 `read_markdown` 读取文献的 Markdown 正文文本，以获取全局信息。\n"
            f"2. 【加载方式判定】关键校验：必须从提取到的图片列表中定位【加载装置示意图】，并强制调用 `inspect_image` 工具查阅该原图。通过原图事实直接评判加载方式是否为偏心加载，且是否为上下等端距离加载（或非等端距离加载）。这一步不可跳过。\n"
            f"3. 【表格错位排查（强制防坑）】数据提取：基于 Markdown 文本中的表格，仔细对比每一行的物理意义，你必须意识到 MinerU 会把原本分为多行的试件标识（如 C1、C2）强行合并到一个单元格（例如 `C1 C2` 或者 `S5 R1`），这会导致右侧所有的数据列发生严重的行错位和单格多值！\n"
            f"   - **若表格清晰且试件标识行列一一对应**：直接从文本提取数据，无需查阅表格原图。\n"
            f"   - **⚠️ 若存在任何异常（尤其是试件名字/Shape列发生合并如 `C1 C2`、数据区出现空格隔开的多个数值如 `76.6 152.3`）**：这代表表格已被严重破坏！绝对禁止运用个人逻辑对数据进行切割分配！你**必须立刻**在图片列表中找到该表格的原图，并调用 `inspect_image` 查看。\n"
            f"4. 【运算工具使用】任何单位换算、几何截面计算需强制使用 `execute_python_calc` 工具。\n"
            f"5. 【综合得出结果】最后，结合上述 Markdown 正文、加载装置查阅结果以及任何可能修正过的表格数据，整理得出结论并输出规范的 JSON 数据。\n"
            f"当前文献目录：{paper_id}\n"
        )
        
        try:
            # 运行 Agent，将 paper_dir 作为依赖注入给工具
            # 因为我们在 tools.py 的具体工具实现中增加了 typer.secho，所以此处不需要特殊 stream 处理也会有原生日志输出
            import typer
            
            if self.model:
                typer.secho("› Initializing inference core...", dim=True)
                result = await cfst_agent.run(prompt, deps=paper_dir, model=self.model)
            else:
                typer.secho("› Initializing inference core...", dim=True)
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
