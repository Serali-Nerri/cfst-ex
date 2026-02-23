"""Tools for the CFST Extraction Agent."""

import ast
import operator
from pathlib import Path
import typer


def list_directory_files(paper_dir: Path) -> list[str]:
    """
    列出当前论文解析目录中的所有可用文件列表。
    """
    typer.secho(f"› Tool list_directory_files called with paper_dir='{paper_dir.name}'", dim=True)
    files = []
    if not paper_dir.exists():
        return [f"错误：目录 {paper_dir} 不存在"]
        
    for f in paper_dir.rglob("*"):
        if f.is_file():
            # 相对于 paper_dir 的路径，方便 Agent 阅读
            files.append(str(f.relative_to(paper_dir)))
    return sorted(files)


def read_markdown(paper_dir: Path) -> str:
    """
    一次性读取论文解析出的 Markdown 正文内容。
    """
    md_files = list(paper_dir.glob("**/*.md"))
    if not md_files:
        return f"未在 {paper_dir} 中找到任何 Markdown 文件"
    
    # 假设第一个或者 `auto` 目录下的就是主文件，如果是 MinerU 的输出，通常在同一级
    main_md = md_files[0]
    for md in md_files:
        if "auto" in str(md):
            main_md = md
            break
            
    typer.secho(f"› Tool read_markdown called with main_md='{main_md.name}'", dim=True)
    try:
        content = main_md.read_text(encoding="utf-8")
        return content
    except Exception as e:
        return f"读取 {main_md.name} 时出错: {e}"


def execute_python_calc(expression: str) -> float:
    """
    一个 Python 计算器。当你需要进行单位转换（如 MPa 换算）、尺寸计算（如通过外径和厚度计算内径）时，传入有效的单行 Python 算术表达式，返回精确浮点数。
    """
    typer.secho(f"› Tool execute_python_calc called with expression='{expression}'", dim=True)
    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"只支持数字常量，不支持 {type(node.value)}")
        elif isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            op_type = type(node.op)
            if op_type in allowed_ops:
                return float(allowed_ops[op_type](left, right))
            raise ValueError(f"不支持的操作符: {op_type}")
        elif isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            op_type = type(node.op)
            if op_type in allowed_ops:
                return float(allowed_ops[op_type](operand))
            raise ValueError(f"不支持的一元操作符: {op_type}")
            
        raise ValueError(f"不支持的AST节点: {ast.dump(node)}")

    # 清理多余空格和可能的恶意代码
    expression = expression.strip()
    try:
        tree = ast.parse(expression, mode="eval")
        result = eval_node(tree.body)
        return result
    except Exception as e:
        raise ValueError(f"计算表达式 '{expression}' 时出错: {e}")


def inspect_image(paper_dir: Path, image_path: str, reason: str) -> bytes:
    """
    视觉读取工具。
    传入相对于论文目录的图片路径（如 'auto/images/img_1.jpg'）。
    参数 reason: 必须用一句话说明你为什么要查看这张图片（例如：发现表格数据错位需校验，或未交代加载方式等）。
    """
    typer.secho(f"› Tool inspect_image called with image_path='{image_path}', reason='{reason}'", dim=True)
    
    full_path = paper_dir / image_path
    if not full_path.exists():
        raise FileNotFoundError(f"找不到图片: {image_path}，建议先用 list_directory_files 检查可用图片路径。")
    
    original_bytes = full_path.read_bytes()
    
    import io
    from PIL import Image
    try:
        with Image.open(full_path) as img:
            # 转换为 RGB (防止带有 alpha 通道的图或调色板图报错)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            # 缩放至最大 512x512，保持宽高比，减少输入 token
            max_size = (512, 512)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # 压缩为质量 75 的 JPEG
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=75)
            compressed_bytes = buffer.getvalue()
            
            typer.secho(f"› Image compressed: {len(original_bytes)//1024}KB -> {len(compressed_bytes)//1024}KB", dim=True)
            return compressed_bytes
    except ImportError:
        typer.secho("› Error: Pillow library missing, returning original image (run uv add pillow)", dim=True)
        return original_bytes
    except Exception as e:
        typer.secho(f"› Error: Image compression failed, returning original image: {e}", dim=True)
        return original_bytes
