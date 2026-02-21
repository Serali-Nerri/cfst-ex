"""CLI entry point for cfst-extractor."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from cfst_extractor.agent.extractor import Extractor
from cfst_extractor.agent.models import PaperExtraction

app = typer.Typer(help="CFST Experimental Data Extractor (Agent-Based)")


def _count_groups(result: PaperExtraction) -> dict[str, int]:
    return {
        "A": len(result.Group_A),
        "B": len(result.Group_B),
        "C": len(result.Group_C)
    }


@app.command()
def single(
    parsed_dir: str = typer.Argument(..., help="Path to MinerU parsed output directory"),
    output: str = typer.Option("output", "-o", help="Output directory"),
    model: str = typer.Option(None, "-m", help="LLM model to use (e.g. google-gla:gemini-2.5-pro)"),
) -> None:
    """Extract CFST data from a single MinerU-parsed document using LLM Agent."""
    
    doc_dir = Path(parsed_dir)
    if not doc_dir.exists() or not doc_dir.is_dir():
        typer.echo(f"Error: Directory {parsed_dir} does not exist.")
        raise typer.Exit(1)
        
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
        
    ext = Extractor(model=model)
    
    typer.echo(f"Starting extraction for {doc_dir.name} using {ext.model or 'gemini-2.5-pro'}...")
    result: PaperExtraction = asyncio.run(ext.extract(doc_dir))

    valid = (len(result.Group_A) + len(result.Group_B) + len(result.Group_C)) > 0
    groups = _count_groups(result)
    total = sum(groups.values())
    
    # Save JSON result
    out_file = out_dir / f"{doc_dir.name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        result_json = result.model_dump_json(indent=2)
        f.write(result_json)
        
        # 统计数量
        total = len(result.Group_A) + len(result.Group_B) + len(result.Group_C)
        typer.secho(
            f"\nExtraction complete! Found {total} specimens.", 
            fg=typer.colors.GREEN
        )
    if valid:
        typer.echo(f"Valid paper. Extracted {total} specimens: {groups}")
        typer.echo(f"Saved result to {out_file}")
    else:
        typer.echo(f"No specimens extracted or extraction failed: {getattr(result, 'reason', 'Unknown reason')}")


@app.command()
def batch(
    parsed_root: str = typer.Argument(..., help="Root directory containing MinerU outputs"),
    output: str = typer.Option("output", "-o", help="Output directory"),
    model: str = typer.Option(None, "-m", help="LLM model to use"),
    workers: int = typer.Option(3, "-w", help="Number of parallel async workers"),
) -> None:
    """Batch-extract CFST data from multiple MinerU-parsed documents."""

    root = Path(parsed_root)
    parsed_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and (d / "auto").is_dir()
    )

    if not parsed_dirs:
        # 兼容那些没有 auto 子目录而是直接把文件放里面的情况
        parsed_dirs = sorted(
            d for d in root.iterdir()
            if d.is_dir() and list(d.glob("*.md"))
        )

    if not parsed_dirs:
        typer.echo(f"No valid parsed directories found in {parsed_root}")
        raise typer.Exit(1)

    typer.echo(f"Found {len(parsed_dirs)} parsed documents")
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ext = Extractor(model=model)

    async def _process_batch():
        sem = asyncio.Semaphore(workers)
        
        async def _process_one(d: Path):
            async with sem:
                typer.echo(f"Processing {d.name}...")
                res = await ext.extract(d)
                
                out_file = out_dir / f"{d.name}.json"
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(res.model_dump_json(indent=2))
                return d.name, res
                
        tasks = [_process_one(d) for d in parsed_dirs]
        return await asyncio.gather(*tasks, return_exceptions=True)

    results_raw = asyncio.run(_process_batch())
    
    summary = {
        "total_papers": len(parsed_dirs),
        "valid_papers": 0,
        "invalid_papers": 0,
        "total_specimens": 0,
        "papers": {}
    }

    for r in results_raw:
        if isinstance(r, Exception):
            typer.echo(f"ERROR: {r}")
            # Assuming the exception means the paper could not be processed
            # We don't have the doc_name here, so we can't add it to summary["papers"]
            summary["invalid_papers"] += 1
        else:
            doc_name, result = r
            count = len(result.Group_A) + len(result.Group_B) + len(result.Group_C)
            summary["papers"][doc_name] = {
                "status": "success" if count > 0 else "empty",
                "specimens": count,
                "notes": getattr(result, 'reason', None) if count == 0 else None
            }
            summary["total_specimens"] += count
            
            if count > 0:
                summary["valid_papers"] += 1
                groups = _count_groups(result)
                typer.secho(f"  OK {doc_name}: {count} specimens {groups}", fg=typer.colors.GREEN)
            else:
                summary["invalid_papers"] += 1
                typer.echo(f"  INVALID {doc_name}: {getattr(result, 'reason', 'Unknown reason')}")

    typer.echo(f"\nBatch Summary: {summary['total_papers']} papers, {summary['valid_papers']} valid, {summary['total_specimens']} specimens")

    summary_path = out_dir / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    app()
