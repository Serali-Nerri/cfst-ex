#!/usr/bin/env python3
"""
CFST 试验数据提取脚本
从 MinerU 解析的 content_list.json 中提取钢管混凝土试验数据
"""

import json
import re
import os
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict
from typing import Optional, List


@dataclass
class CFSTSpecimen:
    """CFST 试件数据 schema"""
    # 来源信息
    source_paper: str  # 论文标识
    specimen_id: str   # 试件编号

    # 截面类型
    section_type: str  # circular/square/rectangular

    # 几何参数
    D: Optional[float] = None   # 外径或宽度 (mm)
    B: Optional[float] = None   # 高度，矩形截面 (mm)
    t: Optional[float] = None   # 壁厚 (mm)
    L: Optional[float] = None   # 长度 (mm)
    D_t_ratio: Optional[float] = None  # 径厚比
    L_D_ratio: Optional[float] = None  # 长细比

    # 钢材属性
    fy: Optional[float] = None  # 屈服强度 (MPa)
    Es: Optional[float] = None  # 弹性模量 (MPa)
    As: Optional[float] = None  # 钢管面积 (mm²)

    # 混凝土属性
    fc: Optional[float] = None  # 抗压强度 (MPa)
    Ec: Optional[float] = None  # 弹性模量 (MPa)
    Ac: Optional[float] = None  # 混凝土面积 (mm²)

    # 试验结果
    Nu: Optional[float] = None  # 极限承载力 (kN)
    Ny: Optional[float] = None  # 屈服荷载 (kN)
    Nu_Ny_ratio: Optional[float] = None  # 延性比


def parse_html_table(body) -> List[List[str]]:
    """解析 HTML 表格为二维数组"""
    if isinstance(body, list):
        body = ''.join(body)
    if not body.strip().startswith('<'):
        return []
    soup = BeautifulSoup(body, 'html.parser')
    rows = soup.find_all('tr')
    return [[c.get_text(strip=True) for c in row.find_all(['th', 'td'])] for row in rows]


def parse_numbers(s: str) -> List[float]:
    """从字符串中提取所有数字"""
    if not s:
        return []
    s = s.replace(',', '')
    # 匹配数字，包括小数
    numbers = re.findall(r'[-+]?\d+\.?\d*', s)
    result = []
    for n in numbers:
        try:
            val = float(n)
            result.append(val)
        except ValueError:
            pass
    return result


def detect_section_type(shape_str: str) -> str:
    """根据试件编号判断截面类型"""
    shape_str = shape_str.upper()
    if shape_str.startswith('C'):
        return 'circular'
    elif shape_str.startswith('S'):
        return 'square'
    elif shape_str.startswith('R'):
        return 'rectangular'
    return 'unknown'


def extract_schneider_1998(content_list_path: str) -> List[CFSTSpecimen]:
    """
    提取 Schneider 1998 论文数据
    处理 OCR 导致的多值合并问题
    """
    with open(content_list_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tables = [item for item in data if item.get('type') == 'table']
    specimens = []

    # Table 1: 试件属性
    # 表头: Shape | D_nom | D_actual | t | D/t | L/D | A/Atotal | As | fy | Es | Ac | fc(kPa) | Ec
    if len(tables) >= 1:
        tbl1 = parse_html_table(tables[0].get('table_body', ''))

        for row in tbl1[3:]:  # 跳过表头
            if not row or not row[0]:
                continue

            # 获取试件ID列表
            shape_ids = [s.strip() for s in row[0].split() if s.strip() and not s.startswith('(')]
            if not shape_ids:
                continue

            # 解析每列的数值列表
            col_values = {}
            for i, cell in enumerate(row[1:], start=1):
                col_values[i] = parse_numbers(cell)

            # 为每个试件分配数据
            num_specimens = len(shape_ids)
            for idx, shape_id in enumerate(shape_ids):
                spec = CFSTSpecimen(
                    source_paper='Schneider_1998',
                    specimen_id=shape_id,
                    section_type=detect_section_type(shape_id)
                )

                # 辅助函数：从列中获取第 idx 个值（如果存在）
                def get_val(col_idx: int) -> Optional[float]:
                    vals = col_values.get(col_idx, [])
                    if not vals:
                        return None
                    # 如果只有一个值，所有试件共享
                    if len(vals) == 1:
                        return vals[0]
                    # 否则按索引取
                    if idx < len(vals):
                        return vals[idx]
                    return vals[0]  # fallback

                # 几何参数
                spec.t = get_val(3)  # col 4 (0-indexed: 3)
                spec.D_t_ratio = get_val(4)
                spec.L_D_ratio = get_val(5)

                # 钢材属性
                spec.As = get_val(7)
                spec.fy = get_val(8)
                spec.Es = get_val(9)

                # 混凝土属性
                spec.Ac = get_val(10)
                fc_kpa = get_val(11)
                if fc_kpa and fc_kpa > 1000:  # 单位是 kPa
                    spec.fc = fc_kpa / 1000
                elif fc_kpa:
                    spec.fc = fc_kpa  # 可能已经是 MPa
                spec.Ec = get_val(12)

                # 从 D/t 和 t 反算 D
                if spec.D_t_ratio and spec.t:
                    spec.D = spec.D_t_ratio * spec.t

                # 从 L/D 和 D 反算 L
                if spec.L_D_ratio and spec.D:
                    spec.L = spec.L_D_ratio * spec.D

                specimens.append(spec)

    # Table 2: 试验结果
    # 表头: Shape | D/t | Py | Pu | Pu/Py | ...
    if len(tables) >= 2:
        tbl2 = parse_html_table(tables[1].get('table_body', ''))

        for row in tbl2[3:]:
            if not row or not row[0]:
                continue

            shape_id = row[0].strip()
            if not shape_id or shape_id.startswith('('):
                continue

            # 查找对应试件
            for spec in specimens:
                if spec.specimen_id == shape_id:
                    vals = {}
                    for i, cell in enumerate(row[1:], start=1):
                        nums = parse_numbers(cell)
                        if nums:
                            vals[i] = nums[0]

                    # Py (col 2), Pu (col 3), Pu/Py (col 4)
                    spec.Ny = vals.get(2)
                    spec.Nu = vals.get(3)
                    spec.Nu_Ny_ratio = vals.get(4)
                    break

    return specimens


def main():
    """主函数"""
    content_list_path = "/home/thelya/tmp/cfst-ex/[A1-2]/auto/[A1-2] SCHNEIDER S P. Axially loaded concrete-filled steel tubes[J]. Journal of Structural Engineering, 1998, 124(10): 1125-1138_content_list.json"

    specimens = extract_schneider_1998(content_list_path)

    print(f"提取到 {len(specimens)} 个试件数据\n")
    print("=" * 80)

    for spec in specimens:
        print(f"\n试件 {spec.specimen_id} ({spec.section_type}):")
        if spec.D and spec.t:
            print(f"  几何: D={spec.D:.1f}mm, t={spec.t:.2f}mm, D/t={spec.D_t_ratio:.1f}, L={spec.L:.0f}mm" if spec.L else f"  几何: D={spec.D:.1f}mm, t={spec.t:.2f}mm, D/t={spec.D_t_ratio:.1f}")
        if spec.fy and spec.As:
            print(f"  钢材: fy={spec.fy:.0f}MPa, As={spec.As:.0f}mm²")
        if spec.fc and spec.Ac:
            print(f"  混凝土: fc={spec.fc:.1f}MPa, Ac={spec.Ac:.0f}mm²")
        if spec.Nu:
            print(f"  承载力: Nu={spec.Nu:.0f}kN, Ny={spec.Ny:.0f}kN, Nu/Ny={spec.Nu_Ny_ratio:.2f}" if spec.Ny else f"  承载力: Nu={spec.Nu:.0f}kN")

    # 输出 JSON
    output_path = "/home/thelya/tmp/cfst-ex/cfst-extractor/output/schneider_1998.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(s) for s in specimens], f, indent=2, ensure_ascii=False)

    print(f"\n\n结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
