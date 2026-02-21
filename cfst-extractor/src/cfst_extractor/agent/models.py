from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class SpecimenBase(BaseModel):
    """提取的单一试件标准格式"""
    ref_no: str = Field(default="", description="留空")
    specimen_label: str = Field(..., description="试件编号")
    fc_value: float = Field(..., description="混凝土抗压强度值 (MPa)")
    fc_type: str = Field(..., description="混凝土类型说明，如 Cube 100 / Cylinder 150x300")
    fy: float = Field(..., description="钢管屈服强度 (MPa)")
    fcy150: str = Field(default="", description="留空")
    r_ratio: float = Field(default=0.0, description="再生骨料比例 (%)。若为普通混凝土填 0。")
    b: float = Field(..., description="宽度/直径/长轴 (mm)。方形为宽度，圆形为直径，圆端形为长轴。")
    h: float = Field(..., description="深度/直径/短轴 (mm)。方形为深度，圆形为直径，圆端形为短轴。")
    t: float = Field(..., description="钢管壁厚 (mm)")
    r0: float = Field(..., description="内圆角半径 (mm)。方形填 0，圆形/圆端形填 h/2。")
    L: float = Field(..., description="长度 (mm)")
    
    # 偏心距需遵循：若未明确区分，e1 = e2 = e。轴压构件为 0。
    e1: float = Field(..., description="上端偏心距 (mm)")
    e2: float = Field(..., description="下端偏心距 (mm)")
    
    n_exp: float = Field(..., description="极限承载力 (kN)")
    source_evidence: str = Field(..., description="数据来源的具体表格或段落引用")

class RefInfo(BaseModel):
    title: str = Field(..., description="论文标题")
    authors: List[str] = Field(..., description="作者列表")
    journal: str = Field(..., description="期刊")
    year: int = Field(..., description="出版年份")

class PaperExtraction(BaseModel):
    """整篇论文提取结果（必须严格遵循此 Schema 输出）"""
    is_valid: bool = Field(..., description="文档是否有效")
    reason: str = Field(..., description="判定的理由")
    
    ref_info: RefInfo
    
    Group_A: List[SpecimenBase] = Field(default_factory=list, description="方形/矩形截面试件 (Square/Rectangular)")
    Group_B: List[SpecimenBase] = Field(default_factory=list, description="圆形截面试件 (Circular)")
    Group_C: List[SpecimenBase] = Field(default_factory=list, description="圆端形/椭圆形截面试件 (Round-ended)")
    
    # 元数据，不对外输出给验证集使用，但供内部调试
    extraction_model: str = Field(default="unknown", description="提取模型")
    extraction_time: str = Field(default="", description="提取时间")
