import json
import re

p = r"E:\Work\projects\cfst-ex\testdata\jsondata\A[1-2].json"
with open(p, "r", encoding="utf-8") as f:
    text = f.read()

# 实际上直接忽略外层的无效字段即可。我们只需要里面的 Group_X。
# 这个文件实际上是包含了 prompt 等大量长文本并且因为包含真实的换行符导致无法被 python 严格解析。
# 我们可以粗暴地将它用 demjson 或者通过正则提取出 {} 中的核心 dict 结构。
# 更简单的做法，直接用 eval 或修改 validate_script 的 try-except 来仅仅处理包含完整且合法结构的文件。

# 因为这个 testdata 本身就有问题，我们只提取合法的子集以供交叉验证。
try:
    data = json.loads(text, strict=False)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print("Fixed.")
except Exception as e:
    print(f"Failed to fix with strict=False: {e}")
