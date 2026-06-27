import os
# 仓库根（finrag/ 的上一级）；各模块的 data/ 路径基于它
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = os.path.join(ROOT, 'data')
