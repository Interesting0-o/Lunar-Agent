"""State Engine —— 连续人格动力系统。

角色"潜意识"的核心，将外部心理刺激转化为连续心理变化。

3 步管线:
  1. Defense Profiles — 防御机制剖面（suppression / vulnerability / attachment）
  2. State Dynamics — 残差式状态更新（含内建稳态恢复）
  3. Surface Projection — 内部状态 → 表面表达

公开 API:
  - update_all(): 完整管线更新
  - initialize_all(): 首次运行初始化
"""

from ._pipeline import update_all, initialize_all

__all__ = [
    "update_all",
    "initialize_all",
]
