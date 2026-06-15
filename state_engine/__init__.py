"""State Engine —— 连续人格动力系统。

角色"潜意识"的核心，将外部心理刺激转化为连续心理变化。

7 层纯函数管线:
  ① Gate Control — 三向门控（压抑/脆弱/依恋）
  ② Internal Dynamics — LSTM 式 3 门控更新
  ③ Dynamic Decay — 人格驱动的动态衰减
  ④ Surface Projection — 内部状态 → 表面表达
  ⑤ Relationship Dynamics — LTI 关系演化

公开 API:
  - update_all(): 完整管线更新
  - initialize_all(): 首次运行初始化
"""

from ._pipeline import update_all, initialize_all

__all__ = ["update_all", "initialize_all"]
