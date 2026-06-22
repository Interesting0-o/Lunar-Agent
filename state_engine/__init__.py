"""State Engine —— 连续人格动力系统。

角色"潜意识"的核心，将外部心理刺激转化为连续心理变化。

4 步管线:
  1. Defense Profiles — 二维防御剖面（deactivation / hyperactivation）
     基于 Bowlby (1980) 依恋防御二分法，7 维逐刺激类型敏感度
  2. Residual Dynamics — 残差式状态更新（刺激+耦合驱动，无内建稳态恢复）
  3. Surface Projection — 内部状态 → 表面表达（带惯性混合）
  4. Surface → Internal Feedback — 情绪失调成本 + 面部反馈 + 表达消耗

稳态恢复由时间衰减（_decay.py）在对话间隔中处理。

公开 API:
  - update_all():      完整管线更新（每轮对话）
  - initialize_all():  首次运行初始化
  - apply_time_decay(): 对话间隔中的时间衰减（拉到 setpoint）
"""

from ._pipeline import update_all, initialize_all
from ._decay import apply_time_decay, apply_time_decay_internal, apply_time_decay_relationship

__all__ = [
    "update_all",
    "initialize_all",
    "apply_time_decay",
    "apply_time_decay_internal",
    "apply_time_decay_relationship",
]
