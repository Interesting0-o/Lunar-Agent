"""state_engine 内部工具函数：数值稳定化与软激活。"""

import numpy as np


def soft_clamp(
    x: np.ndarray,
    low: float = -1.0,
    high: float = 1.0,
    transition: float = 0.1,
) -> np.ndarray:
    """软饱和裁剪 —— 区间内恒等，边界外 tanh 软饱和。

    与硬 clip 的差异：超出 [low, high] 的值通过 tanh 平滑饱和，
    保证 C¹ 连续且单调。区间内的值完全不变（恒等映射）。

    行为（transition=0.1, low=-1, high=1）:
      x=−∞    → low − transition  ≈ −1.1（tanh 渐近）
      x=−1.00 → −1.0（恒等）
      x=0.00  → 0.0（恒等）
      x=1.00  → 1.0（恒等）
      x=+∞    → high + transition ≈ 1.1（tanh 渐近）
    """
    # tanh 软饱和仅应用于超出边界的值
    upper_delta = x - high
    upper_output = high + transition * np.tanh(upper_delta / transition)

    lower_delta = low - x
    lower_output = low - transition * np.tanh(lower_delta / transition)

    return np.where(
        x > high, upper_output,
        np.where(x < low, lower_output, x)
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，处理 ±∞ 和大数不产生 NaN。"""
    pos_mask = x >= 0
    result = np.empty_like(x, dtype=np.float64)
    result[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    neg_x = x[~pos_mask]
    result[~pos_mask] = np.exp(neg_x) / (1.0 + np.exp(neg_x))
    return result
