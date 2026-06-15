"""state_engine 内部工具函数：数值稳定化与软激活。"""

import numpy as np


def soft_clamp(
    x: np.ndarray,
    low: float = 0.0,
    high: float = 1.0,
    transition: float = 0.1,
) -> np.ndarray:
    """软饱和裁剪。

    [low, high] 区间内 = np.clip（完全兼容）。
    区间外用 tanh 平滑压回，保留"超出量"信息，有明确渐近线。

    行为（transition=0.1, low=0, high=1）:
      x=1.00   → 1.0000
      x=1.05   → 0.9975
      x=1.10   → 0.9999
      x=2.00   → 1.0000
    """
    upper_delta = x - high
    upper_output = high - transition * np.tanh(upper_delta / transition)

    lower_delta = low - x
    lower_output = low + transition * np.tanh(lower_delta / transition)

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


def _sigmoid_gate(raw: np.ndarray) -> np.ndarray:
    """门控专用 sigmoid 激活：中点居中（raw-0.5），值域 (0, 1）。

    与 np.clip 的差异：硬阈值 → 软阈值，符合心理学"防御机制软启动"。
    """
    return _sigmoid(raw - 0.5)
