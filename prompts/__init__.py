"""角色与感知的系统提示词模板。

纯数据文件，独立于代码逻辑，可单独编辑。
"""

from .character import SYSTEM_PROMPT
from .perception import PERCEPTION_SYSTEM_PROMPT
from .memory_summery import MEMORY_SYSTEM_PROMPT
from .character_memories import SEED_MEMORIES

__all__ = [
    "SYSTEM_PROMPT", "PERCEPTION_SYSTEM_PROMPT",
    "MEMORY_SYSTEM_PROMPT", "SEED_MEMORIES",
]
