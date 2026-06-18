"""
运行时配置 —— 感知节点参数

此模块仅包含系统级配置，不涉及任何 prompt 文本。
prompt 请直接导入对应子模块：
  character.py          — SYSTEM_PROMPT （角色人设）
  perception.py         — PERCEPTION_SYSTEM_PROMPT （感知层提示词）
"""

PERCEPTION_CONFIG = {
    "max_retries": 3,            # 最大重试次数，耗尽则设置 error=True 并结束本轮
    "context_window": 4,         # 提取最近 N 条消息作为上下文

    "retry_emphases": [          # 逐次升压的强调后缀
        "",
        "只输出 JSON，不加任何其他文字。",
        "严重警告：你的输出将被 json.loads() 直接解析。只输出 JSON 对象本身，不要用 ``` 包裹，不要加注释。",
    ],
}
