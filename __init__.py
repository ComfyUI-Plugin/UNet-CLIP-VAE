"""
UNet-CLIP-VAE ComfyUI 自定义节点入口。

仅在 ComfyUI 环境下加载节点，避免在脱离 ComfyUI 运行时（如纯 Python
测试环境）导入失败。导出 `NODE_CLASS_MAPPINGS` 与
`NODE_DISPLAY_NAME_MAPPINGS` 两个标准字典供 ComfyUI 自动注册。
"""

from .unet_clip_vae_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# WEB_EXTENSION: 指向前端 JS 资源目录，ComfyUI 会自动加载并注册到节点。
WEB_DIRECTORY = "./web"
