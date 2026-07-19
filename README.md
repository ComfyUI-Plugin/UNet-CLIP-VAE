# UNet-CLIP-VAE (ComfyUI 自定义节点)

> 将 ComfyUI 原生的 **Load Diffusion Model (UNet)**、**Load CLIP**、**Load VAE** 三个独立加载器整合为一个节点 `Load UNet+CLIP+VAE (三合一)`，简化工作流连线，提升画布整洁度。

## ✨ 功能特点

- **三合一节点**：一个节点同时输出 `MODEL`、`CLIP`、`VAE` 三个端口，可直接替换原三节点组合使用。
- **完全等价**：参数、默认值、模型列表与原生节点完全一致，加载逻辑直接复用 ComfyUI 公共 API（`comfy.sd` / `comfy.utils` / `folder_paths`）。
- **分组界面**：节点内控件按 UNet → CLIP → VAE 三个区域排序；高级选项（`weight_dtype`、`device`）默认折叠，需要时通过「显示高级选项」展开。
- **中文提示**：每个控件附带 tooltip 说明其归属子加载器及作用。
- **错误处理**：模型未找到 / 加载失败时抛出带有完整上下文的中文错误，方便排错。
- **零依赖**：仅使用 ComfyUI 自带模块，无需额外 pip 包。

## 📦 安装方法

### 方式一：手动安装（推荐用于本地开发）

1. 进入你的 ComfyUI 自定义节点目录：
   ```
   cd ComfyUI/custom_nodes
   ```
2. 将本项目目录拷贝或软链接为 `custom_nodes/UNet-CLIP-VAE`：
   ```
   # 假设项目位于 d:\codeFile\UNet-CLIP-VAE
   mklink /D ComfyUI\custom_nodes\UNet-CLIP-VAE d:\codeFile\UNet-CLIP-VAE
   ```
   或直接把整个文件夹复制到 `custom_nodes` 下。
3. 重启 ComfyUI，刷新网页即可在节点菜单中搜索到 `Load UNet+CLIP+VAE (三合一)`。

### 方式二：通过 ComfyUI-Manager

如果项目已发布到 ComfyUI Registry（`pyproject.toml` 中 `tool.comfy` 已配置），可通过 ComfyUI-Manager 的 `Custom Nodes Manager` 搜索 `UNet-CLIP-VAE` 安装。

## 🧩 节点说明

### Load UNet+CLIP+VAE (三合一)

**节点类名**：`UNetCLIPVAELoader`
**搜索关键词**：`unet clip vae`、`unet+clip+vae`、`三合一加载器`、`load unet clip vae`、`model clip vae`

#### 输入参数

| 参数 | 类型 | 默认 / 取值范围 | 说明 |
| --- | --- | --- | --- |
| `unet_name` | 下拉 | `models/diffusion_models` 或 `models/unet` 下文件 | UNet 扩散模型 |
| `weight_dtype` | 下拉（高级） | `default` / `fp8_e4m3fn` / `fp8_e4m3fn_fast` / `fp8_e5m2` | UNet 权重数据类型，fp8 可省显存 |
| `clip_name` | 下拉 | `models/text_encoders` 或 `models/clip` 下文件 | CLIP 文本编码器 |
| `type` | 下拉 | `stable_diffusion` 等 | CLIP 类型，需与所选模型匹配 |
| `vae_name` | 下拉 | `models/vae` + `vae_approx` 下的 TAESD / pixel_space 等 | VAE 模型 |
| `device` | 下拉（高级） | `default` / `cpu` | CLIP 加载设备 |

#### 输出端口

| 端口 | 类型 | 等价原生节点 |
| --- | --- | --- |
| `MODEL` | `MODEL` | Load Diffusion Model |
| `CLIP` | `CLIP` | Load CLIP |
| `VAE` | `VAE` | Load VAE |

#### 兼容性

- 三个输出端口的类型与原生完全一致，可无缝接入任意依赖 `MODEL` / `CLIP` / `VAE` 的下游节点。
- 节点分类 (`model/loaders`) 与原生加载器一致，方便在右键菜单中找到。
- 若已使用原生三节点组合，可直接删除原节点，把本节点的对应输出连到原下游节点即可。

## 🔄 使用场景

### 场景一：简化标准文生图工作流

原工作流需三个加载器节点 → 模型选择 → 连线到 KSampler，本节点把三步合并为一步，画布上只剩一个加载节点。

### 场景二：快速切换整套模型

切换不同底模时，无需分别打开三个节点修改，所有选项集中在一个节点内，便于对比和调整。

### 场景三：模板化工作流分发

工作流模板中只有一个加载节点，模型名以占位符形式保留，最终用户填入自身路径即可，避免画布上节点过多导致模板复杂。

## ⚠️ 错误处理

- **模型未找到**：抛出 `FileNotFoundError`，提示确认文件路径并刷新模型列表。
- **加载失败**：抛出 `RuntimeError`，包含 `unet_name` / `clip_name` / `vae_name` 等上下文及原始异常堆栈，便于定位是哪个模型出问题。
- **参数非法**：对 `weight_dtype`、`device` 取值做白名单校验，避免传入非法值。

## 🛠️ 实现说明

- `unet_clip_vae_nodes.py` 通过 `comfy.sd.load_diffusion_model` / `comfy.sd.load_clip` / `comfy.sd.VAE` 等公共 API 加载模型；运行时会优先尝试 `from nodes import UNETLoader, CLIPLoader, VAELoader` 复用原生类的静态方法（如 `VAELoader.vae_list` / `VAELoader.load_taesd`），保证 VAE 列表与 TAESD 拼装逻辑与原生完全一致。文件名特意避免使用 `nodes.py`，防止与 ComfyUI 顶层 `nodes` 模块在 import 时发生命名冲突。
- `web/js/unet_clip_vae.js` 为前端扩展，提供节点配色、控件 tooltip 和排序。
- `__init__.py` 设置 `WEB_DIRECTORY = "./web"`，让 ComfyUI 自动加载前端 JS。
- `pyproject.toml` 配置 ComfyUI Registry 元数据，便于通过 ComfyUI-Manager 发布与安装。

## 📁 项目结构

```
UNet-CLIP-VAE/
├── __init__.py                  # 插件入口，注册节点并指定前端目录
├── unet_clip_vae_nodes.py       # UNetCLIPVAELoader 节点实现
├── pyproject.toml               # 项目元数据 (ComfyUI Registry)
├── web/
│   └── js/
│       └── unet_clip_vae.js     # 前端扩展：配色 / tooltip / 控件排序
└── README.md                    # 本文档
```

## 📝 许可证

MIT License
