"""
UNet-CLIP-VAE ComfyUI 自定义节点实现。

本模块提供一个集成节点 `UNetCLIPVAELoader`，将 ComfyUI 原生
`UNETLoader`、`CLIPLoader`、`VAELoader` 三个独立节点整合为一个节点，
便于在单节点界面中一次性加载扩散模型、文本编码器和 VAE。

设计原则:
  - 通过直接调用原生节点类的方法 / 复用其静态方法，最大程度复用原生逻辑，
    确保参数范围、默认值、输出类型与原生完全一致。
  - 任何加载失败均抛出带有清晰上下文的 `RuntimeError`，便于排错。
  - 当 ComfyUI 顶层 `nodes` 模块不可导入时，回退到使用公共 API
    (`comfy.sd` / `comfy.utils` / `folder_paths`) 重新实现，保证兼容性。

注: 文件名特意使用 `unet_clip_vae_nodes` 而非 `nodes`，避免与
ComfyUI 顶层 `nodes` 模块在 import 时发生命名冲突。
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List, Tuple

import torch

import folder_paths
import comfy.sd
import comfy.utils

# 尝试导入原生节点类。ComfyUI 启动时会先把顶层 `nodes` 模块加载进
# sys.modules，custom_nodes 加载时 `from nodes import ...` 即可拿到原生类。
try:
    from nodes import UNETLoader as _NativeUNETLoader
    from nodes import CLIPLoader as _NativeCLIPLoader
    from nodes import VAELoader as _NativeVAELoader
    _HAS_NATIVE_LOADERS = True
except Exception:  # pragma: no cover - 极端情况下回退到自实现
    _HAS_NATIVE_LOADERS = False
    _NativeUNETLoader = None
    _NativeCLIPLoader = None
    _NativeVAELoader = None


# CLIP 类型列表，与 ComfyUI 原生 CLIPLoader 保持完全一致。
# 当原生类可用时，运行时会动态从原生类读取最新的类型列表，保证同步。
_DEFAULT_CLIP_TYPES: List[str] = [
    "stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi",
    "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma",
    "ace", "omnigen2", "qwen_image", "hunyuan_image", "flux2", "ovis",
    "longcat_image", "cogvideox", "lens", "pixeldit", "ideogram4",
    "boogu", "krea2",
]

# UNet 权重数据类型选项，与原生 UNETLoader 保持一致。
_UNET_WEIGHT_DTYPES: List[str] = [
    "default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2",
]


def _get_clip_type_list() -> List[str]:
    """优先从原生 CLIPLoader 读取 CLIP type 列表，保证与原生完全同步。"""
    if _HAS_NATIVE_LOADERS:
        try:
            native_inputs = _NativeCLIPLoader.INPUT_TYPES()
            type_entry = native_inputs.get("required", {}).get("type")
            if isinstance(type_entry, tuple) and type_entry:
                return list(type_entry[0])
        except Exception:
            pass
    return list(_DEFAULT_CLIP_TYPES)


def _get_unet_weight_dtype_list() -> List[str]:
    """优先从原生 UNETLoader 读取 weight_dtype 列表。"""
    if _HAS_NATIVE_LOADERS:
        try:
            native_inputs = _NativeUNETLoader.INPUT_TYPES()
            dtype_entry = native_inputs.get("required", {}).get("weight_dtype")
            if isinstance(dtype_entry, tuple) and dtype_entry:
                return list(dtype_entry[0])
        except Exception:
            pass
    return list(_UNET_WEIGHT_DTYPES)


def _get_vae_name_list() -> List[str]:
    """获取 VAE 名称列表。优先使用原生 VAELoader.vae_list 静态方法。"""
    if _HAS_NATIVE_LOADERS:
        try:
            return list(_NativeVAELoader.vae_list(_NativeVAELoader))
        except Exception:
            pass
    # 回退实现：仅列出常规 vae 与 vae_approx 中符合 taesd 模式的项
    vaes = list(folder_paths.get_filename_list("vae"))
    approx_vaes = folder_paths.get_filename_list("vae_approx")
    image_taes = ["taesd", "taesdxl", "taesd3", "taef1", "taef2"]
    video_taes = ["taehv", "lighttaew2_2", "lighttaew2_1", "lighttaehy1_5",
                  "taeltx_2"]
    have_img_encoder, have_img_decoder = set(), set()
    for v in approx_vaes:
        parts = v.split("_", 1)
        if len(parts) != 2 or parts[0] not in image_taes:
            for tae in video_taes:
                if v.startswith(tae):
                    vaes.append(v)
                    break
            continue
        if parts[1].startswith("encoder."):
            have_img_encoder.add(parts[0])
        elif parts[1].startswith("decoder."):
            have_img_decoder.add(parts[0])
    vaes += [k for k in have_img_decoder if k in have_img_encoder]
    vaes.append("pixel_space")
    return vaes


def _load_unet(unet_name: str, weight_dtype: str):
    """加载扩散模型，逻辑与原生 UNETLoader.load_unet 完全一致。"""
    model_options: Dict[str, Any] = {}
    if weight_dtype == "fp8_e4m3fn":
        model_options["dtype"] = torch.float8_e4m3fn
    elif weight_dtype == "fp8_e4m3fn_fast":
        model_options["dtype"] = torch.float8_e4m3fn
        model_options["fp8_optimizations"] = True
    elif weight_dtype == "fp8_e5m2":
        model_options["dtype"] = torch.float8_e5m2

    if unet_name not in folder_paths.get_filename_list("diffusion_models"):
        raise FileNotFoundError(
            "UNet 模型未找到: '{}'。请确认文件已放置在 models/unet 或 "
            "models/diffusion_models 目录下，并刷新模型列表。".format(unet_name)
        )
    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
    model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
    if model is None:
        raise RuntimeError(
            "UNet 加载失败: '{}' 不是有效的扩散模型文件。".format(unet_name)
        )
    return model


def _load_clip(clip_name: str, clip_type: str, device: str = "default"):
    """加载 CLIP 文本编码器，逻辑与原生 CLIPLoader.load_clip 一致。"""
    model_options: Dict[str, Any] = {}
    if device == "cpu":
        model_options["load_device"] = torch.device("cpu")
        model_options["offload_device"] = torch.device("cpu")

    if clip_name not in folder_paths.get_filename_list("text_encoders"):
        raise FileNotFoundError(
            "CLIP 模型未找到: '{}'。请确认文件已放置在 models/clip 或 "
            "models/text_encoders 目录下，并刷新模型列表。".format(clip_name)
        )
    clip_type_enum = getattr(
        comfy.sd.CLIPType, clip_type.upper(),
        comfy.sd.CLIPType.STABLE_DIFFUSION,
    )
    clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
    clip = comfy.sd.load_clip(
        ckpt_paths=[clip_path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=clip_type_enum,
        model_options=model_options,
    )
    if clip is None:
        raise RuntimeError(
            "CLIP 加载失败: '{}' 不是有效的文本编码器文件。".format(clip_name)
        )
    return clip


def _load_vae(vae_name: str):
    """加载 VAE，逻辑与原生 VAELoader.load_vae 一致。"""
    image_taes = ["taesd", "taesdxl", "taesd3", "taef1", "taef2"]
    video_taes = ["taehv", "lighttaew2_2", "lighttaew2_1", "lighttaehy1_5",
                  "taeltx_2"]
    metadata = None
    vae_path = None
    sd: Dict[str, Any] = {}

    if vae_name == "pixel_space":
        sd["pixel_space_vae"] = torch.tensor(1.0)
    elif vae_name in image_taes:
        # 复用原生 load_taesd（若可用），否则使用本地实现
        if _HAS_NATIVE_LOADERS:
            sd = _NativeVAELoader.load_taesd(vae_name)
        else:
            sd = _load_taesd_fallback(vae_name)
    else:
        if os.path.splitext(vae_name)[0] in video_taes:
            vae_path = folder_paths.get_full_path_or_raise("vae_approx", vae_name)
        else:
            if vae_name not in folder_paths.get_filename_list("vae"):
                raise FileNotFoundError(
                    "VAE 模型未找到: '{}'。请确认文件已放置在 models/vae "
                    "目录下，并刷新模型列表。".format(vae_name)
                )
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)

    if vae_name == "taef2":
        if metadata is None:
            metadata = {"tae_latent_channels": 128}
        else:
            metadata["tae_latent_channels"] = 128

    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    # 复用原生 VAELoader 的 patcher 工厂注册逻辑（仅在原生可用时）。
    if vae_path is not None and _HAS_NATIVE_LOADERS:
        try:
            vae.patcher.cached_patcher_init = (
                comfy.sd.load_vae_patcher, (vae_path, metadata, None),
            )
        except Exception:
            # 不同 ComfyUI 版本字段可能存在差异，失败时静默忽略，
            # 不影响 VAE 正常使用。
            pass
    return vae


def _load_taesd_fallback(name: str) -> Dict[str, Any]:
    """load_taesd 回退实现：从 vae_approx 中拼装 encoder/decoder。"""
    sd: Dict[str, Any] = {}
    approx_vaes = folder_paths.get_filename_list("vae_approx")
    encoder = next(
        filter(lambda a: a.startswith("{}_encoder.".format(name)), approx_vaes)
    )
    decoder = next(
        filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes)
    )
    enc = comfy.utils.load_torch_file(
        folder_paths.get_full_path_or_raise("vae_approx", encoder)
    )
    for k in enc:
        sd["taesd_encoder.{}".format(k)] = enc[k]
    dec = comfy.utils.load_torch_file(
        folder_paths.get_full_path_or_raise("vae_approx", decoder)
    )
    for k in dec:
        sd["taesd_decoder.{}".format(k)] = dec[k]
    if name == "taesd":
        sd["vae_scale"] = torch.tensor(0.18215)
        sd["vae_shift"] = torch.tensor(0.0)
    elif name == "taesdxl":
        sd["vae_scale"] = torch.tensor(0.13025)
        sd["vae_shift"] = torch.tensor(0.0)
    elif name == "taesd3":
        sd["vae_scale"] = torch.tensor(1.5305)
        sd["vae_shift"] = torch.tensor(0.0)
    elif name == "taef1":
        sd["vae_scale"] = torch.tensor(0.3611)
        sd["vae_shift"] = torch.tensor(0.1159)
    return sd


class UNetCLIPVAELoader:
    """整合 UNet/CLIP/VAE 三合一加载节点。

    输入参数与原生 `UNETLoader` + `CLIPLoader` + `VAELoader` 完全一致，
    输出 ``(MODEL, CLIP, VAE)``，可无缝替换原三节点组合使用。
    """

    # 保留与原生 VAELoader 相同的内部常量，方便扩展或被外部引用。
    video_taes = ["taehv", "lighttaew2_2", "lighttaew2_1", "lighttaehy1_5",
                  "taeltx_2"]
    image_taes = ["taesd", "taesdxl", "taesd3", "taef1", "taef2"]

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Any]:
        return {
            "required": {
                # UNet 区域
                "unet_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                ),
                "weight_dtype": (
                    _get_unet_weight_dtype_list(),
                    {"advanced": True},
                ),
                # CLIP 区域
                "clip_name": (
                    folder_paths.get_filename_list("text_encoders"),
                ),
                "type": (
                    _get_clip_type_list(),
                ),
                # VAE 区域
                "vae_name": (
                    _get_vae_name_list(),
                ),
            },
            "optional": {
                # CLIP 高级选项
                "device": (
                    ["default", "cpu"],
                    {"advanced": True},
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_unet_clip_vae"
    CATEGORY = "model/loaders"
    DESCRIPTION = (
        "整合节点：一次性加载 UNet / CLIP / VAE 三个模型，输出端口与原生"
        "三个独立加载器完全一致，可直接替换原节点组合使用。"
    )
    SEARCH_ALIASES = [
        "unet clip vae", "unet+clip+vae", "三合一加载器",
        "load unet clip vae", "model clip vae",
    ]

    # 节点输出与三个原生节点逐一对应，避免与原生节点同时使用时造成歧义。
    OUTPUT_NODE = False

    def load_unet_clip_vae(
        self,
        unet_name: str,
        weight_dtype: str,
        clip_name: str,
        type: str,
        vae_name: str,
        device: str = "default",
    ) -> Tuple[Any, Any, Any]:
        # 参数基础校验，提前暴露问题。
        if not unet_name:
            raise ValueError("unet_name 不能为空，请先选择扩散模型。")
        if not clip_name:
            raise ValueError("clip_name 不能为空，请先选择文本编码器。")
        if not vae_name:
            raise ValueError("vae_name 不能为空，请先选择 VAE 模型。")
        if weight_dtype not in _get_unet_weight_dtype_list():
            raise ValueError(
                "weight_dtype 取值非法: '{}'" .format(weight_dtype)
            )
        if device not in ("default", "cpu"):
            raise ValueError("device 取值非法: '{}'".format(device))

        # 分别加载三个模型，失败时给出清晰上下文。
        try:
            model = _load_unet(unet_name, weight_dtype)
        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(
                "加载 UNet 失败 (unet_name='{}', weight_dtype='{}')。\n"
                "原始错误: {}\n{}".format(
                    unet_name, weight_dtype, e, traceback.format_exc()
                )
            ) from e

        try:
            clip = _load_clip(clip_name, type, device)
        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(
                "加载 CLIP 失败 (clip_name='{}', type='{}', device='{}')。\n"
                "原始错误: {}\n{}".format(
                    clip_name, type, device, e, traceback.format_exc()
                )
            ) from e

        try:
            vae = _load_vae(vae_name)
        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(
                "加载 VAE 失败 (vae_name='{}')。\n"
                "原始错误: {}\n{}".format(
                    vae_name, e, traceback.format_exc()
                )
            ) from e

        return (model, clip, vae)


# 节点注册表：ComfyUI 通过 __init__.py 暴露此映射加载节点。
NODE_CLASS_MAPPINGS = {
    "UNetCLIPVAELoader": UNetCLIPVAELoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UNetCLIPVAELoader": "Load UNet+CLIP+VAE (三合一)",
}
