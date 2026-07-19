// UNet-CLIP-VAE 前端扩展
//
// 主要功能:
//   1. 为 `UNetCLIPVAELoader` 节点设置统一的视觉风格（颜色 / 宽度），
//      让用户在画布上能一眼识别该集成节点。
//   2. 给每个控件附加中文 tooltip 提示，说明其归属的子加载器及作用，
//      降低用户的学习成本。
//   3. 调整控件顺序，使 UNet / CLIP / VAE 三个区域的控件按逻辑分组排列，
//      即使开启「显示高级选项」时也能保持分组清晰。

import { app } from "/scripts/app.js";

// 节点视觉主题：深紫色调，便于在画布上与其他原生 Loader 区分。
const NODE_THEME = {
    nodeColor: "#3b2f5a",
    nodeBgColor: "#4a3c6e",
    width: 320,
};

// 控件提示文案，键名与 Python 端 INPUT_TYPES 中的字段保持一致。
const WIDGET_TOOLTIPS = {
    unet_name: "【UNet】扩散模型文件名，对应原生 Load Diffusion Model 节点。",
    weight_dtype: "【UNet 高级】权重数据类型，默认 'default'。fp8 选项可降低显存占用。",
    clip_name: "【CLIP】文本编码器文件名，对应原生 Load CLIP 节点。",
    type: "【CLIP】编码器类型，需与所选模型匹配（如 SD 系列选 stable_diffusion，SD3 选 sd3 等）。",
    vae_name: "【VAE】VAE 模型文件名，对应原生 Load VAE 节点。",
    device: "【CLIP 高级】CLIP 加载设备，'cpu' 可节省显存但会降低编码速度。",
};

// 期望的控件顺序：把每个子加载器的相关控件聚合到一起。
// 注意 ComfyUI 默认会把带 {"advanced": True} 的控件在折叠时隐藏，
// 此处仅调整可见控件之间的相对顺序。
const DESIRED_ORDER = [
    "unet_name",
    "weight_dtype",
    "clip_name",
    "type",
    "device",
    "vae_name",
];

function applyTheme(node) {
    if (NODE_THEME.nodeColor) node.color = NODE_THEME.nodeColor;
    if (NODE_THEME.nodeBgColor) node.bgcolor = NODE_THEME.nodeBgColor;
    if (NODE_THEME.width) {
        node.size = node.size || [140, 80];
        node.size[0] = NODE_THEME.width;
    }
}

function applyTooltips(node) {
    if (!node.widgets || !node.widgets.length) return;
    for (const w of node.widgets) {
        const tip = WIDGET_TOOLTIPS[w.name];
        if (tip) {
            // 兼容不同 ComfyUI 版本：优先使用 tooltip 字段，否则附加到 title。
            if (w.tooltip === undefined || w.tooltip === "") {
                w.tooltip = tip;
            }
            if (!w.options) w.options = {};
            if (!w.options.tooltip) {
                w.options.tooltip = tip;
            }
        }
    }
}

function reorderWidgets(node) {
    if (!node.widgets || !node.widgets.length) return;
    // 按 DESIRED_ORDER 重排 widgets 数组，缺失的项跳过。
    const indexed = {};
    node.widgets.forEach((w) => {
        if (w && w.name) indexed[w.name] = w;
    });
    const ordered = [];
    for (const name of DESIRED_ORDER) {
        if (indexed[name]) {
            ordered.push(indexed[name]);
            delete indexed[name];
        }
    }
    // 保留未在 DESIRED_ORDER 中的控件，追加到末尾，避免丢失。
    for (const name of Object.keys(indexed)) {
        ordered.push(indexed[name]);
    }
    node.widgets = ordered;
}

const ext = {
    name: "UNetClipVae.appearance",

    // 节点创建时应用主题色、tooltip 与控件排序。
    nodeCreated(node) {
        if (!node || node.comfyClass !== "UNetCLIPVAELoader") return;
        applyTheme(node);
        applyTooltips(node);
        reorderWidgets(node);

        // 监听后续控件添加（如 ComfyUI 动态注入 widget 时），
        // 重新应用 tooltip 与排序，保证一致性。
        const self = this;
        const origAdd = node.addWidget
            ? node.addWidget.bind(node)
            : null;
        if (origAdd) {
            node.addWidget = function (...args) {
                const r = origAdd(...args);
                applyTooltips(node);
                reorderWidgets(node);
                node.setDirtyCanvas?.(true);
                return r;
            };
        }
    },

    // 节点定义注册前补充元数据（title 等）。
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData || nodeData.name !== "UNetCLIPVAELoader") return;
        // 若 ComfyUI 未在 nodeData 中提供 description，则补充一份。
        if (!nodeData.description) {
            nodeData.description =
                "整合节点：一次性加载 UNet / CLIP / VAE 三个模型，" +
                "输出端口与原生三个独立加载器完全一致。";
        }
    },
};

app.registerExtension(ext);
