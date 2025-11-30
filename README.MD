# ComfyUI-LMStudio-Controller

[English](#english) | [中文](#chinese)

<a name="english"></a>
## 🇬🇧 English

A Custom Node for ComfyUI that integrates **LM Studio**'s CLI (`lms`) to perform Vision Language Model (VLM) inference locally. 

It allows you to use local models (like Qwen-VL, LLaVA, etc.) to analyze images directly within ComfyUI workflows.

### ✨ Features

*   **Smart Model Management**: automatically unloads the previous model and loads the new one when you switch models in the dropdown.
*   **VRAM Friendly**: Includes an "Unload After" option to free up GPU memory immediately after generation.
*   **Full Parameter Control**: Supports Temperature, Top P, Top K, Repetition Penalty, etc.
*   **CLI Integration**: Automatically fetches your downloaded models from LM Studio using `lms ls`.
*   **Clean UI**: Filters out system text from the model list for a clean selection experience.

### ⚙️ Prerequisites (Important)

1.  **Install LM Studio**: Download from [lmstudio.ai](https://lmstudio.ai/).
2.  **Install the CLI Tool**: 
    *   Open LM Studio.
    *   Go to the **"Developer"** tab (or Settings).
    *   Click **"Install lms to PATH"** (or ensure `lms` command works in your terminal).
3.  **Start Local Server**:
    *   In LM Studio, go to the **"Local Server"** tab.
    *   **Start the Server** (Default port: 1234).
    *   *Note: This node uses the CLI to load models, but the API request is sent to the local server port.*

### 📥 Installation

1.  Navigate to your ComfyUI custom nodes directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/dandancow874/ComfyUI-LMStudio-Controller.git
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Restart ComfyUI.

### 🚀 Usage

1.  Search for the node **"LM Studio Vision Controller"**.
2.  Connect an image and a prompt.
3.  **Model Name**: Select a vision model from the dropdown (list is fetched from your LM Studio downloads).
4.  **Unload After**: Check this if you want to release VRAM immediately after the task.

---

<a name="chinese"></a>
## 🇨🇳 中文说明

这是一个 ComfyUI 自定义节点，通过集成 **LM Studio** 的命令行工具 (`lms`)，实现本地视觉大模型 (VLM) 的推理。

它允许你在 ComfyUI 工作流中直接调用本地模型（如 Qwen-VL, LLaVA 等）来识别和描述图像。

### ✨ 主要功能

*   **智能模型管理**：在下拉菜单切换模型时，自动检测并卸载旧模型，加载新模型。
*   **显存友好**：提供 "Unload After"（运行后卸载）选项，生成完成后立即释放显存。
*   **参数全掌控**：支持调节 Temperature, Top P, Top K, 重复惩罚 (Repetition Penalty) 等参数。
*   **CLI 深度集成**：通过 `lms ls` 自动获取你本地下载的所有模型列表。
*   **列表自动清洗**：自动过滤掉无关的系统文本，只显示清晰的模型名称。

### ⚙️ 前置要求 (重要)

1.  **安装 LM Studio**: 请前往 [lmstudio.ai](https://lmstudio.ai/) 下载。
2.  **启用 CLI 工具**: 
    *   打开 LM Studio。
    *   进入 **"Developer" (开发者)** 选项卡（或设置页面）。
    *   点击 **"Install lms to PATH"**（确保在终端输入 `lms` 能看到输出）。
3.  **启动本地服务 (Local Server)**:
    *   在 LM Studio 中，点击左侧的 **"Local Server"** 图标。
    *   点击 **"Start Server"** (默认端口 1234)。
    *   *注意：本节点使用 CLI 加载模型，但通过 API 端口发送图像请求，所以服务必须处于开启状态。*

### 📥 安装方法

1.  进入你的 ComfyUI 自定义节点目录：
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  克隆本项目：
    ```bash
    git clone https://github.com/dandancow874/ComfyUI-LMStudio-Controller.git
    ```
3.  安装依赖：
    ```bash
    pip install -r requirements.txt
    ```
4.  重启 ComfyUI。

### 🚀 使用指南

1.  在 ComfyUI 中搜索节点 **"LM Studio Vision Controller"**。
2.  连接图像 (Image) 和提示词 (Prompt)。
3.  **Model Name**: 从下拉菜单中选择一个视觉模型（列表会自动同步 LM Studio 中已下载的模型）。
4.  **Unload After**: 如果你的显存紧张，建议开启此选项，它会在任务完成后自动卸载模型。

---