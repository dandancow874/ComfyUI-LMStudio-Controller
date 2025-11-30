import torch
import numpy as np
from PIL import Image
import base64
import io
import requests
import subprocess
import json
import os
import logging
import time

# 设置日志
logger = logging.getLogger("LMS_Controller")

class LMS_CLI_Handler:
    """
    处理与 LM Studio CLI (lms) 的交互
    """
    _model_cache = None
    _last_cache_time = 0
    CACHE_TTL = 10 

    @staticmethod
    def get_lms_path():
        if os.name == 'nt':
            user_home = os.path.expanduser("~")
            candidates = [
                os.path.join(user_home, ".lmstudio", "bin", "lms.exe"),
                os.path.join(user_home, "AppData", "Local", "LM-Studio", "app", "bin", "lms.exe")
            ]
            for path in candidates:
                if os.path.exists(path):
                    return path
        return "lms"

    @staticmethod
    def run_cmd(args, timeout=30):
        lms_path = LMS_CLI_Handler.get_lms_path()
        cmd = [lms_path] + args
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace',
                startupinfo=startupinfo
            )
            return result.returncode == 0, result.stdout, result.stderr
        except Exception as e:
            return False, "", str(e)

    @classmethod
    def get_models(cls):
        """获取并清洗模型列表"""
        if cls._model_cache and (time.time() - cls._last_cache_time < cls.CACHE_TTL):
            return cls._model_cache

        success, stdout, stderr = cls.run_cmd(["ls"], timeout=5)
        if not success:
            logger.error(f"LMS LS Error: {stderr}")
            return ["Error: lms ls failed"]

        models = []
        lines = stdout.strip().splitlines()
        
        BLACKLIST = {
            "size", "ram", "type", "architecture", "model", "path", 
            "llm", "llms", "embedding", "embeddings", "vision", "image",
            "name", "loading", "fetching", "downloaded", "bytes", "date",
            "publisher", "repository", "you", "have", "features"
        }

        for line in lines:
            line = line.strip()
            if not line: continue
            if all(c in "-=*" for c in line): continue

            parts = line.split()
            if not parts: continue
            
            raw_name = parts[0]
            raw_lower = raw_name.lower()

            if raw_lower.rstrip(":") in BLACKLIST: continue
            if raw_lower[0].isdigit() and ("gb" in raw_lower or "mb" in raw_lower): continue

            clean_name = raw_name
            if "/" in clean_name:
                clean_name = clean_name.split("/")[-1]
            if clean_name.lower().endswith(".gguf"):
                clean_name = clean_name[:-5]

            if len(clean_name) < 2: continue
            models.append(clean_name)

        unique_models = sorted(list(set(models)))
        if not unique_models:
            unique_models = ["No models found"]

        cls._model_cache = unique_models
        cls._last_cache_time = time.time()
        return unique_models

    @classmethod
    def load_model(cls, model_name, identifier):
        """加载模型"""
        logger.info(f"LMS: Loading '{model_name}'...")
        args = ["load", model_name, "--identifier", identifier, "--gpu", "max"]
        success, stdout, stderr = cls.run_cmd(args, timeout=120)
        
        if not success:
            logger.error(f"LMS Load Error: {stderr}")
        return success

    @classmethod
    def unload_all(cls):
        """强力卸载所有模型"""
        logger.info("LMS: Unloading ALL models (Clean slate)...")
        success, _, stderr = cls.run_cmd(["unload", "--all"], timeout=20)
        return success


class LMS_VisionController:
    """
    ComfyUI 节点
    """
    _current_loaded_model = None 

    def __init__(self):
        self.cli = LMS_CLI_Handler()

    @classmethod
    def INPUT_TYPES(cls):
        model_list = LMS_CLI_Handler.get_models()
        return {
            "required": {
                "image": ("IMAGE",),
                "user_prompt": ("STRING", {"multiline": True, "default": "Describe this image in detail."}),
                "model_name": (model_list,),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 32768}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 2.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.3, "min": -2.0, "max": 2.0, "step": 0.1}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                # 只有 unload_after，移除了 reload_model
                "unload_after": ("BOOLEAN", {"default": False, "label_on": "Unload", "label_off": "Keep Loaded"}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful AI assistant capable of vision analysis."}),
                "base_url": ("STRING", {"default": "http://localhost:1234/v1"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response_text",)
    FUNCTION = "generate_content"
    CATEGORY = "LM Studio/VLM"

    def tensor_to_base64(self, tensor):
        try:
            img_np = (tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=90)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Image conversion error: {e}")
            return None

    # [修复] 这里的参数列表即使包含 INPUT_TYPES 里没有的参数也没关系
    # 添加 **kwargs 吸收旧节点发送的多余参数，防止错位
    def generate_content(self, image, user_prompt, model_name, 
                         max_tokens, temperature, top_p, top_k, repetition_penalty, frequency_penalty, presence_penalty, 
                         seed, unload_after, system_prompt="", base_url="http://localhost:1234/v1", 
                         reload_model=None, **kwargs):
        
        # 安全检查：如果 base_url 被错误赋值成了 prompt 文本，强制修正
        if "http" not in base_url and "localhost" not in base_url and len(base_url) > 50:
             logger.warning("Detected argument mismatch (base_url received prompt text). Using default URL.")
             base_url = "http://localhost:1234/v1"

        IDENTIFIER = "comfy_vlm_worker"
        
        # --- 自动模型管理逻辑 ---
        # 如果模型名发生了变化，或者当前没有任何记录
        if LMS_VisionController._current_loaded_model != model_name:
            logger.info(f"Auto-switch: '{LMS_VisionController._current_loaded_model}' -> '{model_name}'")
            
            if "Error" in model_name or "No models" in model_name:
                return ("Error: Invalid model selection.",)

            # 1. 卸载所有
            self.cli.unload_all()
            time.sleep(1.0) 

            # 2. 加载新模型
            success = self.cli.load_model(model_name, IDENTIFIER)
            
            if success:
                LMS_VisionController._current_loaded_model = model_name
                time.sleep(2.0)
            else:
                return (f"Error: Failed to load model '{model_name}'.",)
        else:
            logger.info(f"Model '{model_name}' ready. Skipping load.")

        # --- 图像处理 ---
        img_b64 = self.tensor_to_base64(image)
        if not img_b64:
            return ("Error: Image conversion failed.",)

        # --- API 请求 ---
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "model": IDENTIFIER,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "seed": seed,
            "stream": False
        }

        content = ""
        try:
            api_endpoint = f"{base_url.rstrip('/')}/chat/completions"
            logger.info(f"Sending request to {api_endpoint}...")
            
            response = requests.post(api_endpoint, headers=headers, json=payload, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
            else:
                content = f"API Error {response.status_code}: {response.text}"
                logger.error(content)

        except Exception as e:
            content = f"Connection Error: {str(e)}. Is LM Studio running and server started?"
            logger.error(content)

        # --- 运行后处理 ---
        if unload_after:
            self.cli.unload_all()
            LMS_VisionController._current_loaded_model = None
            logger.info("Auto-unload executed.")

        return (content,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "LMS_VisionController": LMS_VisionController
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LMS_VisionController": "LM Studio Vision Controller"
}