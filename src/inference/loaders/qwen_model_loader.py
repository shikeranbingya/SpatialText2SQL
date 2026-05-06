"""Qwen系列模型加载器"""
import os
from typing import Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.inference.base import BaseModelLoader, GenerationResult
from src.inference.sql_utils import extract_sql_from_text


class QwenModelLoader(BaseModelLoader):
    """Qwen系列模型加载器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_path = os.path.expanduser(config.get('model_path', ''))
        self.device_map = config.get('device_map', 'auto')
        self.generation_config = config.get('generation_config', {})
    
    def load_model(self, model_path: str = None, **kwargs):
        """
        加载Qwen模型和tokenizer
        
        Args:
            model_path: 模型路径（可选，如果不提供则使用配置中的路径）
            **kwargs: 其他加载参数
        """
        if model_path is None:
            model_path = self.model_path
        
        model_path = os.path.expanduser(model_path)

        print(f"\n加载模型: {model_path}")
        
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            print("✓ Tokenizer加载成功")
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=self.device_map,
                trust_remote_code=True,
                **kwargs
            )
            print(f"✓ 模型加载成功 (device: {self.device_map})")
            
        except Exception as e:
            print(f"✗ 模型加载失败: {str(e)}")
            raise
    
    def generate(self, prompt: str, **gen_kwargs) -> GenerationResult:
        """
        根据 prompt 生成结构化结果。
        
        Args:
            prompt: 输入提示词
            **gen_kwargs: 生成参数（会覆盖配置中的默认参数）
            
        Returns:
            结构化生成结果
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未加载，请先调用load_model()")
        
        # 合并生成配置
        gen_config = {**self.generation_config, **gen_kwargs}
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=gen_config.get('max_new_tokens', 2048),
                temperature=gen_config.get('temperature', 0.0),
                top_p=gen_config.get('top_p', 1.0),
                do_sample=gen_config.get('do_sample', False),
                repetition_penalty=gen_config.get('repetition_penalty', 1.0),
                pad_token_id=self.tokenizer.eos_token_id
            )

        prompt_token_count = inputs["input_ids"].shape[-1]
        completion_tokens = outputs[0][prompt_token_count:]
        completion_text = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return GenerationResult(
            sql=extract_sql_from_text(full_text, prompt),
            raw_text=completion_text,
        )

    def generate_sql(self, prompt: str, **gen_kwargs) -> str:
        """兼容旧接口，仅返回 SQL 文本。"""
        return self.generate(prompt, **gen_kwargs).sql
    
    def unload(self):
        """
        卸载模型并清理 GPU 内存
        """
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # 清理 GPU 缓存
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
        
        # 垃圾回收
        import gc
        gc.collect()
    
    def __del__(self):
        """析构函数：确保对象销毁时清理资源"""
        try:
            self.unload()
        except Exception:
            pass
    
    def get_model_info(self) -> Dict:
        """
        返回模型元信息
        
        Returns:
            模型元信息字典
        """
        info = {
            'model_path': self.model_path,
            'device_map': self.device_map,
            'loaded': self.model is not None
        }
        
        if self.model is not None:
            try:
                # 尝试获取模型参数量
                total_params = sum(p.numel() for p in self.model.parameters())
                info['total_parameters'] = total_params
                info['total_parameters_billions'] = total_params / 1e9
            except:
                pass
        
        return info
