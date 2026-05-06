"""模型推理统一入口 - 工厂模式 + 批量推理"""
import json
import os
import time
from typing import Dict, List, Any

import yaml
from tqdm import tqdm

from src.inference.loaders.qwen_model_loader import QwenModelLoader
from src.inference.loaders.vllm_openai_loader import VllmOpenAILoader
from src.inference.base import GenerationResult
from src.inference.sql_utils import (
    extract_sql_from_text,
    looks_like_sql_text,
    normalize_floodsql_predicted_sql,
    normalize_spatialsql_predicted_sql,
)


def build_model_run_name(model_name: str, backend: str) -> str:
    """构造带后端信息的运行标识，避免结果互相覆盖。"""
    return f"{model_name}__{backend}"


class ModelLoaderFactory:
    """模型加载器工厂类"""
    
    # 注册模型加载器映射
    _loaders = {
        'QwenModelLoader': QwenModelLoader,
        'VllmOpenAILoader': VllmOpenAILoader,
    }
    
    @classmethod
    def create(cls, model_type: str, config: Dict[str, Any]):
        """
        创建模型加载器实例
        
        Args:
            model_type: 模型类型对应的加载器类名
            config: 模型配置
            
        Returns:
            模型加载器实例
        """
        loader_class = cls._loaders.get(model_type)
        if not loader_class:
            raise ValueError(f"未知的模型加载器类型: {model_type}")
        return loader_class(config)
    
    @classmethod
    def register_loader(cls, name: str, loader_class):
        """
        注册新的模型加载器
        
        Args:
            name: 加载器名称
            loader_class: 加载器类
        """
        cls._loaders[name] = loader_class


class ModelInference:
    """模型推理器 - 批量推理，带进度条"""
    
    def __init__(self, model_config_path: str, eval_config_path: str, eval_config_override: Dict[str, Any] | None = None):
        """
        初始化模型推理器
        
        Args:
            model_config_path: 模型配置文件路径
            eval_config_path: 评估配置文件路径
        """
        # 加载配置
        with open(model_config_path, 'r', encoding='utf-8') as f:
            self.model_config = yaml.safe_load(f)
        
        with open(eval_config_path, 'r', encoding='utf-8') as f:
            self.eval_config = yaml.safe_load(f)

        if eval_config_override:
            self.eval_config = eval_config_override
        
        self.inference_config = self.model_config.get('inference', {})
        self.results_config = self.eval_config.get('results', {})
        self.default_backend = self.model_config.get('default_backend', 'vllm')
        prediction_postprocess = self.eval_config.get('prediction_postprocess', {})
        self.enable_spatialsql_prediction_normalization = bool(
            prediction_postprocess.get('enable_spatialsql_normalization', False)
        )
        self.enable_floodsql_prediction_normalization = bool(
            prediction_postprocess.get('enable_floodsql_normalization', False)
        )

    @staticmethod
    def get_run_name(model_name: str, backend: str) -> str:
        """返回模型在当前后端下的运行名称。"""
        return build_model_run_name(model_name, backend)

    def resolve_model_config(self, model_name: str, backend: str = None):
        """
        解析逻辑模型在指定后端下的真实配置。

        Args:
            model_name: 逻辑模型名称
            backend: 推理后端名称

        Returns:
            (解析后的模型配置, 实际后端名)
        """
        model_info = self.model_config['models'].get(model_name)
        if not model_info:
            raise ValueError(f"未找到模型配置: {model_name}")

        resolved_backend = backend or self.default_backend

        # 兼容旧配置：若不存在 backends，按 transformers 单后端处理
        if 'backends' not in model_info:
            if resolved_backend != 'transformers':
                raise ValueError(
                    f"模型 {model_name} 使用旧版配置，仅支持 transformers 后端，"
                    f"当前请求后端为: {resolved_backend}"
                )
            return model_info, resolved_backend

        backends = model_info.get('backends', {})
        backend_config = backends.get(resolved_backend)
        if not backend_config:
            available = ', '.join(sorted(backends.keys()))
            raise ValueError(
                f"模型 {model_name} 未配置后端 {resolved_backend}。"
                f"可用后端: {available or '无'}"
            )

        shared_config = {k: v for k, v in model_info.items() if k != 'backends'}
        merged_config = {**shared_config, **backend_config}
        merged_config['generation_config'] = {
            **shared_config.get('generation_config', {}),
            **backend_config.get('generation_config', {}),
        }
        merged_config['backend'] = resolved_backend
        merged_config['logical_model_name'] = model_name
        return merged_config, resolved_backend

    @staticmethod
    def _normalize_usage(usage: Dict[str, Any] | None) -> Dict[str, Any] | None:
        if not usage:
            return None
        prompt_tokens = usage.get('prompt_tokens')
        completion_tokens = usage.get('completion_tokens')
        total_tokens = usage.get('total_tokens')
        if total_tokens is None and (
            prompt_tokens is not None or completion_tokens is not None
        ):
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        if prompt_tokens is None and completion_tokens is None and total_tokens is None:
            return None
        return {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
        }

    def _estimate_token_metrics(
        self,
        model_loader,
        prompt: str,
        generation_result: GenerationResult | None,
    ) -> Dict[str, Any]:
        usage = self._normalize_usage(
            generation_result.usage if generation_result is not None else None
        )
        if usage is not None:
            return {
                'input_tokens': usage.get('prompt_tokens'),
                'output_tokens': usage.get('completion_tokens'),
                'total_tokens': usage.get('total_tokens'),
                'measurement_source': 'api_usage',
            }

        raw_output = ""
        if generation_result is not None:
            raw_output = generation_result.raw_text or generation_result.sql or ""

        input_tokens = None
        output_tokens = None
        if hasattr(model_loader, 'count_tokens'):
            input_tokens = model_loader.count_tokens(prompt)
            if raw_output:
                output_tokens = model_loader.count_tokens(raw_output)
            elif input_tokens is not None:
                output_tokens = 0

        total_tokens = None
        if input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        elif input_tokens is not None and not raw_output:
            total_tokens = input_tokens

        if input_tokens is None and output_tokens is None and total_tokens is None:
            source = 'unavailable'
        else:
            source = 'tokenizer_fallback'

        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'measurement_source': source,
        }

    def _build_inference_metrics(
        self,
        model_loader,
        prompt: str,
        generation_result: GenerationResult | None,
        started_at_unix_ms: int,
        finished_at_unix_ms: int,
        latency_ms: float,
        status: str,
    ) -> Dict[str, Any]:
        token_metrics = self._estimate_token_metrics(model_loader, prompt, generation_result)
        return {
            'input_tokens': token_metrics['input_tokens'],
            'output_tokens': token_metrics['output_tokens'],
            'total_tokens': token_metrics['total_tokens'],
            'latency_ms': latency_ms,
            'started_at_unix_ms': started_at_unix_ms,
            'finished_at_unix_ms': finished_at_unix_ms,
            'measurement_source': token_metrics['measurement_source'],
            'status': status,
        }

    @staticmethod
    def _result_status_from_exception(exc: Exception) -> str:
        return 'skipped' if getattr(exc, 'reason_code', None) is not None else 'error'
    
    def run_inference(self, model_name: str, config_type: str,
                     prompts: List[str], data_items: List[Dict],
                     save_dir: str, backend: str = None):
        """
        运行模型推理
        
        Args:
            model_name: 模型名称
            config_type: 配置类型 (base/rag/keyword/full)
            prompts: prompt列表
            data_items: 数据项列表（包含id, question, gold_sql等）
            save_dir: 结果保存目录
            backend: 推理后端名称
        """
        model_info, resolved_backend = self.resolve_model_config(model_name, backend)
        run_name = self.get_run_name(model_name, resolved_backend)

        print(f"\n{'='*70}")
        print(f"模型推理: {run_name} | 配置: {config_type}")
        print(f"{'='*70}\n")

        # 创建模型加载器
        loader_class_name = model_info['loader_class']
        model_loader = ModelLoaderFactory.create(loader_class_name, model_info)
        
        # 加载模型
        model_loader.load_model()
        
        # 准备结果列表
        results = []
        prompt_records = []
        
        # 批量推理
        batch_size = self.inference_config.get('batch_size', 1)
        save_interval = self.inference_config.get('save_interval', 10)
        show_progress = self.inference_config.get('show_progress', True)
        
        # 使用tqdm显示进度
        iterator = tqdm(enumerate(prompts), total=len(prompts), 
                       desc=f"{run_name}-{config_type}",
                       disable=not show_progress)
        
        for idx, prompt in iterator:
            prompt_records.append(self._build_prompt_record(data_items[idx], prompt))
            started_at_unix_ms = time.time_ns() // 1_000_000
            started_perf_ns = time.perf_counter_ns()
            try:
                generation_result = model_loader.generate(prompt)
                finished_at_unix_ms = time.time_ns() // 1_000_000
                latency_ms = (time.perf_counter_ns() - started_perf_ns) / 1_000_000.0

                predicted_sql = self._normalize_prediction(generation_result.sql, data_items[idx])
                inference_metrics = self._build_inference_metrics(
                    model_loader=model_loader,
                    prompt=prompt,
                    generation_result=generation_result,
                    started_at_unix_ms=started_at_unix_ms,
                    finished_at_unix_ms=finished_at_unix_ms,
                    latency_ms=latency_ms,
                    status='success',
                )
                
                # 记录结果
                result_item = {
                    'id': data_items[idx]['id'],
                    'question': data_items[idx]['question'],
                    'gold_sql': data_items[idx]['gold_sql'],
                    'gold_sql_candidates': data_items[idx].get('gold_sql_candidates', []),
                    'predicted_sql': predicted_sql,
                    'metadata': data_items[idx].get('metadata', {}),
                    'source_backend': data_items[idx].get('source_backend'),
                    'target_backend': data_items[idx].get('target_backend'),
                    'source_split': data_items[idx].get('source_split'),
                    'target_table_prefix': data_items[idx].get('target_table_prefix'),
                    'repair_status': data_items[idx].get('repair_status'),
                    'repair_source': data_items[idx].get('repair_source'),
                    'inference_metrics': inference_metrics,
                }
                results.append(result_item)
                
                # 定期保存
                if (idx + 1) % save_interval == 0:
                    self._save_intermediate_results(
                        results,
                        prompt_records,
                        save_dir,
                        run_name,
                        config_type,
                    )
                
            except Exception as e:
                finished_at_unix_ms = time.time_ns() // 1_000_000
                latency_ms = (time.perf_counter_ns() - started_perf_ns) / 1_000_000.0
                print(f"\n错误: 处理第{idx+1}条数据时出错 - {str(e)}")
                inference_metrics = self._build_inference_metrics(
                    model_loader=model_loader,
                    prompt=prompt,
                    generation_result=getattr(e, 'generation_result', None),
                    started_at_unix_ms=started_at_unix_ms,
                    finished_at_unix_ms=finished_at_unix_ms,
                    latency_ms=latency_ms,
                    status=self._result_status_from_exception(e),
                )
                results.append(
                    self._build_failed_result_item(
                        e,
                        data_items[idx],
                        inference_metrics=inference_metrics,
                    )
                )
        
        # 最终保存
        self._save_final_results(results, prompt_records, save_dir, run_name, config_type)

        # 释放模型和清理GPU内存，避免在多个配置/模型之间累积占用
        print("\n" + "="*70)
        print("清理 GPU 内存...")
        print("="*70)
        
        try:
            # 记录清理前的 GPU 内存使用
            try:
                import torch
                if torch.cuda.is_available():
                    before_mem = []
                    for i in range(torch.cuda.device_count()):
                        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                        before_mem.append(f"GPU{i}: {mem_allocated:.2f}GB")
                    print(f"清理前: {', '.join(before_mem)}")
            except Exception:
                pass
            
            # 调用模型加载器的 unload 方法
            if hasattr(model_loader, 'unload'):
                model_loader.unload()
                print("✅ 已卸载模型并清理 GPU 缓存")
            else:
                # 兼容旧版本：手动删除
                if hasattr(model_loader, 'model') and model_loader.model is not None:
                    del model_loader.model
                    model_loader.model = None
                if hasattr(model_loader, 'tokenizer') and model_loader.tokenizer is not None:
                    del model_loader.tokenizer
                    model_loader.tokenizer = None
                print("✅ 已删除模型和 tokenizer 引用")
            
            # 删除整个 model_loader 对象
            del model_loader
            print("✅ 已删除 model_loader 对象")

            # 执行垃圾回收
            try:
                import gc
                gc.collect()
                print("✅ 执行垃圾回收")
            except Exception:
                pass

            # 再次清理所有 GPU 的缓存（确保彻底）
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                    print(f"✅ 已清理 {torch.cuda.device_count()} 个 GPU 的缓存")
                    
                    # 最后一次垃圾回收
                    gc.collect()
                    
                    # 记录清理后的 GPU 内存使用
                    after_mem = []
                    for i in range(torch.cuda.device_count()):
                        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                        after_mem.append(f"GPU{i}: {mem_allocated:.2f}GB")
                    print(f"清理后: {', '.join(after_mem)}")
            except Exception as e:
                print(f"⚠️  GPU 缓存清理出现异常（可忽略）: {e}")
        except Exception as e:
            print(f"⚠️  内存清理出现异常（可忽略）: {e}")
        
        print("="*70)
        print(f"\n推理完成! 共处理 {len(results)} 条数据")
        print(f"结果已保存到: {save_dir}")

        return results

    def _normalize_prediction(self, predicted_sql: str, data_item: Dict[str, Any]) -> str:
        """对数据集专属预测结果做统一归一化，减少后端与方言差异。"""
        cleaned_sql = extract_sql_from_text(predicted_sql)
        if cleaned_sql:
            predicted_sql = cleaned_sql
        elif not looks_like_sql_text(predicted_sql):
            predicted_sql = ""

        enable_spatialsql = getattr(self, "enable_spatialsql_prediction_normalization", False)
        enable_floodsql = getattr(self, "enable_floodsql_prediction_normalization", False)
        if (
            not enable_spatialsql
            and not enable_floodsql
        ):
            return predicted_sql
        dataset_name = data_item.get("dataset")
        metadata = data_item.get("metadata", {})
        if enable_spatialsql and (
            dataset_name == "spatialsql_pg" or metadata.get("split")
        ):
            return normalize_spatialsql_predicted_sql(predicted_sql, metadata)
        if enable_floodsql and (
            dataset_name == "floodsql_pg" or metadata.get("family")
        ):
            return normalize_floodsql_predicted_sql(predicted_sql, metadata)
        return predicted_sql

    @staticmethod
    def _build_prompt_record(data_item: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        return {
            'id': data_item.get('id'),
            'prompt': prompt,
        }

    def _build_failed_result_item(
        self,
        exc: Exception,
        data_item: Dict[str, Any],
        inference_metrics: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """构造推理失败或跳过时的结果项。"""
        result_item = {
            'id': data_item['id'],
            'question': data_item['question'],
            'gold_sql': data_item['gold_sql'],
            'gold_sql_candidates': data_item.get('gold_sql_candidates', []),
            'predicted_sql': '',
            'error': str(exc),
            'metadata': data_item.get('metadata', {}),
            'source_backend': data_item.get('source_backend'),
            'target_backend': data_item.get('target_backend'),
            'source_split': data_item.get('source_split'),
            'target_table_prefix': data_item.get('target_table_prefix'),
            'repair_status': data_item.get('repair_status'),
            'repair_source': data_item.get('repair_source'),
            'skipped': False,
            'inference_metrics': inference_metrics or {},
        }

        reason_code = getattr(exc, 'reason_code', None)
        if reason_code is not None:
            result_item['skipped'] = True
            result_item['skip_reason_code'] = reason_code
            result_item['skip_details'] = {
                'attempts': getattr(exc, 'attempts', None),
                'elapsed_sec': getattr(exc, 'elapsed_sec', None),
            }
            last_error = getattr(exc, 'last_error', None)
            if last_error is not None:
                result_item['skip_details']['last_error_type'] = type(last_error).__name__
                result_item['skip_details']['last_error'] = str(last_error)
        return result_item
    
    @staticmethod
    def _get_prompt_output_file(save_dir: str, temporary: bool = False) -> str:
        filename = "prompts_temp.json" if temporary else "prompts.json"
        return os.path.join(save_dir, filename)

    def _save_intermediate_results(self, results: List[Dict], prompt_records: List[Dict], save_dir: str,
                                 model_name: str, config_type: str):
        """保存中间结果"""
        os.makedirs(save_dir, exist_ok=True)
        del model_name, config_type
        output_file = os.path.join(save_dir, "predictions_temp.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        prompt_file = self._get_prompt_output_file(save_dir, temporary=True)
        with open(prompt_file, 'w', encoding='utf-8') as f:
            json.dump(prompt_records, f, ensure_ascii=False, indent=2)
    
    def _save_final_results(self, results: List[Dict], prompt_records: List[Dict], save_dir: str,
                          model_name: str, config_type: str):
        """保存最终结果"""
        os.makedirs(save_dir, exist_ok=True)
        del model_name, config_type
        output_file = os.path.join(save_dir, "predictions.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        prompt_file = self._get_prompt_output_file(save_dir, temporary=False)
        with open(prompt_file, 'w', encoding='utf-8') as f:
            json.dump(prompt_records, f, ensure_ascii=False, indent=2)

        # 删除临时文件
        temp_file = os.path.join(save_dir, "predictions_temp.json")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        temp_prompt_file = self._get_prompt_output_file(save_dir, temporary=True)
        if os.path.exists(temp_prompt_file):
            os.remove(temp_prompt_file)
