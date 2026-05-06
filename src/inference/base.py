"""模型加载器抽象基类"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from transformers import AutoTokenizer


@dataclass
class GenerationResult:
    """结构化生成结果，供推理埋点与后处理复用。"""

    sql: str
    raw_text: str = ""
    usage: Optional[Dict[str, Any]] = None
    response_metadata: Dict[str, Any] = field(default_factory=dict)


class BaseModelLoader(ABC):
    """
    模型加载器抽象基类
    为不同架构的模型提供统一的加载和推理接口，支持未来模型扩展
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模型加载器
        
        Args:
            config: 模型配置信息
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.tokenizer_name_or_path = (
            config.get("tokenizer_name_or_path")
            or config.get("model_path")
            or config.get("model")
        )
        self._counting_tokenizer = None
        self._counting_tokenizer_load_failed = False

    def generate(self, prompt: str, **gen_kwargs) -> GenerationResult:
        """
        生成结构化结果。

        旧版 loader 如未重写此方法，则回退为仅返回 SQL 文本。
        """
        sql = self.generate_sql(prompt, **gen_kwargs)
        return GenerationResult(sql=sql, raw_text=sql)

    def get_counting_tokenizer(self):
        """返回用于 token 统计的 tokenizer。"""
        tokenizer = self.tokenizer
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            return tokenizer

        if self._counting_tokenizer is not None:
            return self._counting_tokenizer
        if self._counting_tokenizer_load_failed:
            return None
        if not self.tokenizer_name_or_path:
            self._counting_tokenizer_load_failed = True
            return None

        try:
            self._counting_tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name_or_path,
                trust_remote_code=True,
            )
        except Exception:
            self._counting_tokenizer_load_failed = True
            return None
        return self._counting_tokenizer

    def count_tokens(self, text: str) -> Optional[int]:
        """使用 tokenizer 统计文本 token 数。"""
        if text is None:
            return None
        tokenizer = self.get_counting_tokenizer()
        if tokenizer is None:
            return None
        try:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            token_ids = tokenizer.encode(text)
        except Exception:
            return None
        return len(token_ids)
    
    @abstractmethod
    def load_model(self, model_path: str, **kwargs):
        """
        加载模型和tokenizer
        
        Args:
            model_path: 模型路径
            **kwargs: 其他加载参数
        """
        pass
    
    @abstractmethod
    def generate_sql(self, prompt: str, **gen_kwargs) -> str:
        """
        根据prompt生成SQL
        
        Args:
            prompt: 输入提示词
            **gen_kwargs: 生成参数
            
        Returns:
            生成的SQL语句
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict:
        """
        返回模型元信息（名称、参数量等）
        
        Returns:
            模型元信息字典
        """
        pass
