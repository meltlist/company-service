"""LLM 服务：支持 DeepSeek / Gemini 等多种 API"""
import json
import re
import time
from typing import Any, Optional

import httpx

try:
    import tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    tiktoken = None
    _HAS_TIKTOKEN = False

from config import settings


class LLMConfig:
    """LLM 配置"""

    PROVIDERS = {
        "deepseek": {
            "api_base": "https://api.deepseek.com",
            "models": ["deepseek-chat", "deepseek-coder"],
            "supports_stream": True,
        },
        "gemini": {
            "api_base": "https://generativelanguage.googleapis.com",
            "models": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"],
            "supports_stream": True,
        },
        "openai": {
            "api_base": "https://api.openai.com/v1",
            "models": ["gpt-4o-mini", "gpt-4o"],
            "supports_stream": True,
        },
        "zhipu": {
            "api_base": "https://open.bigmodel.cn/api/paas/v4",
            "models": ["glm-4", "glm-4-flash"],
            "supports_stream": True,
        },
    }

    @classmethod
    def get_provider_config(cls, provider: str) -> dict:
        return cls.PROVIDERS.get(provider, {})


class TokenCounter:
    """Token 计数器（使用 tiktoken，回退到按字符估算）"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.encoding = None
        if _HAS_TIKTOKEN and tiktoken is not None:
            try:
                self.encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.encoding = None

    def count(self, text: str) -> int:
        if self.encoding is not None:
            return len(self.encoding.encode(text))
        # 回退：按 4 字符 ≈ 1 token 估算
        return max(1, len(text) // 4)


class LLMService:
    """LLM 服务"""

    def __init__(self, api_key: str, provider: str = "deepseek", model: str = None):
        self.api_key = api_key
        self.provider = provider
        self.provider_config = LLMConfig.get_provider_config(provider)
        self.model = model or self._get_default_model()
        self.api_base = self.provider_config.get("api_base", settings.DEFAULT_API_BASE)
        self.token_counter = TokenCounter()

    def _get_default_model(self) -> str:
        return self.provider_config.get("models", ["gpt-4o-mini"])[0]

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> dict:
        """发送对话请求"""
        url = f"{self.api_base}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Gemini 特殊处理
        if self.provider == "gemini":
            return self._gemini_chat(messages, temperature, max_tokens, stream)

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()

        # 计算 token
        prompt_tokens = result.get("usage", {}).get("prompt_tokens", 0)
        completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
        total_tokens = result.get("usage", {}).get("total_tokens", 0)

        # 如果 API 没返回，使用估算
        if total_tokens == 0:
            total_tokens = self._estimate_tokens(messages)

        return {
            "content": result["choices"][0]["message"]["content"],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "model": self.model,
        }

    def _gemini_chat(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> dict:
        """Gemini 特殊处理"""
        # Gemini 使用不同的 API 格式
        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}],
            })

        url = f"{self.api_base}/v1beta/models/{self.model}:generateContent"
        params = {"key": self.api_key}

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload, params=params)
            response.raise_for_status()
            result = response.json()

        content = result["candidates"][0]["content"]["parts"][0]["text"]

        # Gemini token 估算
        prompt_text = "\n".join([m["content"] for m in messages])
        prompt_tokens = self._estimate_tokens(prompt_text)
        completion_tokens = self._estimate_tokens(content)

        return {
            "content": content,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "model": self.model,
        }

    def _estimate_tokens(self, messages: list[dict] | str) -> int:
        """估算 token 数量"""
        if isinstance(messages, str):
            return self.token_counter.count(messages) * 2  # 估算增长
        text = "\n".join([m.get("content", "") for m in messages])
        return self.token_counter.count(text)


class RAGService:
    """RAG 问答服务"""

    def __init__(
        self,
        llm_service: LLMService,
        system_prompt: str = None,
    ):
        self.llm = llm_service
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    def generate_answer(
        self,
        query: str,
        context_chunks: list[dict],
        temperature: float = 0.7,
    ) -> dict:
        """基于检索结果生成回答"""
        if not context_chunks:
            return {
                "answer": "抱歉，没有找到相关文档来回答您的问题。",
                "sources": [],
                "usage": {"total_tokens": 0},
            }

        # 构建上下文
        context = self._build_context(context_chunks)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"上下文：\n{context}\n\n问题：{query}"},
        ]

        result = self.llm.chat(messages, temperature=temperature)

        # 提取来源
        sources = self._extract_sources(context_chunks)

        return {
            "answer": result["content"],
            "sources": sources,
            "usage": result["usage"],
            "model": result["model"],
        }

    def _build_context(self, chunks: list[dict], max_len: int = 8000) -> str:
        """构建上下文"""
        context_parts = []
        total_len = 0

        for chunk in chunks:
            content = chunk["content"]
            chunk_len = len(content)

            if total_len + chunk_len > max_len:
                break

            # 添加来源标注
            doc_title = chunk.get("metadata", {}).get("title", "文档")
            page_info = f"[{doc_title}]" if chunk.get("page_number") else f"[{doc_title}]"
            context_parts.append(f"{page_info}\n{content}")
            total_len += chunk_len

        return "\n\n---\n\n".join(context_parts)

    def _extract_sources(self, chunks: list[dict]) -> list[dict]:
        """提取引用来源"""
        sources = []
        seen = set()

        for chunk in chunks:
            doc_id = chunk.get("doc_id", "")
            if doc_id in seen:
                continue
            seen.add(doc_id)

            sources.append({
                "doc_id": doc_id,
                "title": chunk.get("metadata", {}).get("title", "未知文档"),
                "score": chunk.get("score", 0),
                "preview": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
            })

        return sources


DEFAULT_SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请根据提供的上下文信息，准确、简洁地回答用户的问题。

回答要求：
1. 只基于提供的上下文进行回答，不要编造信息
2. 如果上下文中没有相关信息，明确告知用户
3. 回答要条理清晰，重要信息要突出
4. 在回答末尾，可以标注参考的文档来源

请使用中文回答。"""
