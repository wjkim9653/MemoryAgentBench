import os
import re
import time
from collections import Counter
from typing import List, Sequence


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _document_text(document) -> str:
    return getattr(document, "page_content", str(document))


class _FallbackRetriever:
    def __init__(self, texts: Sequence[str]):
        self.texts = list(texts)
        self.doc_tokens = [_tokenize(text) for text in self.texts]
        self.doc_counters = [Counter(tokens) for tokens in self.doc_tokens]

    def get_top_n(self, query_tokens: List[str], n: int) -> List[str]:
        query_counts = Counter(query_tokens)
        scored = []
        for index, counter in enumerate(self.doc_counters):
            score = sum(counter[token] * query_counts[token] for token in query_counts)
            scored.append((score, index))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [self.texts[index] for score, index in scored[:n] if score > 0] or self.texts[:n]


class SelfRAG:
    """A dependency-light Self-RAG variant for OpenAI-compatible local LLMs.

    The original file depended on OpenAI embeddings, FAISS, and LangChain
    structured outputs. For local vLLM experiments this implementation keeps the
    Self-RAG control flow but uses BM25-style retrieval and simple text prompts.
    """

    def __init__(
        self,
        documents,
        temperature=0.7,
        top_k=3,
        model_name="gpt-4o-mini",
        provider=None,
        api_base=None,
        api_key=None,
        max_tokens=256,
        force_retrieval=True,
        filter_retrieved=False,
        max_context_chars=24000,
    ):
        start_time = time.time()
        self.documents = [_document_text(document).replace("\t", " ") for document in documents]
        self.temperature = temperature
        self.top_k = top_k
        self.model_name = model_name
        self.provider = provider
        self.api_base = api_base
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.max_tokens = max_tokens
        self.force_retrieval = force_retrieval
        self.filter_retrieved = filter_retrieved
        self.max_context_chars = max_context_chars

        self.uses_rank_bm25 = False
        self.retriever = self._build_retriever(self.documents)
        self.client = self._build_client()
        self.memory_construction_time = time.time() - start_time
        self._memory_time_reported = False

    def _build_client(self):
        from openai import OpenAI

        if self.provider == "openai_compatible":
            return OpenAI(api_key=self.api_key, base_url=self.api_base)
        return OpenAI(api_key=self.api_key)

    def _build_retriever(self, texts):
        if not texts:
            return _FallbackRetriever([])
        try:
            from rank_bm25 import BM25Okapi

            tokenized = [_tokenize(text) for text in texts]
            self.uses_rank_bm25 = True
            return BM25Okapi(tokenized)
        except Exception:
            self.uses_rank_bm25 = False
            return _FallbackRetriever(texts)

    def _chat(self, system_prompt: str, user_prompt: str, max_tokens=None) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _extract_retrieval_query(self, query: str) -> str:
        patterns = [
            r"Now Answer the Question:\s*(.*)",
            r"Here is the conversation:\s*(.*)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.DOTALL)
            if match:
                return match.group(1).strip()
        return query.strip()

    def _retrieve(self, retrieval_query: str) -> List[str]:
        if not self.documents:
            return []
        query_tokens = _tokenize(retrieval_query)
        if self.uses_rank_bm25:
            return self.retriever.get_top_n(query_tokens, self.documents, n=self.top_k)
        return self.retriever.get_top_n(query_tokens, self.top_k)

    def _needs_retrieval(self, retrieval_query: str) -> bool:
        if self.force_retrieval:
            return True
        response = self._chat(
            "You decide whether retrieval is needed. Answer only Yes or No.",
            f"Question:\n{retrieval_query}\n\nIs retrieval needed?",
            max_tokens=4,
        )
        return response.lower().startswith("yes")

    def _filter_contexts(self, retrieval_query: str, contexts: List[str]) -> List[str]:
        if not self.filter_retrieved:
            return contexts

        relevant = []
        for context in contexts:
            response = self._chat(
                "You judge whether a retrieved passage is relevant. Answer only Relevant or Irrelevant.",
                f"Question:\n{retrieval_query}\n\nPassage:\n{context}\n\nRelevant?",
                max_tokens=8,
            )
            if response.lower().startswith("relevant"):
                relevant.append(context)
        return relevant or contexts

    def _build_context_block(self, contexts: List[str]) -> str:
        parts = []
        total_chars = 0
        for index, context in enumerate(contexts, start=1):
            block = f"Passage {index}:\n{context.strip()}"
            if total_chars + len(block) > self.max_context_chars:
                break
            parts.append(block)
            total_chars += len(block)
        return "\n\n".join(parts)

    def _generate(self, query: str, context_block: str) -> str:
        system_prompt = (
            "You are a memory agent answering benchmark questions. "
            "Use only the provided retrieved memory passages and the rules in the question. "
            "If retrieved passages contain numbered conflicting facts, choose the fact with the larger serial number. "
            "Do not use real-world knowledge when it conflicts with the provided memory. "
            "Give a concise answer without extra explanation."
        )
        user_prompt = (
            f"[Retrieved Memory]\n{context_block}\n\n"
            f"[Question]\n{query}\n\n"
            "Answer concisely:"
        )
        return self._chat(system_prompt, user_prompt, max_tokens=self.max_tokens)

    def run(self, query):
        start_time = time.time()
        retrieval_query = self._extract_retrieval_query(query)

        if self._needs_retrieval(retrieval_query):
            contexts = self._retrieve(retrieval_query)
            contexts = self._filter_contexts(retrieval_query, contexts)
        else:
            contexts = []

        context_block = self._build_context_block(contexts)
        if not context_block:
            context_block = "No relevant memory was retrieved."

        response = self._generate(query, context_block)
        memory_construction_time = 0
        if not self._memory_time_reported:
            memory_construction_time = self.memory_construction_time
            self._memory_time_reported = True
        query_time_len = time.time() - start_time
        return response, contexts, memory_construction_time, query_time_len
