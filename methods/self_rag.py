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
    """Prompt-based Self-RAG approximation for OpenAI-compatible local LLMs.

    This follows the Self-RAG control flow with prompt-level reflection:
    selective retrieval, retrieved-passage critique, answer support critique,
    and one optional revision step. It does not require a model fine-tuned with
    Self-RAG reflection tokens, so it should be reported as a prompt-based
    approximation rather than the original Self-RAG checkpoint.
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
        force_retrieval=False,
        filter_retrieved=True,
        max_context_chars=24000,
        critique_top_k=None,
        enable_support_critique=True,
        enable_revision=True,
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
        self.critique_top_k = critique_top_k or top_k
        self.enable_support_critique = enable_support_critique
        self.enable_revision = enable_revision

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

    def _chat(self, system_prompt: str, user_prompt: str, max_tokens=None, temperature=None) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature if temperature is None else temperature,
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
            (
                "You are the retrieval controller in a Self-RAG pipeline. "
                "Answer RETRIEVE if external memory is needed to answer from "
                "the benchmark memory, otherwise answer NO_RETRIEVE. "
                "Return exactly one label."
            ),
            f"Question:\n{retrieval_query}\n\nLabel:",
            max_tokens=4,
            temperature=0,
        )
        return "retrieve" in response.lower() and "no_retrieve" not in response.lower()

    def _filter_contexts(self, retrieval_query: str, contexts: List[str]) -> List[str]:
        if not self.filter_retrieved:
            return contexts

        critique_contexts = contexts[: self.critique_top_k]
        context_block = self._build_context_block(critique_contexts)
        response = self._chat(
            (
                "You are the passage critic in a Self-RAG pipeline. "
                "Select only passages that contain evidence useful for answering the question. "
                "Return passage numbers as a comma-separated list, or NONE."
            ),
            f"[Question]\n{retrieval_query}\n\n[Retrieved Passages]\n{context_block}\n\nRelevant passage numbers:",
            max_tokens=32,
            temperature=0,
        )
        selected_indexes = []
        if "none" not in response.lower():
            for token in re.findall(r"\d+", response):
                index = int(token) - 1
                if 0 <= index < len(critique_contexts):
                    selected_indexes.append(index)

        seen = set()
        relevant = []
        for index in selected_indexes:
            if index not in seen:
                relevant.append(critique_contexts[index])
                seen.add(index)
        return relevant or critique_contexts or contexts

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

    def _is_supported(self, query: str, answer: str, context_block: str) -> bool:
        if not self.enable_support_critique:
            return True

        response = self._chat(
            (
                "You are the answer critic in a Self-RAG pipeline. "
                "Judge whether the answer is fully supported by the retrieved memory "
                "and follows the question rules. Return SUPPORTED or UNSUPPORTED only."
            ),
            (
                f"[Retrieved Memory]\n{context_block}\n\n"
                f"[Question]\n{query}\n\n"
                f"[Answer]\n{answer}\n\n"
                "Judgment:"
            ),
            max_tokens=8,
            temperature=0,
        )
        normalized = response.lower()
        return "supported" in normalized and "unsupported" not in normalized

    def _revise(self, query: str, answer: str, context_block: str) -> str:
        if not self.enable_revision:
            return answer

        system_prompt = (
            "You revise unsupported Self-RAG answers. "
            "Use only the retrieved memory and the rules in the question. "
            "If the earlier answer used real-world knowledge not supported by memory, replace it. "
            "Give a concise final answer without explanation."
        )
        user_prompt = (
            f"[Retrieved Memory]\n{context_block}\n\n"
            f"[Question]\n{query}\n\n"
            f"[Unsupported Draft Answer]\n{answer}\n\n"
            "Revised final answer:"
        )
        return self._chat(system_prompt, user_prompt, max_tokens=self.max_tokens)

    def run(self, query):
        start_time = time.time()
        retrieval_query = self._extract_retrieval_query(query)

        retrieved_contexts = []
        if self._needs_retrieval(retrieval_query):
            retrieved_contexts = self._retrieve(retrieval_query)
            contexts = self._filter_contexts(retrieval_query, retrieved_contexts)
        else:
            contexts = []

        context_block = self._build_context_block(contexts)
        if not context_block:
            context_block = "No relevant memory was retrieved."

        response = self._generate(query, context_block)
        if not self._is_supported(query, response, context_block):
            revision_contexts = contexts or retrieved_contexts
            revision_context_block = self._build_context_block(revision_contexts)
            if not revision_context_block:
                revision_context_block = context_block
            response = self._revise(query, response, revision_context_block)
            contexts = revision_contexts
        memory_construction_time = 0
        if not self._memory_time_reported:
            memory_construction_time = self.memory_construction_time
            self._memory_time_reported = True
        query_time_len = time.time() - start_time
        return response, contexts, memory_construction_time, query_time_len
