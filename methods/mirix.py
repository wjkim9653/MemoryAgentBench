import json
import os
import re
import time
from collections import Counter
from typing import Dict, List, Sequence


MEMORY_TYPES = [
    "core",
    "episodic",
    "semantic",
    "procedural",
    "resource",
    "knowledge_vault",
]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _extract_json_array(text: str):
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, list) else []
    except Exception:
        return []


def _extract_json_object(text: str):
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


class MirixMemoryAgent:
    """Prompt-based MIRIX approximation with typed memory stores.

    MIRIX is a multi-agent memory architecture with six memory types. This local
    implementation keeps that behavioral shape without depending on an external
    MIRIX package: a controller classifies updates/retrieval, typed stores keep
    memories, lexical retrieval recalls entries, and a generator answers from
    grouped memories.
    """

    def __init__(
        self,
        model_name: str,
        provider=None,
        api_base=None,
        api_key=None,
        temperature=0.7,
        max_tokens=256,
        retrieve_num=10,
        extraction_max_tokens=512,
        controller_max_tokens=160,
        max_context_chars=24000,
        max_extracted_memories=8,
        use_controller=True,
        use_extractor=True,
    ):
        self.model_name = model_name
        self.provider = provider
        self.api_base = api_base
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retrieve_num = retrieve_num
        self.extraction_max_tokens = extraction_max_tokens
        self.controller_max_tokens = controller_max_tokens
        self.max_context_chars = max_context_chars
        self.max_extracted_memories = max_extracted_memories
        self.use_controller = use_controller
        self.use_extractor = use_extractor
        self.memory_stores: Dict[str, List[dict]] = {memory_type: [] for memory_type in MEMORY_TYPES}
        self.sequence_id = 0
        self.client = self._build_client()

    def _build_client(self):
        from openai import OpenAI

        if self.provider == "openai_compatible":
            return OpenAI(api_key=self.api_key, base_url=self.api_base)
        return OpenAI(api_key=self.api_key)

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

    def memorize(self, text: str):
        self.sequence_id += 1
        self._append_memory("episodic", text, "raw benchmark memory chunk", source="raw")

        if not self.use_extractor:
            return

        memories = self._extract_typed_memories(text)
        for memory in memories[: self.max_extracted_memories]:
            memory_type = str(memory.get("memory_type", "semantic")).strip().lower()
            if memory_type not in MEMORY_TYPES:
                memory_type = "semantic"
            content = str(memory.get("content", "")).strip()
            if not content:
                continue
            rationale = str(memory.get("rationale", "")).strip()
            self._append_memory(memory_type, content, rationale, source="extracted")

    def _append_memory(self, memory_type: str, content: str, rationale: str, source: str):
        self.memory_stores[memory_type].append(
            {
                "id": f"{memory_type}-{len(self.memory_stores[memory_type]) + 1}",
                "sequence_id": self.sequence_id,
                "content": content,
                "rationale": rationale,
                "source": source,
            }
        )

    def _extract_typed_memories(self, text: str) -> List[dict]:
        system_prompt = (
            "You are the MIRIX memory-update controller. Extract durable memories "
            "from the input and assign each to one memory_type: core, episodic, "
            "semantic, procedural, resource, knowledge_vault. "
            "Return only a JSON array. Each item must have memory_type, content, rationale. "
            "Prefer concise factual memories. Keep serial numbers and timestamps when present."
        )
        user_prompt = (
            f"Input memory chunk:\n{text}\n\n"
            f"Return at most {self.max_extracted_memories} typed memories as JSON:"
        )
        response = self._chat(system_prompt, user_prompt, max_tokens=self.extraction_max_tokens, temperature=0)
        return _extract_json_array(response)

    def answer(self, query: str):
        start_time = time.time()
        retrieval_plan = self._plan_retrieval(query)
        memory_types = retrieval_plan.get("memory_types") or MEMORY_TYPES
        search_query = retrieval_plan.get("search_query") or self._extract_retrieval_query(query)
        retrieved = self._retrieve(search_query, memory_types)
        retrieval_context = self._format_retrieval_context(retrieved)
        if not retrieval_context:
            retrieval_context = "No relevant MIRIX memory was retrieved."
        answer = self._generate_answer(query, retrieval_context)
        return answer, retrieval_context, time.time() - start_time

    def _plan_retrieval(self, query: str) -> dict:
        if not self.use_controller:
            return {"memory_types": MEMORY_TYPES, "search_query": self._extract_retrieval_query(query)}
        system_prompt = (
            "You are the MIRIX retrieval controller. Choose memory stores likely "
            "needed for the question. Available stores: core, episodic, semantic, "
            "procedural, resource, knowledge_vault. Return only JSON with keys "
            "memory_types and search_query."
        )
        user_prompt = f"Question:\n{query}\n\nRetrieval plan JSON:"
        response = self._chat(system_prompt, user_prompt, max_tokens=self.controller_max_tokens, temperature=0)
        plan = _extract_json_object(response)
        memory_types = [
            str(memory_type).strip().lower()
            for memory_type in plan.get("memory_types", [])
            if str(memory_type).strip().lower() in MEMORY_TYPES
        ]
        if not memory_types:
            memory_types = MEMORY_TYPES
        return {
            "memory_types": memory_types,
            "search_query": str(plan.get("search_query") or self._extract_retrieval_query(query)),
        }

    def _extract_retrieval_query(self, query: str) -> str:
        patterns = [
            r"Now Answer the Question:\s*(.*)",
            r"Here is the conversation:\s*(.*)",
            r"Question:\s*(.*?)(?:\n\n\s*Answer:|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.DOTALL)
            if match:
                return match.group(1).strip()
        return query.strip()

    def _retrieve(self, query: str, memory_types: Sequence[str]) -> List[dict]:
        query_counts = Counter(_tokenize(query))
        scored = []
        for memory_type in memory_types:
            for entry in self.memory_stores.get(memory_type, []):
                tokens = Counter(_tokenize(entry["content"]))
                score = sum(tokens[token] * query_counts[token] for token in query_counts)
                if score > 0:
                    scored.append((score, entry["sequence_id"], memory_type, entry))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

        if not scored:
            fallback = []
            for memory_type in memory_types:
                fallback.extend((entry["sequence_id"], memory_type, entry) for entry in self.memory_stores.get(memory_type, []))
            fallback.sort(key=lambda item: item[0], reverse=True)
            return [
                {"memory_type": memory_type, **entry}
                for _, memory_type, entry in fallback[: self.retrieve_num]
            ]

        return [
            {"memory_type": memory_type, **entry}
            for _, _, memory_type, entry in scored[: self.retrieve_num]
        ]

    def _format_retrieval_context(self, retrieved: List[dict]) -> str:
        parts = []
        total_chars = 0
        for index, entry in enumerate(retrieved, start=1):
            block = (
                f"Memory {index} [{entry['memory_type']} #{entry['sequence_id']} | {entry['source']}]\n"
                f"{entry['content']}"
            )
            if entry.get("rationale"):
                block += f"\nRationale: {entry['rationale']}"
            if total_chars + len(block) > self.max_context_chars:
                break
            parts.append(block)
            total_chars += len(block)
        return "\n\n".join(parts)

    def _generate_answer(self, query: str, retrieval_context: str) -> str:
        system_prompt = (
            "You are a MIRIX-style memory agent. Answer using only the retrieved "
            "typed memories and the rules in the question. Respect chronological "
            "or serial-number rules when present. Give a concise answer without extra explanation."
        )
        user_prompt = (
            f"[MIRIX Retrieved Memories]\n{retrieval_context}\n\n"
            f"[Question]\n{query}\n\n"
            "Answer:"
        )
        return self._chat(system_prompt, user_prompt, max_tokens=self.max_tokens)

    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "mirix_state.json"), "w") as file:
            json.dump(
                {
                    "sequence_id": self.sequence_id,
                    "memory_stores": self.memory_stores,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

    def load(self, folder: str):
        with open(os.path.join(folder, "mirix_state.json"), "r") as file:
            state = json.load(file)
        self.sequence_id = state.get("sequence_id", 0)
        self.memory_stores = state.get("memory_stores", {memory_type: [] for memory_type in MEMORY_TYPES})
        for memory_type in MEMORY_TYPES:
            self.memory_stores.setdefault(memory_type, [])
