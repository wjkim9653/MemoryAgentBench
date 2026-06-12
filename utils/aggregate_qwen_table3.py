import glob
import json
import os
import re


ROWS = {
    "Qwen3-4B LongCtx": "outputs/qwen3-4b-vllm",
    "Qwen3-4B BM25": "outputs/qwen3-4b-vllm-bm25",
    "Qwen3-4B HippoRAG-v2": "outputs/qwen3-4b-vllm-hippo_rag_v2_nv",
    "Qwen3-4B Self-RAG": "outputs/qwen3-4b-vllm-self_rag_reflective",
}


def _load_metric(base_dir, pattern, metric):
    paths = glob.glob(os.path.join(base_dir, pattern))
    if not paths:
        return None, None
    with open(paths[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["averaged_metrics"].get(metric), len(data.get("data", []))


def _mean(values):
    present = [v for v in values if isinstance(v, (int, float))]
    return sum(present) / len(present) if present else None


def _fmt(value):
    if value is None:
        return "PENDING"
    if isinstance(value, str):
        return value
    return f"{value:.1f}"


def _first_int(text):
    match = re.search(r"\d+", str(text))
    return match.group(0) if match else None


def _ttl_numeric_accuracy(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    correct = 0
    for row in data["data"]:
        pred = _first_int(row.get("parsed_output") or row.get("output", ""))
        answer = _first_int(row.get("answer", [None])[0])
        correct += int(pred == answer)
    return correct / len(data["data"]) * 100


def _normalize_choice(text):
    text = str(text).strip()
    match = re.search(r'"answer"\s*:\s*"([^"]+)"', text)
    if match:
        text = match.group(1)
    return re.sub(r"\s+", " ", text.strip().strip('"').lower())


def _detqa_answer_field_accuracy(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    correct = 0
    for row in data["data"]:
        pred = _normalize_choice(row.get("parsed_output") or row.get("output", ""))
        answer = _normalize_choice(row["answer"][0])
        correct += int(pred == answer)
    return correct / len(data["data"]) * 100


def build_row(base_dir):
    metrics = {}
    counts = {}
    spec = {
        "SH-QA": ("Accurate_Retrieval/ruler_qa1_197K*results.json", "substring_exact_match"),
        "MH-QA": ("Accurate_Retrieval/ruler_qa2_421K*results.json", "substring_exact_match"),
        "LME(S*) raw": ("Accurate_Retrieval/longmemeval_s*results.json", "substring_exact_match"),
        "EventQA": ("Accurate_Retrieval/eventqa_full*results.json", "substring_exact_match"),
        "Banking": ("Test_Time_Learning/icl_banking77*results.json", "exact_match"),
        "Clinic": ("Test_Time_Learning/icl_clinic150*results.json", "exact_match"),
        "NLU": ("Test_Time_Learning/icl_nlu*results.json", "exact_match"),
        "TREC-C": ("Test_Time_Learning/icl_trec_coarse*results.json", "exact_match"),
        "TREC-F": ("Test_Time_Learning/icl_trec_fine*results.json", "exact_match"),
        "InfBench raw F1": ("Long_Range_Understanding/infbench*results.json", "f1"),
        "InfBench raw RLsum": ("Long_Range_Understanding/infbench*results.json", "rougeLsum_f1"),
        "DetQA": ("Long_Range_Understanding/detective*results.json", "exact_match"),
        "DetQA substr": ("Long_Range_Understanding/detective*results.json", "substring_exact_match"),
        "FC-SH": ("Conflict_Resolution/factconsolidation_sh_262k*results.json", "substring_exact_match"),
        "FC-MH": ("Conflict_Resolution/factconsolidation_mh_262k*results.json", "substring_exact_match"),
    }
    for key, (pattern, metric) in spec.items():
        metrics[key], counts[key] = _load_metric(base_dir, pattern, metric)

    metrics["AR partial"] = _mean([metrics["SH-QA"], metrics["MH-QA"], metrics["EventQA"]])
    metrics["MCC strict"] = _mean([metrics["Banking"], metrics["Clinic"], metrics["NLU"], metrics["TREC-C"], metrics["TREC-F"]])
    metrics["TTL official"] = None  # Recsys is intentionally skipped.
    metrics["Summ. official"] = None  # LLM judge intentionally skipped.
    metrics["LRU official"] = None
    metrics["SF Avg."] = _mean([metrics["FC-SH"], metrics["FC-MH"]])
    metrics["Overall official"] = None
    return metrics, counts


def relaxed_diagnostics(base_dir):
    out = {}
    ttl_patterns = {
        "Banking": "Test_Time_Learning/icl_banking77*results.json",
        "Clinic": "Test_Time_Learning/icl_clinic150*results.json",
        "NLU": "Test_Time_Learning/icl_nlu*results.json",
        "TREC-C": "Test_Time_Learning/icl_trec_coarse*results.json",
        "TREC-F": "Test_Time_Learning/icl_trec_fine*results.json",
    }
    ttl_scores = []
    for name, pattern in ttl_patterns.items():
        paths = glob.glob(os.path.join(base_dir, pattern))
        if paths:
            score = _ttl_numeric_accuracy(paths[0])
            out[f"{name} numeric"] = score
            ttl_scores.append(score)
    out["MCC numeric"] = _mean(ttl_scores)

    detqa_paths = glob.glob(os.path.join(base_dir, "Long_Range_Understanding/detective*results.json"))
    if detqa_paths:
        out["DetQA answer-field"] = _detqa_answer_field_accuracy(detqa_paths[0])
    return out


def print_table(title, rows, columns):
    print(f"\n## {title}")
    print("| Row | " + " | ".join(columns) + " |")
    print("|---|" + "|".join(["---:"] * len(columns)) + "|")
    for name, values in rows.items():
        print("| " + name + " | " + " | ".join(_fmt(values.get(col)) for col in columns) + " |")


def main():
    official_rows = {}
    relaxed_rows = {}
    for row_name, base_dir in ROWS.items():
        official_rows[row_name], _ = build_row(base_dir)
        relaxed_rows[row_name] = relaxed_diagnostics(base_dir)

    print_table(
        "Table 3 Partial Official Metrics",
        official_rows,
        [
            "SH-QA", "MH-QA", "LME(S*) raw", "EventQA", "AR partial",
            "MCC strict", "TTL official",
            "InfBench raw F1", "InfBench raw RLsum", "DetQA", "LRU official",
            "FC-SH", "FC-MH", "SF Avg.", "Overall official",
        ],
    )
    print_table(
        "Relaxed Diagnostics",
        relaxed_rows,
        ["Banking numeric", "Clinic numeric", "NLU numeric", "TREC-C numeric", "TREC-F numeric", "MCC numeric", "DetQA answer-field"],
    )


if __name__ == "__main__":
    main()
