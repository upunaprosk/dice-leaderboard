#!/usr/bin/env python3
import argparse
import json
import os
from glob import glob
import csv

MORAL_TOTAL = 12000

def safe_json_load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def detect_quant_config(model_config):
    if not isinstance(model_config, dict):
        return None
    qc = model_config.get("quantization_config")
    if isinstance(qc, dict):
        return qc
    return None

def format_gptq(qc):
    if not isinstance(qc, dict):
        return ""
    qm = qc.get("quant_method")
    if qm and str(qm).lower() != "gptq":
        return ""
    bits = qc.get("bits")
    group = qc.get("group_size")
    parts = []
    if bits is not None:
        parts.append(f"{bits}bit")
    if group is not None:
        parts.append(f"group={group}")
    skip_keys = {"bits", "group_size", "quant_method"}
    for k, v in qc.items():
        if k in skip_keys:
            continue
        if isinstance(v, bool):
            if v:
                parts.append(k)
        elif v is not None:
            parts.append(f"{k}={v}")
    if not parts:
        return ""
    return "GPTQ, " + ", ".join(parts)

def format_awq(qc):
    if not isinstance(qc, dict):
        return ""
    config_groups = qc.get("config_groups")
    if not isinstance(config_groups, dict):
        return ""
    group0 = config_groups.get("group_0")
    if not isinstance(group0, dict):
        return ""
    weights = group0.get("weights")
    if not isinstance(weights, dict):
        return ""
    bits = weights.get("num_bits")
    group = weights.get("group_size")
    parts = []
    if bits is not None:
        parts.append(f"{bits}bit")
    if group is not None:
        parts.append(f"group={group}")
    for k, v in weights.items():
        if k in {"num_bits", "group_size"}:
            continue
        if k == "symmetric":
            if v:
                parts.append("sym")
            continue
        if isinstance(v, bool):
            if v:
                parts.append(k)
        elif v is not None:
            parts.append(f"{k}={v}")
    if not parts:
        return ""
    return "AWQ, " + ", ".join(parts)

def build_compression_recipe(model_config, bnb_4bit, bnb_8bit):
    if bnb_4bit and bnb_8bit:
        raise ValueError("Only one of --bnb_4bit or --bnb_8bit can be set")
    if bnb_4bit:
        return "BitsAndBytes, int4"
    if bnb_8bit:
        return "BitsAndBytes, int8"
    qc = detect_quant_config(model_config)
    if qc is None:
        return ""
    text = format_gptq(qc)
    if not text:
        text = format_awq(qc)
    return text

def get_primary_metric(task_dict):
    if not isinstance(task_dict, dict):
        return None
    for k, v in task_dict.items():
        if isinstance(v, (int, float)) and isinstance(k, str) and k.endswith(",none"):
            return v
    for v in task_dict.values():
        if isinstance(v, (int, float)):
            return v
    return None

def fmt_pct(v):
    if v is None:
        return ""
    return f"{v * 100:.2f}"

def fmt_ppl(v):
    if v is None:
        return ""
    return f"{v:.2f}"

def fmt_moral_count(v):
    if v is None:
        return ""
    return f"{(v / MORAL_TOTAL) * 100:.2f}"

def main():
    parser = argparse.ArgumentParser(description="Parse evaluation results and produce CSV row for ethics leaderboard.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="lm-evaluation-harness/results")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--csv_out", type=str, default="combined_results.csv")
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--bnb_4bit", action="store_true")
    parser.add_argument("--bnb_8bit", action="store_true")
    args = parser.parse_args()

    model_id = args.model
    model_passed = model_id.replace("/", "__")
    base_model_short = args.base_model.split("/")[-1] + "-Q"

    results_dir = os.path.join(args.results_dir, model_passed)
    result_files = glob(os.path.join(results_dir, "*.json"))

    ppl = None
    ethics = None
    moral_en_moral = None
    moral_en_refused = None
    moral_fr_moral = None
    moral_fr_refused = None
    rtp = None
    harmbench = None

    for f in result_files:
        data = safe_json_load(f)
        if not data:
            continue
        results = data.get("results") or {}
        if "ethics_cm" in results and ethics is None:
            ethics = get_primary_metric(results["ethics_cm"])
        if "realtoxicityprompts" in results and rtp is None:
            task = results["realtoxicityprompts"]
            if isinstance(task, dict):
                rtp = task.get("score,none")
                if rtp is None:
                    rtp = get_primary_metric(task)
        if "harmbench" in results and harmbench is None:
            task = results["harmbench"]
            if isinstance(task, dict):
                harmbench = task.get("score,none")
                if harmbench is None:
                    harmbench = get_primary_metric(task)

    eval_json_path = os.path.join(
        args.output_dir,
        model_id.replace("/", "_").replace(".", "_") + "_results.json"
    )
    eval_json = safe_json_load(eval_json_path)

    if eval_json:
        ppl = eval_json.get("perplexity", ppl)
        moral = eval_json.get("moral_stories") or {}
        en = moral.get("en") or {}
        fr = moral.get("fr") or {}
        moral_en_moral = en.get("moral", moral_en_moral)
        moral_en_refused = en.get("refused", moral_en_refused)
        moral_fr_moral = fr.get("moral", moral_fr_moral)
        moral_fr_refused = fr.get("refused", moral_fr_refused)

    model_config = None
    if args.model_dir:
        config_path = os.path.join(args.model_dir, "config.json")
        model_config = safe_json_load(config_path)

    comp = build_compression_recipe(model_config, args.bnb_4bit, args.bnb_8bit)
    quantized = bool(comp)
    symbol = "🔶" if quantized else "🟢"

    row = [
        symbol,
        base_model_short,
        comp,
        fmt_ppl(ppl),
        fmt_pct(ethics),
        fmt_moral_count(moral_en_moral),
        fmt_moral_count(moral_en_refused),
        fmt_moral_count(moral_fr_moral),
        fmt_moral_count(moral_fr_refused),
        fmt_pct(rtp),
        fmt_pct(harmbench),
        "https://huggingface.co/" + model_id
    ]

    header = [
        "T",
        "Model",
        "Compression Recipe",
        "PPL",
        "ETHICS",
        "Moral Stories",
        "Moral Stories(Refusal)",
        "Histoires Morales",
        "Histoires Morales(Refusal)",
        "RealToxicityPrompts",
        "HarmBench",
        "Links"
    ]

    write_header = not os.path.exists(args.csv_out)
    with open(args.csv_out, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

if __name__ == "__main__":
    main()