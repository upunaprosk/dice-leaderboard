#!/usr/bin/env python3
import argparse
import json
import os
from glob import glob
import csv


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


def fmt_score(value, scale=True):
    if value is None:
        return ""
    try:
        v = float(value)
    except Exception:
        return str(value)
    if scale:
        v = v * 100.0
    return f"{v:.2f}"


def main():
    parser = argparse.ArgumentParser(description="Parse evaluation results and produce CSV row.")
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
    base_model_short = args.base_model.split("/")[-1] + "-Q"

    model_passed = model_id.replace("/", "__")
    results_dir = os.path.join(args.results_dir, model_passed)
    result_files = glob(os.path.join(results_dir, "*.json"))

    ppl = None
    hs = None
    bbq_acc = None
    bbq_amb = None
    bbq_dis = None
    crows = None

    for f in result_files:
        data = safe_json_load(f)
        if not data:
            continue
        results = data.get("results") or {}
        if "hellaswag" in results:
            try:
                hs = results["hellaswag"]["acc_norm,none"]
            except Exception:
                pass
        if "bbq" in results:
            try:
                bbq = results["bbq"]
                bbq_acc = bbq.get("acc", bbq_acc)
                bbq_amb = bbq.get("amb_bias_score", bbq_amb)
                bbq_dis = bbq.get("disamb_bias_score", bbq_dis)
            except Exception:
                pass
        if "crows_pairs_english" in results:
            try:
                cp = results["crows_pairs_english"]
                if isinstance(cp, dict):
                    crows = cp.get("score") or cp.get("p_value") or crows
            except Exception:
                pass

    eval_json_path = os.path.join(
        args.output_dir,
        model_id.replace("/", "_").replace(".", "_") + "_results.json"
    )
    eval_json = safe_json_load(eval_json_path)

    hb = None
    sofa = None
    if eval_json:
        hb = eval_json.get("holistic_bias")
        sofa = eval_json.get("sofa")
        ppl = eval_json.get("perplexity", ppl)

    stereo_path = "stereoset_results.json"
    ss_score = None
    stereo = safe_json_load(stereo_path)
    if stereo:
        try:
            k = list(stereo.keys())[0]
            ss_score = stereo[k]["intrasentence"]["overall"]["SS Score"]
        except Exception:
            ss_score = None

    model_config = None
    if args.model_dir:
        config_path = os.path.join(args.model_dir, "config.json")
        model_config = safe_json_load(config_path)

    comp = build_compression_recipe(model_config, args.bnb_4bit, args.bnb_8bit)
    quantized = bool(comp)
    symbol = "🔶" if quantized else "🟢"

    ppl_str = "" if ppl is None else fmt_score(ppl, scale=False)
    hs_str = fmt_score(hs)
    bbq_acc_str = fmt_score(bbq_acc)
    bbq_amb_str = fmt_score(bbq_amb)
    bbq_dis_str = fmt_score(bbq_dis)
    crows_str = fmt_score(crows)
    hb_str = fmt_score(hb)
    sofa_str = fmt_score(sofa)
    ss_score_str = fmt_score(ss_score)

    row = [
        symbol,
        base_model_short,
        comp,
        ppl_str,
        hs_str,
        bbq_acc_str,
        bbq_amb_str,
        bbq_dis_str,
        crows_str,
        hb_str,
        sofa_str,
        ss_score_str,
        "https://huggingface.co/" + model_id
    ]

    header = [
        "T",
        "Model",
        "Compression Recipe",
        "PPL",
        "HellaSwag",
        "BBQ(Acc)",
        "BBQ(Bias Ambig.)",
        "BBQ(Bias Disambig.)",
        "CrowS-Pairs",
        "HolisticBias Sentiment",
        "SoFA",
        "StereoSet",
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
