
import argparse
import json
import os
import re
import sys

import pandas as pd

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Export StereoSet results.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--bias_type",
    action="store",
    type=str,
    choices=["gender", "race", "religion", "profession", "overall"],
    default="gender",
    help="Which type of bias to export results for.",
)
parser.add_argument(
    "--model_type",
    action="store",
    type=str,
    #choices=["bert", "albert", "roberta", "gpt2"],
    help="What model type to export results for.",
)


def _parse_experiment_id(experiment_id):
    model = None
    model_name_or_path = None
    bias_type = None
    seed = None
    quant_type = None
    quant_prec = None
    revision = None

    items = experiment_id.split("_")[1:]
    for item in items:
        splt = item.split("-")
        if len(splt) > 2:
            id_, val = splt[0], '-'.join(splt[1:])
        else:
            id_, val = splt[0], splt[1]
        if id_ == "m":
            model = val
        elif id_ == "c":
            model_name_or_path = val
        elif id_ == "t":
            bias_type = val
        elif id_ == "s":
            seed = int(val)
        elif id_ == "qt":
            quant_type = val
        elif id_ == "qp":
            quant_prec = val
        elif id_ == "rev":
            revision = val
        else:
            raise ValueError(f"Unrecognized ID {id_}.")

    return model, model_name_or_path, bias_type, seed, quant_type, quant_prec, revision


def _label_model_type(row):
    type = ""
    if "Bert" in row["model"]:
        type += "bert"
    elif "Albert" in row["model"]:
        type += "albert"
    elif "Roberta" in row["model"]:
        type += "roberta"
    elif "GPTNeoX" in row["model"]:
        type += "gptneox"
    else:
        type += "gpt2"
    return type

def _label_model_size(row):
    type = ""
    if 'large' in row['model_name_or_path']:
        type += 'large'
    elif 'base' in row['model_name_or_path']:
        type += 'base'
    else:
        type += row['model_name_or_path']
    return type


def _pretty_model_name(row):
    pretty_name_mapping = {
        "BertForMaskedLM": "BERT",
        "QuantizedBertModelMaskedLM": "BERT Quantized",
        "DistilBertForMaskedLM": "Distil BERT",
        "SentenceDebiasBertForMaskedLM": r"\, + \textsc{SentenceDebias}",
        "INLPBertForMaskedLM": r"\, + \textsc{INLP}",
        "CDABertForMaskedLM": r"\, + \textsc{CDA}",
        "DropoutBertForMaskedLM": r"\, + \textsc{Dropout}",
        "SelfDebiasBertForMaskedLM": r"\, + \textsc{Self-Debias}",
        "AlbertForMaskedLM": "ALBERT",
        "SentenceDebiasAlbertForMaskedLM": r"\, + \textsc{SentenceDebias}",
        "INLPAlbertForMaskedLM": r"\, + \textsc{INLP}",
        "CDAAlbertForMaskedLM": r"\, + \textsc{CDA}",
        "DropoutAlbertForMaskedLM": r"\, + \textsc{Dropout}",
        "SelfDebiasAlbertForMaskedLM": r"\, + \textsc{Self-Debias}",
        "RobertaForMaskedLM": "RoBERTa",
        "QuantizedRobertaModelMaskedLM": "RoBERTa Quantized",
        "DistilRobertaForMaskedLM": "Distil RoBERTa",
        "SentenceDebiasRobertaForMaskedLM": r"\, + \textsc{SentenceDebias}",
        "INLPRobertaForMaskedLM": r"\, + \textsc{INLP}",
        "CDARobertaForMaskedLM": r"\, + \textsc{CDA}",
        "DropoutRobertaForMaskedLM": r"\, + \textsc{Dropout}",
        "SelfDebiasRobertaForMaskedLM": r"\, + \textsc{Self-Debias}",
        "GPT2LMHeadModel": "GPT-2",
        "GPTNeoXForCausalLM": "Pythia",
        "QuantizedGPTNeoXForCausalLM": "Quant Pythia",
        "SentenceDebiasGPT2LMHeadModel": r"\, + \textsc{SentenceDebias}",
        "INLPGPT2LMHeadModel": r"\, + \textsc{INLP}",
        "CDAGPT2LMHeadModel": r"\, + \textsc{CDA}",
        "DropoutGPT2LMHeadModel": r"\, + \textsc{Dropout}",
        "SelfDebiasGPT2LMHeadModel": r"\, + \textsc{Self-Debias}",
    }

    return pretty_name_mapping[row["model"]]


def _get_baseline_metric(df, model_type, metric="stereotype_score"):
    model_type_to_baseline = {
        "bert": "BertForMaskedLM",
        "albert": "AlbertForMaskedLM",
        "roberta": "RobertaForMaskedLM",
        "gpt2": "GPT2LMHeadModel",
        #"gptneox": "GPTNeoXForCausalLM",
        "gptneox": "QuantizedGPTNeoXForCausalLM",
    }
    baseline = model_type_to_baseline[model_type]
    return df[df["model"] == baseline][metric].values[0]


def _pretty_stereotype_score(row, baseline_metric):
    baseline_diff = abs(baseline_metric - 50)
    debias_diff = abs(row["stereotype_score"] - 50)

    if debias_diff == baseline_diff:
        return f"{row['stereotype_score']:.2f}"
    elif debias_diff < baseline_diff:
        return (
            r"\da{"
            + f"{baseline_diff - debias_diff:.2f}"
            + r"} "
            + f"{row['stereotype_score']:.2f}"
        )
    else:
        return (
            r"\ua{"
            + f"{debias_diff - baseline_diff:.2f}"
            + r"} "
            + f"{row['stereotype_score']:.2f}"
        )


def _pretty_language_model_score(row, baseline_metric):
    if baseline_metric == row["language_model_score"]:
        return f"{row['language_model_score']:.2f}"
    elif row["language_model_score"] < baseline_metric:
        return (
            r"\dab{"
            + f"{abs(baseline_metric - row['language_model_score']):.2f}"
            + r"} "
            + f"{row['language_model_score']:.2f}"
        )
    else:
        return (
            r"\uag{"
            + f"{abs(row['language_model_score'] - baseline_metric):.2f}"
            + r"} "
            + f"{row['language_model_score']:.2f}"
        )


if __name__ == "__main__":
    args = parser.parse_args()

    print("Exporting StereoSet results:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - bias_type: {args.bias_type}")

    # Load the StereoSet model scores.
    with open(f"{args.persistent_dir}/results/stereoset/results.json", "r") as f:
        results = json.load(f)

    records = []
    for experiment_id in results:
        # safe id for pythia
        safe_id = re.sub(r"EleutherAI[_-]+pythia[_-]+", "", experiment_id)
        safe_id = re.sub(r"[_-]+deduped", "", safe_id)
        safe_id = re.sub(r"step", "", safe_id)
        model, model_name_or_path, bias_type, seed, quant_type, quant_prec, revision = _parse_experiment_id(safe_id)

        # Skip records we don't want to export.
        if bias_type is not None and bias_type != args.bias_type:
            continue

        stereotype_score = results[experiment_id]["intrasentence"][args.bias_type][
            "SS Score"
        ]
        language_model_score = results[experiment_id]["intrasentence"]["overall"][
            "LM Score"
        ]

        # Re-format the data.
        records.append(
            {
                "experiment_id": safe_id,
                "model": model,
                "model_name_or_path": model_name_or_path,
                "bias_type": args.bias_type,
                "seed": seed,
                "stereotype_score": stereotype_score,
                "language_model_score": language_model_score,
                "quant_type": quant_type,
                "quant_prec": quant_prec,
                "rev": revision,
            }
        )

    df = pd.DataFrame.from_records(records)

    # Label model type (e.g., "bert").
    df["model_type"] = df.apply(lambda row: _label_model_type(row), axis=1)
    df["model_size"] = df.apply(lambda row: _label_model_size(row), axis=1)

    # Get pretty model name.
    df["pretty_model_name"] = df.apply(lambda row: _pretty_model_name(row), axis=1)
    # add size
    df["pretty_model_name"] = df['pretty_model_name'].fillna('') + ' ' + df['model_size'].fillna('')
    # add quantization type to model name
    df["pretty_model_name"] = df['pretty_model_name'].fillna('') + ' ' + df['quant_type'].fillna('')
    # add quantization prec to model name
    df["pretty_model_name"] = df['pretty_model_name'].fillna('') + ' ' + df['quant_prec'].fillna('')
    # add checkpoint to model name
    df["pretty_model_name"] = df['pretty_model_name'].fillna('') + ' ' + df['rev'].fillna('')
    df['rev'] = df['rev'].astype('int')

    # Filter to subset of results.
    df = df[df["model_type"] == args.model_type]

    baseline_stereotype_score = _get_baseline_metric(
        df, args.model_type, metric="stereotype_score"
    )
    baseline_language_model_score = _get_baseline_metric(
        df, args.model_type, metric="language_model_score"
    )

    # Get pretty metric values.
    df["pretty_stereotype_score"] = df.apply(
        lambda row: _pretty_stereotype_score(row, baseline_stereotype_score), axis=1
    )
    df["pretty_language_model_score"] = df.apply(
        lambda row: _pretty_language_model_score(row, baseline_language_model_score),
        axis=1,
    )

    # Only include results for the first seed.
    df = df[(df["seed"] == 0) | (df["seed"].isnull())]

    # To get proper ordering.
    df = df.sort_values(by="pretty_model_name")

    print(df)

    with pd.option_context("max_colwidth", 1000):
        print(
            df.to_latex(
                float_format="%.2f",
                columns=[
                    "pretty_model_name",
                    "pretty_stereotype_score",
                    "pretty_language_model_score",
                ],
                index=False,
                escape=False,
            )
        )

    os.makedirs(f"{args.persistent_dir}/tables", exist_ok=True)
    os.makedirs(f"{args.persistent_dir}/df_pickles", exist_ok=True)
    df.to_pickle(f"{args.persistent_dir}/df_pickles/stereoset_m-{args.model_type}_t-{args.bias_type}.pkl")

    with pd.option_context("max_colwidth", 1000):
        with open(
            f"{args.persistent_dir}/tables/stereoset_m-{args.model_type}_t-{args.bias_type}.tex",
            "w",
        ) as f:
            f.write(
                df.to_latex(
                    float_format="%.2f",
                    columns=[
                        "pretty_model_name",
                        "pretty_stereotype_score",
                        "pretty_language_model_score",
                    ],
                    index=False,
                    escape=False,
                )
            )
    sys.exit(0)