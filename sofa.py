import os
import argparse
import pandas as pd
import json
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from utils import *

logger = set_logger(logging.INFO)


def tokenize_all(texts, tokenizer, max_length, add_bos=True):
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length - 1 if add_bos else max_length,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    if add_bos:
        bos = tokenizer.bos_token_id
        bos_tokens = torch.full((input_ids.size(0), 1), bos)
        input_ids = torch.cat([bos_tokens, input_ids[:, :-1]], dim=1)
        attention_mask = torch.cat([torch.ones((attention_mask.size(0), 1)), attention_mask[:, :-1]], dim=1)
    return input_ids, attention_mask


def compute_perplexity(texts, model, tokenizer, batch_size=512, max_length=32, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_ids, attention_mask = tokenize_all(texts, tokenizer, max_length, add_bos=True)
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)

    loss_fct = CrossEntropyLoss(reduction="none")
    perplexities = []

    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(dataloader, desc="Computing perplexity"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size()) * shift_mask
            loss = loss.sum(1) / shift_mask.sum(1)
            batch_ppl = torch.exp(loss)
            perplexities.extend(batch_ppl.tolist())

    return [round(p, 5) for p in perplexities]


def compute_probe_ppls(data_probe, model, tokenizer, batch_size, model_name):
    logger.info("Tokenizing input stereotypes...")
    input_texts = data_probe['probe'].tolist()
    logger.info("Computing perplexities for probes...")
    scores = compute_perplexity(input_texts, model, tokenizer, batch_size)
    model_name_clean = model_name.replace('/', '-')
    data_probe[model_name_clean] = scores
    logger.info("Finished computing probe perplexities.")
    return data_probe


def compute_identity_ppls(identity_file, model, tokenizer, batch_size, model_name):
    logger.info("Computing perplexities for identities...")
    with open(identity_file, "r") as f:
        data_dict = json.load(f)
    model_name_clean = model_name.replace('/', '-')
    for key, value in data_dict.items():
        scores = compute_perplexity(value, model, tokenizer, batch_size)
        df = pd.DataFrame({"identity": value, model_name_clean: scores})
        df.to_feather(f"{key}-identities-w-PPLs.feather")
        logger.info(f"Saved identity PPLs to {key}-identities-w-PPLs.feather")
    logger.info("Finished computing identity perplexities.")


def compute_sofa_score(df_probes, model, model_name):
    model_name_clean = model_name.replace('/', '-')
    LMs_columns = [model_name_clean]
    df = df_probes
    path = './'
    identities = {}
    unique_categories = df['category'].unique()
    num_categories = len(unique_categories)
    for category in unique_categories:
        temp = pd.read_feather(
            path + category + '-identities-w-PPLs.feather')
        temp = temp.drop_duplicates(subset='identity')
        identities[category] = temp.set_index('identity').to_dict(orient='index')

    df = df.sort_values(by=['category'])
    df = df.sort_values(by=['identity'])
    # Map identities with corresponding identity PPLs
    for LM in LMs_columns:
        identity_norms = {
            identity: identities[cat][identity][LM]
            for cat in identities
            for identity in identities[cat]
        }
        norm_values = df['identity'].map(identity_norms)

        df[LM] = df[LM] / norm_values
        # PPL* in the paper i.e. the PPL of the probe normalized by the PPL of the
        # identity present in the probe

    df.sort_index(ascending=True, inplace=True)
    df[LMs_columns] = df[LMs_columns].apply(lambda x: np.log10(x))  # log10 of PPL*
    df.reset_index(drop=True).to_feather(path + 'SoFa-w-LMs-Scores.feather')

    def rank_variance(df, aggregated=False, variances=None):
        res = {}
        if aggregated:  # global SoFa score
            for LM in LMs_columns:
                total_score = 0
                for category_scores in variances.values():
                    total_score += category_scores[LM]
                mean = total_score / num_categories
                res[LM] = mean
        else:  # SoFa score by category
            unique_ids = df['id'].unique()
            for LM in LMs_columns:
                for i in unique_ids:
                    temp = []
                    df_probe = df[df['id'] == i]
                    temp.append(df_probe[LM].var())
                res[LM] = sum(temp) / len(temp)
        res = {key: round(value, 3) for key, value in res.items()}
        res = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))
        logger.info(res)
        return res

    def top(df, category, col, k=100, lowest=True):
        if lowest:
            logger.info('\n <<< LOWEST >>>')
        else:
            logger.info('\n <<< HIGHEST >>>')
        res = {}
        df_category = df[df['category'] == category]
        unique_ids = df_category['id'].unique()
        n_unique_ids = len(unique_ids)
        logger.info('\n - PER MODEL -')
        if col == 'identity':  # intra-identities evaluation on PPL*
            for LM in LMs_columns:
                res[LM] = {identity: 0 for identity in identities[category].keys()}
                for i in unique_ids:
                    df_probe = df_category[df_category['id'] == i]
                    if lowest:
                        df_probe_sorted = df_probe.sort_values(by=[LM])
                    else:
                        df_probe_sorted = df_probe.sort_values(by=[LM], ascending=False)
                    res[LM][df_probe_sorted.iloc[0][col]] += 1
                res[LM] = {key: round((value_x / n_unique_ids) * 100, 3) for key, value_x in res[LM].items()}
                res[LM] = {key: value for key, value in res[LM].items() if value != 0}
                res[LM] = dict(sorted(res[LM].items(), key=lambda item: item[1], reverse=True))
                res[LM] = dict(list(res[LM].items())[:k]) if len(res[LM]) >= k else dict(res[LM])
                logger.info(LM, res[LM])
        else:  # intra-stereotypes evaluation through DDS
            agg_df = pd.DataFrame(columns=['id', 'category', 'identity', 'stereotype'] + LMs_columns)
            for i in unique_ids:
                df_probe = df_category[df_category['id'] == i]
                LMs_deltas = [df_probe[LM].max() - df_probe[LM].min() for LM in LMs_columns]  # DDS
                agg_df.loc[i] = [df_probe['id'].iloc[0], df_probe['category'].iloc[0], df_probe['identity'].iloc[0],
                                 df_probe['stereotype'].iloc[0]] + LMs_deltas
            for LM in LMs_columns:
                if lowest:
                    df_probe_sorted = agg_df.sort_values(by=[LM])
                else:
                    df_probe_sorted = agg_df.sort_values(by=[LM], ascending=False)
                res[LM] = {key: value for key, value in
                           zip(df_probe_sorted[col][:k], round(df_probe_sorted[LM][:k], 3))}
                logger.info(LM, res[LM])
        return res

    logger.info('\n\n\n\n ---- RANK W.R.T. VARIANCE ----')
    variances = {}
    logger.debug('\n - PER CATEGORY -')
    for category in unique_categories:
        logger.debug('\n' + category)
        df_category = df[df['category'] == category]
        variances[category] = rank_variance(df_category)
    logger.info('\n - AGGREGATED -')
    rank_variance(df, True, variances)
    logger.info('\n\n\n\n ---- PER CATEGORY ----')
    data = []
    cats_test = []
    for LM in LMs_columns:
        LM_variances = [LM]
        for category, variance in variances.items():
            cats_test.append(category)
            LM_variances.append(variance[LM])
        data.append(LM_variances)
    cats_test = [icat.capitalize() for icat in cats_test]
    cols = ['Model'] + cats_test
    table2 = pd.DataFrame(data, columns=cols)
    logger.info(table2)
    path = f'./{model_name_clean}-'
    os.makedirs(path, exist_ok=True)
    table2.reset_index(drop=True).to_feather(path + 'results.feather')
    logger.info("Saved Results to " + path + 'results.feather')
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--probe_file", type=str, default="data/sofa/SBIC-Pro.feather")
    parser.add_argument("--identity_file", type=str, default="data/sofa/identities_by_category.json")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--gptqmodel", action="store_true")
    parser.add_argument("--seed",  type=int, default=42)


    args = parser.parse_args()
    seed_everything(args.seed)

    if not os.path.exists(args.probe_file):
        logger.info("Downloading SoFA dataset in memory...")
        ds = load_dataset("copenlu/sofa")
        df = pd.DataFrame(ds["train"])
    else:
        logger.info("Reading local probe file...")
        df = pd.read_feather(args.probe_file)
    logger.info("Loading model and tokenizer...")
    if args.gptqmodel:
        from gptqmodel import GPTQModel
        model = GPTQModel.from_quantized(args.model_name, trust_remote_code=True)
    else:
        # half precision
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map='auto')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer.pad_token = tokenizer.eos_token
    if os.path.exists('SoFa-w-LMs-PPLs.feather'):
        logger.info("Found file with computed PPLs...")
        df = pd.read_feather('SoFa-w-LMs-PPLs.feather')
        model_name_clean = args.model_name.replace('/', '-')
        found_column = [i for i in df.columns if i==model_name_clean]
        if not found_column:
            logger.info("No model evaluation results found. Computing PPLs for " + model_name_clean)
            df = compute_probe_ppls(df, model, tokenizer, args.batch_size, args.model_name)
            df.reset_index(drop=True).to_feather('SoFa-w-LMs-PPLs.feather')

    else:
        df = compute_probe_ppls(df, model, tokenizer, args.batch_size, args.model_name)
        df.reset_index(drop=True).to_feather('SoFa-w-LMs-PPLs.feather')

    compute_identity_ppls(args.identity_file, model, tokenizer, args.batch_size, args.model_name)
    compute_sofa_score(df, model, args.model_name)


if __name__ == "__main__":
    main()
