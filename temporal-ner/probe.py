import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import argparse
import importlib
from tqdm import tqdm
import json
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cls_representations(model, tokenizer, dataset, batch_size=16):
    model.eval()
    cls_vectors = []
    labels = []

    dataloader = DataLoader(dataset, batch_size=batch_size)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting CLS"):
            real_batch_size = batch["input_ids"].size(0)
            batch["year_index"] = torch.randint(0, num_years, (real_batch_size,), device=DEVICE)
            inputs = {
                "input_ids": batch["input_ids"].to(DEVICE),
                "attention_mask": batch["attention_mask"].to(DEVICE),
                "token_labels": {
                    task: labels.to(DEVICE)
                    for task, labels in batch["token_targets"].items()
                },
                "date_indices": batch["date_indices"].to(DEVICE) if "date_indices" in batch else None,
                "year_index": batch["year_index"].to(DEVICE),
            }

            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            last_hidden_state = outputs.hidden_states[-1] if outputs.hidden_states else outputs.last_hidden_state
            cls_vecs = last_hidden_state[:, 0, :].cpu().numpy()

            cls_vectors.extend(cls_vecs)
            labels.extend(inputs["year_index"].cpu().numpy())
    return np.array(cls_vectors), np.array(labels)


def train_probe(x_train, y_train, x_test, y_test):
    clf = LogisticRegression(max_iter=5000, multi_class="multinomial")
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    acc = accuracy_score(y_test, preds)
    return acc


def run_probe(checkpoint, args, model_class, tokenizer, train_dataset, test_dataset, num_token_labels_dict, num_years):
    config = AutoConfig.from_pretrained(os.path.join(checkpoint, "best_checkpoint"),
                                        problem_type="single_label_classification",
                                        local_files_only=True)

    model = model_class(config, num_token_labels_dict=num_token_labels_dict,
                        temporal_fusion_strategy=args.temporal_fusion_strategy,
                        num_years=num_years).to(DEVICE)

    model_ckpt = torch.load(os.path.join(checkpoint, "best_checkpoint", "pytorch_model.bin"))
    model.load_state_dict(model_ckpt)

    x_train_vecs, y_train_vecs = get_cls_representations(model, tokenizer, train_dataset)
    x_test_vecs, y_test_vecs = get_cls_representations(model, tokenizer, test_dataset)

    return train_probe(x_train_vecs, y_train_vecs, x_test_vecs, y_test_vecs)


def parse_checkpoint_metadata(checkpoint_path):
    parts = checkpoint_path.split('/')[-1].split('_')
    use_relative_year = parts[-1].split(".")[-1].lower() == 'true'
    temporal_fusion_strategy = parts[-1].split(".")[-2]
    max_sequence_len = int(parts[-5])
    model_name = "dbmdz/bert-base-historic-multilingual-cased" if "historic" in checkpoint_path else "bert-base-multilingual-cased"

    return use_relative_year, temporal_fusion_strategy, max_sequence_len, model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_class", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--folder", type=str)
    parser.add_argument("--output_tsv", type=str, default="probing_results.tsv")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of probing runs per experiment.")
    args = parser.parse_args()

    module_name, class_name = args.model_class.rsplit(".", 1)
    model_class = getattr(importlib.import_module(module_name), class_name)
    output_tsv = os.path.join(args.folder, args.output_tsv)

    from dataset import NewsDataset

    results = []

    checkpoints = [args.checkpoint] if args.checkpoint else [
        os.path.join(args.folder, ckpt) for ckpt in os.listdir(args.folder)
        if os.path.isdir(os.path.join(args.folder, ckpt)) and os.path.exists(
            os.path.join(args.folder, ckpt, "best_checkpoint"))
    ]

    for checkpoint in checkpoints:
        use_relative_year, strategy, max_len, model_name = parse_checkpoint_metadata(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(checkpoint, "best_checkpoint"))
        setattr(args, "use_relative_year", use_relative_year)
        setattr(args, "temporal_fusion_strategy", strategy)
        setattr(args, "max_sequence_len", max_len)
        setattr(args, "model_name_or_path", model_name)

        label_map_path = os.path.join(checkpoint, "label_map.json")
        if os.path.exists(label_map_path):
            label_map = json.load(open(label_map_path))
        else:
            label_map = None

        train_dataset = NewsDataset(args.train_dataset, tokenizer, max_len, args=args, label_map=label_map)
        test_dataset = NewsDataset(args.test_dataset, tokenizer, max_len, args=args,
                                   label_map=train_dataset.get_label_map())
        label_map = train_dataset.get_label_map()
        if not os.path.exists(label_map_path):
            with open(label_map_path, "w") as f:
                json.dump(label_map, f)

        num_sequence_labels, num_token_labels_dict, num_years = train_dataset.get_info()

        for run_id in range(args.num_runs):
            acc = run_probe(checkpoint, args, model_class, tokenizer, train_dataset, test_dataset,
                            num_token_labels_dict, num_years)
            print(
                f"[{run_id + 1}/{args.num_runs}] Accuracy: {acc} | model={model_name}, strategy={strategy}, "
                f"use_relative_year={use_relative_year}, len={max_len}")

            results.append({
                "checkpoint": checkpoint,
                "accuracy": acc,
                "temporal_fusion_strategy": strategy,
                "use_relative_year": use_relative_year,
                "max_sequence_len": max_len,
                "model_name_or_path": model_name,
                "run_id": run_id
            })

    # Save full run results
    df = pd.DataFrame(results)
    df.to_csv(output_tsv, sep="\t", index=False)
    print(f"Saved probing results to {args.output_tsv}")

    # Save summary
    summary = df.groupby([
        "checkpoint", "temporal_fusion_strategy", "use_relative_year",
        "max_sequence_len", "model_name_or_path"
    ])["accuracy"].agg(["mean", "std"]).reset_index()
    summary.rename(columns={"mean": "mean_accuracy", "std": "std_accuracy"}, inplace=True)
    summary.to_csv("summary_results.tsv", sep="\t", index=False)
    print("Saved summarized results to summary_results.tsv")
