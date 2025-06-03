import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import AutoConfig
import numpy as np
import argparse
import importlib
import json

ne_tasks = ["NE-COARSE-LIT"]


def tsne_year_embeddings(experiments_root, output_dir, model_class):
    os.makedirs(output_dir, exist_ok=True)

    def load_year_embeddings(model_dir, num_token_labels_dict):
        config = AutoConfig.from_pretrained(model_dir)
        model = model_class.from_pretrained(model_dir, config=config, num_token_labels_dict=num_token_labels_dict)
        if hasattr(model, "year_embedding"):
            return model.year_embedding.weight.detach().cpu().numpy()
        return None

    for subdir in os.listdir(experiments_root):
        model_path = os.path.join(experiments_root, subdir, "best_checkpoint")
        if not os.path.isdir(model_path):
            continue

        label_map_path = os.path.join(experiments_root, subdir, "label_map.json")
        print(f"Label map already exists in {label_map_path}.")
        label_map = json.load(open(label_map_path, "r"))
        print(f"Label map loaded: {label_map}.")

        num_token_labels_dict = {
            task: len(label_map[task]) for task in ne_tasks
        }

        weights = load_year_embeddings(model_path, num_token_labels_dict)
        if weights is None:
            continue

        # Check for NaNs
        if np.isnan(weights).any():
            print(f"Warning: NaNs found in {subdir} year embeddings. Replacing with zeros.")
            weights = np.nan_to_num(weights)  # Replace NaNs with 0
        if weights is None:
            continue

        year_labels = np.arange(1700, 1700 + weights.shape[0])
        tsne = TSNE(n_components=2, perplexity=30)
        proj = tsne.fit_transform(weights)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(proj[:, 0], proj[:, 1], c=year_labels, cmap="viridis", s=15)
        plt.colorbar(scatter, label="Year")
        plt.title(f"t-SNE of year embeddings\n{subdir}")
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"{subdir}_tsne.png")
        plt.savefig(out_path)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_class", type=str, required=True,
                        help="Fully qualified class name, e.g. mymodule.MyModel")
    parser.add_argument("--experiments_root", type=str, default="experiments")
    parser.add_argument("--output_dir", type=str, default="images")
    args = parser.parse_args()

    module_name, class_name = args.model_class.rsplit(".", 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)

    tsne_year_embeddings(args.experiments_root, args.output_dir, model_class)
