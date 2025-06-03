import os
import shutil
import argparse
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
)
import subprocess
from transformers import pipeline
from transformers.pipelines import PIPELINE_REGISTRY
from huggingface_hub import HfApi, Repository
import json
from .configuration_stacked import ImpressoConfig
from .modeling_stacked import ExtendedMultitaskModelForTokenClassification
from .generic_ner import MultitaskTokenClassificationPipeline
from pprint import pprint


def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [
        d
        for d in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, d))
        and d.startswith("checkpoint-")
    ]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]), reverse=True)
    return os.path.join(checkpoint_dir, checkpoints[0])


def get_info(label_map):
    num_token_labels_dict = {task: len(labels) for task, labels in label_map.items()}
    return num_token_labels_dict


def push_model_to_hub(checkpoint_dir, repo_name, script_path):
    checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    config = ImpressoConfig.from_pretrained(checkpoint_path)
    config.pretrained_config = AutoConfig.from_pretrained(config.name_or_path)
    config.label_map = json.load(
        open(os.path.join(checkpoint_dir, "label_map.json"), "r")
    )
    config.save_pretrained("stacked_bert")
    config = ImpressoConfig.from_pretrained("stacked_bert")

    model = ExtendedMultitaskModelForTokenClassification.from_pretrained(
        checkpoint_path, config=config
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    local_repo_path = "./repo"
    repo_url = HfApi().create_repo(repo_id=repo_name, exist_ok=True)
    repo = Repository(local_dir=local_repo_path, clone_from=repo_url)

    try:
        # Try to pull the latest changes from the remote repository using subprocess
        subprocess.run(["git", "pull"], check=True, cwd=local_repo_path)
    except subprocess.CalledProcessError as e:
        # If fast-forward is not possible, reset the local branch to match the remote branch
        subprocess.run(
            ["git", "reset", "--hard", "origin/main"],
            check=True,
            cwd=local_repo_path,
        )

    # Copy all Python files to the local repository directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for filename in os.listdir(current_dir):
        if filename.endswith(".py") or filename.endswith(".json"):
            shutil.copy(
                os.path.join(current_dir, filename),
                os.path.join(local_repo_path, filename),
            )

    ImpressoConfig.register_for_auto_class()
    AutoConfig.register("stacked_bert", ImpressoConfig)
    AutoModelForTokenClassification.register(
        ImpressoConfig, ExtendedMultitaskModelForTokenClassification
    )
    ExtendedMultitaskModelForTokenClassification.register_for_auto_class(
        "AutoModelForTokenClassification"
    )

    model.save_pretrained(local_repo_path)
    tokenizer.save_pretrained(local_repo_path)

    model.config.custom_pipelines = {
        "generic-ner": {
            "impl": "generic_ner.MultitaskTokenClassificationPipeline",
            "pt": "AutoModelForTokenClassification",
        }
    }
    PIPELINE_REGISTRY.register_pipeline(
        "generic-ner",
        pipeline_class=MultitaskTokenClassificationPipeline,
        pt_model=ExtendedMultitaskModelForTokenClassification,
    )
    classifier = pipeline("generic-ner", model=model, tokenizer=tokenizer)

    pprint(
        classifier(
            "1. Le public est averti que Charlotte née Bourgoin, femme-de Joseph Digiez, et Maurice Bourgoin, enfant mineur représenté par le sieur Jaques Charles Gicot son curateur, ont été admis par arrêt du Conseil d'Etat du 5 décembre 1797, à solliciter une renonciation générale et absolue aux biens et aux dettes présentes et futures de Jean-Baptiste Bourgoin leur père."
        )
    )
    # repo.push_to_hub(commit_message="Initial commit of the trained NER model with code")

    # Add, commit and push the changes to the repository
    subprocess.run(["git", "add", "."], check=True, cwd=local_repo_path)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit including model and configuration"],
        check=True,
        cwd=local_repo_path,
    )
    repo.push_to_hub(repo_name)

    # Push the model to the hub (this includes the README template)
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)

    print(f"Model and repo pushed to: {repo_url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push NER model to Hugging Face Hub")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Type of the model (e.g., stacked-bert)",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Language of the model (e.g., multilingual)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing checkpoint folders",
    )
    parser.add_argument(
        "--script_path", type=str, required=True, help="Path to the models.py script"
    )
    args = parser.parse_args()
    repo_name = f"impresso-project/ner-{args.model_type}-{args.language}"
    push_model_to_hub(args.checkpoint_dir, repo_name, args.script_path)
