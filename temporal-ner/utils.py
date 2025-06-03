"""
@ hipe-eval / HIPE-2022-baseline
A few general utilities for transformers_baseline.
"""

import random
import logging
import time
from datetime import timedelta
import json

import numpy as np
import urllib.request
from typing import Set, List, Union, NamedTuple, Dict, Optional
import torch

from dataset import COLUMNS
import os
import subprocess

SEED = 42


def set_seed(seed) -> None:
    # np.random.seed(seed)
    # random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    # os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")


def check_for_existing_files(output_dir, keywords):
    for filename in os.listdir(output_dir):
        if all(keyword in filename for keyword in keywords):
            return True
    return False


def run_hipe_scorer(
        output_folder,
        ref_file,
        pred_file,
        task="nerc_coarse",
):
    """
    Runs the HIPE scorer evaluation script.

    :param output_folder: Folder where the predictions are saved.
    :param model_path: Path to the model directory.
    :param max_seq_len: Maximum sequence length used in the model.
    :param logging_suffix: Suffix for logging purposes.
    :param language: Language of the dataset (e.g., 'fr' for French).
    :param ref_file: Reference file path for the scorer.
    :param pred_file: Prediction file path for the scorer.
    """
    command = [
        "python",
        "../impresso-named-entity-recognition/HIPE-scorer/clef_evaluation.py",
        "--ref",
        ref_file,
        "--pred",
        pred_file,
        "--task",
        task,
        "--outdir",
        output_folder,
        "--hipe_edition",
        "HIPE-2022",
        "--log",
        os.path.join(output_folder, "logs_dev_scorer.txt"),
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"HIPE scorer failed with error:\n{result.stderr}")
    else:
        print(f"HIPE scorer completed successfully:\n{result.stdout}")


def get_custom_logger(
        name: str,
        level: int = logging.INFO,
        fmt: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt: str = "%Y-%m-%d %H:%M",
):
    """Custom logging wraper, called each time a logger is declared in the package."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(fmt, datefmt=datefmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def set_seed(seed):
    """Sets seed for `random`, `np.random` and `torch`."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


logger = get_custom_logger(__name__)


def get_tsv_data(path: Optional[str] = None, url: Optional[str] = None) -> str:
    """Fetches tsv data from a path or an url."""

    assert path or url, """`path` or `url` must be provided"""

    if url:
        response = urllib.request.urlopen(url)  # TODO: dangerous
        return response.read().decode("utf-8")

    elif path:
        with open(path) as f:
            return f.read()


def write_predictions(
        args, output_folder, tsv_dataset, words_list_dict, preds_list_dict
):
    """
    Write predictions to the output folder in a specified format.

    @param output_folder: Folder where the predictions will be saved.
    @param tsv_dataset: Path to the original tsv dataset.
    @param words_list_dict: Dictionary of word lists for each task.
    @param preds_list_dict: Dictionary of prediction lists for each task.
    """
    with open(tsv_dataset, "r", encoding="utf-8", errors="replace") as f:
        tsv_lines = f.readlines()

    output_file = os.path.join(
        output_folder, tsv_dataset.split("/")[-1].replace(".tsv", "_pred.tsv")
    )

    flat_words_list_dict = {
        task: [item for sublist in words_list for item in sublist]
        for task, words_list in words_list_dict.items()
    }
    flat_preds_list_dict = {
        task: [item for sublist in preds_list for item in sublist]
        for task, preds_list in preds_list_dict.items()
    }

    with open(output_file, "w", encoding="utf-8") as f:
        idx = 0
        for idx_tsv_line, tsv_line in enumerate(tsv_lines):
            if idx_tsv_line == 0:
                f.write(tsv_line)
            elif len(tsv_line.split("\t")) != len(COLUMNS):
                f.write(tsv_line)
            elif len(tsv_line.strip()) == 0:
                f.write(tsv_line)
            else:
                try:
                    pred_values = [
                        flat_preds_list_dict[task][idx] for task in flat_preds_list_dict
                    ]
                    word_values = [
                        flat_words_list_dict[task][idx] for task in flat_words_list_dict
                    ]

                    # Assuming the order of columns in COLUMNS is preserved
                    output_line = word_values[0]  # Start with the token
                    for i, _ in enumerate(pred_values):
                        output_line += "\t" + pred_values[i]

                    remaining_columns = tsv_line.split("\t")[
                                        len(flat_preds_list_dict) + 1:
                                        ]
                    output_line += "\t" + "\t".join(remaining_columns)

                    f.write(output_line)
                except Exception as e:
                    import pdb

                    pdb.set_trace()
                    # continue
                idx += 1
                f.flush()

    if "test" in tsv_dataset:
        run_hipe_scorer(output_folder, args.test_dataset, output_file, "nerc_coarse")
        run_hipe_scorer(output_folder, args.test_dataset, output_file, "nerc_fine")
    else:
        run_hipe_scorer(args.output_dir, args.dev_dataset, output_file, "nerc_coarse")
        run_hipe_scorer(args.output_dir, args.dev_dataset, output_file, "nerc_fine")
    return output_file


def write_predictions_to_tsv(
        words: List[List[Union[str, None]]],
        labels: List[List[Union[str, None]]],
        tsv_line_numbers: List[List[Union[int, None]]],
        output_file: str,
        labels_column: str,
        tsv_path: str = None,
        tsv_url: str = None,
):
    """Get the source tsv, replaces its labels with predicted labels and write a new file to `output`.

    `words`, `labels` and `tsv_line_numbers` should be three aligned list, so as in HipeDataset.
    """

    logger.info(f"Writing predictions to {output_file}")

    tsv_lines = [l.split("\t") for l in get_tsv_data(tsv_path, tsv_url).split("\n")]
    import pdb

    pdb.set_trace()
    label_col_number = tsv_lines[0].index(labels_column)
    for i in range(len(words)):
        for j in range(len(words[i])):
            if words[i][j]:
                assert tsv_lines[tsv_line_numbers[i][j]][0] == words[i][j]
                tsv_lines[tsv_line_numbers[i][j]][label_col_number] = labels[i][j]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(["\t".join(l) for l in tsv_lines]))


class Timer:
    """Basic timer"""

    def __init__(self):
        self.start = time.time()
        self.intermediate = time.time()

    def tick(self):
        elapsed_time = time.time() - self.intermediate
        self.intermediate = time.time()
        return str(timedelta(seconds=elapsed_time))

    def stop(self):
        elapsed_time = time.time() - self.start
        return str(timedelta(seconds=elapsed_time))


def parse_json(filename):
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            return json.load(f)
    else:
        logger.info(f"File {filename} does not exist.")


def chunk(list, chunksize):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(list), chunksize):
        yield list[i: i + chunksize]
