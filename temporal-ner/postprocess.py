import os
import pandas as pd
from tqdm import tqdm


def extract_info_from_folder_name(folder_name):
    parts = folder_name.split("_")
    model_type = "_".join(parts[parts.index("run") + 1 :])
    seq_length = parts[parts.index("length") + 1]
    llm_type = "_".join(parts[1 : parts.index("max")])
    run_type = parts[parts.index("run") + 1]
    return llm_type, seq_length, model_type, run_type


def read_tsv_files(folder_path, llm_type, seq_length, model_type, run_type):
    data = {}
    for file_name in os.listdir(folder_path):
        if "test" in file_name and ("coarse" in file_name or "fine" in file_name):
            evaluation_type = "coarse" if "coarse" in file_name else "fine"
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, sep="\t")
            for _, row in df.iterrows():
                if "Evaluation" in row:
                    if row["Evaluation"] in [
                        "NE-COARSE-LIT-micro-fuzzy-TIME-ALL-LED-ALL",
                        "NE-COARSE-LIT-micro-strict-TIME-ALL-LED-ALL",
                        "NE-FINE-LIT-micro-fuzzy-TIME-ALL-LED-ALL",
                        "NE-FINE-LIT-micro-strict-TIME-ALL-LED-ALL",
                    ]:
                        if row["Label"] == "ALL":
                            eval_type = row["Evaluation"].split("-")[4]
                            key = (llm_type, seq_length, model_type, run_type)
                            if key not in data:
                                data[key] = {
                                    "Model Type": model_type,
                                    "Seq Length": seq_length,
                                    "LLM Type": llm_type,
                                    "Run Type": run_type,
                                }
                            prefix = f"{evaluation_type.capitalize()} {eval_type}"
                            data[key][f"{prefix} P"] = row["P"]
                            data[key][f"{prefix} R"] = row["R"]
                            data[key][f"{prefix} F1"] = row["F1"]
    return data


def main():
    experiments_folder = "experiments"
    all_data = []
    list_dir = os.listdir(experiments_folder)
    for folder_name in tqdm(list_dir, total=len(list_dir)):
        folder_path = os.path.join(experiments_folder, folder_name)
        if os.path.isdir(folder_path):
            llm_type, seq_length, model_type, run_type = extract_info_from_folder_name(
                folder_name
            )
            folder_data = read_tsv_files(folder_path, llm_type, seq_length, model_type, run_type)
            print(f"Processing folder: {folder_name} -> {folder_data}")
            all_data.extend(folder_data.values())

    df = pd.DataFrame(all_data)
    df.sort_values(by=["Model Type", "LLM Type", "Seq Length", "Run Type"], inplace=True)
    output_path = os.path.join(experiments_folder, "compiled_results.csv")
    df.to_csv(output_path, index=False, sep="\t")
    print(f"Data saved to {output_path}")


if __name__ == "__main__":
    main()
