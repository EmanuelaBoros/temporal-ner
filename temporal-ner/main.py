from peft import LoraConfig, TaskType, get_peft_model

from models import train, evaluate
import argparse
from models import (
    MultitaskModelForTokenClassification,
    MultitaskTimeModelForTokenClassification,
)  # Update the import to the new model
import wandb
import os
from dataset import NewsDataset, export_entity_statistics
import logging
from transformers import (
    AutoTokenizer,
    AutoConfig,
)
from torch.optim import AdamW
from accelerate import Accelerator, FullyShardedDataParallelPlugin
import json
from utils import write_predictions
import torch
from datetime import datetime
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
from utils import set_seed, SEED, check_for_existing_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-multilingual-cased",
        help="The model to be loaded. It can be a pre-trained model "
             "or a fine-tuned model (folder on the disk).",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="baseline",
        choices=[
            "baseline",
            "extended",
            "date",
            "separated_date",
            "time",
            "time_swiglu",
            "time_separated_date_swiglu",
            "time_extended_separated_date_swiglu",
            "time_extended_date_swiglu",
            "position_extended_date_swiglu",
            "position_extended_separated_date_swiglu",
        ],
        help="The type of model to be used: multitask or extended_multitask.",
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="",
        help="Path to the *csv or *tsv train file.",
    )
    parser.add_argument(
        "--dev_dataset", type=str, default="", help="Path to the *csv or *tsv dev file."
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="",
        help="Path to the *csv or *tsv test file.",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--max_sequence_len", type=int, default=64, help="Maximum text length."
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs. Default to 3 (can be 5 - max 10)",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="The training batch size - can be changed depending on the GPU.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="The training batch size - can be changed depending on the GPU.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="The folder where the experiment details and the predictions should be saved.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="The folder with a checkpoint model to be loaded and continue training or evaluate.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--logging_steps", type=int, default=50, help="Log every X updates steps."
    )

    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--n_warmup_steps",
        type=int,
        default=0,
        help="The warmup steps - the number of steps in on epoch or 0.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="The device on which should the model run - cpu or cuda.",
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )

    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument("--wandb", action="store_true", help="Whether to run wandb.")
    parser.add_argument(
        "--continue_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval or not."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to make experiment reproducible."
    )
    # "film", "add", "concat", "adapter"
    parser.add_argument("--temporal_fusion_strategy", type=str, default="none",
                        help="The strategy to use for temporal fusion.")
    parser.add_argument(
        "--use_relative_year", action="store_true",
        help="Use relative year encoding (e.g., 2025 - year) instead of the exact year."
    )
    args = parser.parse_args()

    set_seed(args.seed)

    logging_suffix = args.model_type + "-swiglu" + '.' + args.temporal_fusion_strategy + '.' + str(
        args.use_relative_year)
    args.model_name_or_path = args.model_name_or_path.lower()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.output_dir = os.path.join(
        args.output_dir,
        "model_{}_max_sequence_length_{}_epochs_{}_run_{}".format(
            args.model_name_or_path.replace("/", "_").replace("-", "_"),
            args.max_sequence_len,
            args.epochs,
            logging_suffix
        ),
    )

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Check for existing files containing "test" and ("coarse" or "fine")
    if check_for_existing_files(
            args.output_dir, ["test", "coarse"]
    ) or check_for_existing_files(args.output_dir, ["test", "fine"]):
        if args.do_train:
            print("Experiment skipped due to existing files.")
            exit()

    # Generate a dynamic name based on current time and other parameters
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.output_dir}_{current_time}"

    # Initialize wandb with a specific run name
    if args.wandb:
        wandb.init(
            project="long-horizon",
            name=run_name,
            config={
                "learning_rate": args.learning_rate,
                "batch_size": args.train_batch_size,
                "num_epochs": args.epochs,
            },
        )

    if "multilingual" not in logging_suffix:
        # we only look for results if we are not in multilingual mode
        for lang in ["fr", "de"]:
            if os.path.exists(
                    os.path.join(args.output_dir, f"all_results_{lang}.json")
            ):
                logging.info(f"Results already exist in {args.output_dir}.")
                exit()

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "logging.log")),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    logging.info(
        "Trained models and results are saved in {}.".format(args.output_dir)
    )

    # Setup CUDA, GPU & distributed training
    if args.device == "cpu":
        device = torch.device("cpu")
        args.n_gpu = 1
    elif args.local_rank == -1 or args.device == "cuda":
        device = torch.device(
            "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    if "bloom" in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, add_prefix_space=True
        )
    elif "llama" in args.model_name_or_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, add_prefix_space=True, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    elif "pleias" in args.model_name_or_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, add_prefix_space=True, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        print(f"----Tokenizer pad token has been set to {tokenizer.pad_token}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    label_map = None
    label_map_path = os.path.join(args.output_dir, "label_map.json")
    if not os.path.exists(label_map_path):
        print(f"Label map does not exist in {label_map_path}. Creating it now.")
        train_dataset = NewsDataset(
            tsv_dataset=args.train_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len,
            args=args,
        )

        label_map = train_dataset.get_label_map()

        dev_dataset = NewsDataset(
            tsv_dataset=args.dev_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len,
            label_map=label_map,
            args=args,
        )

        label_map = dev_dataset.get_label_map()

        test_dataset = NewsDataset(
            tsv_dataset=args.test_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len,
            label_map=label_map,
            args=args,
        )

        label_map = test_dataset.get_label_map()

        json.dump(label_map, open(label_map_path, "w"))
    else:
        print(f"Label map already exists in {label_map_path}.")
        label_map = json.load(open(label_map_path, "r"))
        print(f"Label map loaded: {label_map}.")

        train_dataset = NewsDataset(
            tsv_dataset=args.train_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len,
            label_map=label_map,
            args=args,
        )

        dev_dataset = NewsDataset(
            tsv_dataset=args.dev_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len,
            label_map=label_map,
            args=args,
        )

        test_dataset = NewsDataset(
            tsv_dataset=args.test_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len,
            label_map=label_map,
            args=args,
        )

    num_sequence_labels, num_token_labels_dict, num_years = train_dataset.get_info()

    export_entity_statistics(
        train_dataset,
        split_name="train",
        output_path=os.path.join(args.output_dir, f"entity_statistics_train.tsv"),
        label_map=label_map,
    )
    export_entity_statistics(
        dev_dataset,
        split_name="dev",
        output_path=os.path.join(args.output_dir, f"entity_statistics_dev.tsv"),
        label_map=label_map,
    )
    export_entity_statistics(
        test_dataset,
        split_name="test",
        output_path=os.path.join(args.output_dir, f"entity_statistics_test.tsv"),
        label_map=label_map,
    )
    logging.info(
        "Number of unique token labels for each task: {}.".format(num_token_labels_dict)
    )

    config = AutoConfig.from_pretrained(args.model_name_or_path)

    if any(
            model in args.model_name_or_path.lower()
            for model in ["llama", "mistral", "mixtral", 'pleias']
    ):

        def setup_accelerator():
            fsdp_plugin = FullyShardedDataParallelPlugin(
                state_dict_config=FullStateDictConfig(
                    offload_to_cpu=True, rank0_only=False
                ),
                optim_state_dict_config=FullOptimStateDictConfig(
                    offload_to_cpu=True, rank0_only=False
                ),
            )
            return Accelerator(fsdp_plugin=fsdp_plugin)


        accelerator = setup_accelerator()
        model = MultitaskModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            config=config,
            num_token_labels_dict=num_token_labels_dict,
        ).bfloat16()
        print(model)

        lora_r = 12
        if "mistral" in args.model_name_or_path.lower():
            peft_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "w1",
                    "w2",
                    "w3",
                ],
                bias="none",
                lora_dropout=0,  # Conventional
                task_type=TaskType.TOKEN_CLS,
            )
        else:
            peft_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                bias="none",
                lora_dropout=0,  # Conventional
                task_type=TaskType.TOKEN_CLS,
            )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model = accelerator.prepare_model(model)


    elif args.model_type == "time":
        model = MultitaskTimeModelForTokenClassification(config, num_token_labels_dict,
                                                         temporal_fusion_strategy=args.temporal_fusion_strategy,
                                                         num_years=num_years)

    model = model.to(args.device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    print(model)
    if args.do_train:
        train(
            args=args,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            test_dataset=test_dataset,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            label_map=label_map,
            model_class=model.__class__.__name__,

        )

    elif args.continue_train:
        logger.info(f"Resumed from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]

        train(
            args=args,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            test_dataset=test_dataset,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            label_map=label_map,
            model_class=model.__class__.__name__,
        )
    if args.do_eval:
        config = AutoConfig.from_pretrained(
            args.checkpoint,
            problem_type="single_label_classification",
            local_files_only=True,
        )

        logger.info(f"Resumed from checkpoint: {args.checkpoint}")
        best_output_dir = args.checkpoint
        print(f"Best model saved to {best_output_dir} - loading..")
        print(f"num_token_labels_dict: {num_token_labels_dict}")

        model = MultitaskTimeModelForTokenClassification(config, num_token_labels_dict,
                                                         temporal_fusion_strategy=args.temporal_fusion_strategy,
                                                         num_years=num_years)
        checkpoint = torch.load(os.path.join(args.checkpoint, "pytorch_model.bin"))
        print(f"Loading {checkpoint.keys()}...")
        model.load_state_dict(checkpoint)

        model = model.to(args.device)

        results, words_list, preds_list, report_bin, report_class = evaluate(
            args, model, test_dataset, label_map, tokenizer=tokenizer
        )

        write_predictions(
            args, args.output_dir, test_dataset.get_filename(), words_list, preds_list
        )

        results_testset = {"global": results, "token-level": report_class}

        print("Results on test set:", results_testset)
