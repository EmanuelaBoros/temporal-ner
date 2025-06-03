from utils import set_seed, SEED
from modeling_horizon_bert import train, evaluate
import argparse
from modeling_horizon_bert import (
    LongHorizonBertForTokenClassificationTheSecond,
    LongHorizonBertForTokenClassification,
    LongHorizonBertForTokenClassificationTheThird,
    BertForTokenClassification,
    LongHorizonBertForTokenClassificationTheForth,
    LongHorizonBertForTokenClassificationTemporal,
)
import json
import wandb
import os
from dataset import NewsDataset
import logging
from transformers import (
    AutoTokenizer,
    AutoConfig,
    BloomForTokenClassification,
    AutoModelForTokenClassification,
)
from torch.optim import AdamW
from utils import write_predictions
import torch
from transformers import BertForTokenClassification

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
        "--label_map",
        type=str,
        default="data/label_map.json",
        help="Path to the *json file for the label mapping.",
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
        "--logging_suffix",
        type=str,
        default="",
        help="Suffix to further specify name of the folder where the logging is stored.",
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
    parser.add_argument(
        "--continue_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval or not."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to make experiment reproducible."
    )
    parser.add_argument("--model_type", type=str, default="first", help="or second.")
    parser.add_argument(
        "--do_classif",
        action="store_true",
        help="Whether to run binary classification or not.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    args.model_name_or_path = args.model_name_or_path.lower()
    do_classif = args.do_classif

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.output_dir = os.path.join(
        args.output_dir,
        "model_{}_max_sequence_length_{}_epochs_{}_run_{}_{}".format(
            args.model_name_or_path.replace("/", "_").replace("-", "_"),
            args.max_sequence_len,
            args.epochs,
            args.logging_suffix,
            args.model_type,
        ),
    )
    from datetime import datetime

    # Generate a dynamic name based on current time and other parameters
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.output_dir}_{current_time}"

    # Initialize wandb with a specific run name
    wandb.init(
        project="long-horizon",
        name=run_name,
        config={
            "learning_rate": args.learning_rate,
            "batch_size": args.train_batch_size,
            "num_epochs": args.epochs,
            "model_type": args.model_type,
        },
    )

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if "multilingual" not in args.logging_suffix:
        # we only look for results if we are not in multilingual mode
        for lang in ["fr", "de"]:
            if os.path.exists(
                    os.path.join(args.output_dir, f"all_results_{lang}.json")
            ):
                logging.info(f"Results already exist in {args.output_dir}.")
                exit()

    # logging.basicConfig(level=logging.INFO)
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
        "Trained models and results will be saved in {}.".format(args.output_dir)
    )
    wandb.log({"output_dir": args.output_dir})
    wandb.log({"warning": f"High eval loss: ddfdf"})

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
    elif "phi" in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, add_prefix_space=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # if label map was not specified, generate it from data and save it in
    # output folder
    if os.path.exists(args.label_map) is False:
        train_dataset = NewsDataset(
            tsv_dataset=args.train_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len,
        )

        label_map = train_dataset.get_label_map()

        # dataset, tokenizer, max_len, test = False, label_map = None
        dev_dataset = NewsDataset(
            tsv_dataset=args.dev_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len,
            label_map=label_map,
        )

        label_map = dev_dataset.get_label_map()

        test_dataset = NewsDataset(
            tsv_dataset=args.test_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len,
            label_map=label_map,
        )

        label_map = test_dataset.get_label_map()

        json.dump(label_map, open(os.path.join(args.output_dir, "label_map.json"), "w"))
    # if specified, load the label map and use it for the data
    else:
        label_map = json.load(open(args.label_map, "r"))

        train_dataset = NewsDataset(
            tsv_dataset=args.train_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len,
            label_map=label_map,
        )

        dev_dataset = NewsDataset(
            tsv_dataset=args.dev_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len,
            label_map=label_map,
        )

        test_dataset = NewsDataset(
            tsv_dataset=args.test_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len,
            label_map=label_map,
        )

    num_sequence_labels, num_token_labels = test_dataset.get_info()

    logging.info(
        "Number of unique token labels {}, number of unique sequence labels {}.".format(
            num_token_labels, num_sequence_labels
        )
    )

    config = AutoConfig.from_pretrained(
        args.model_name_or_path, problem_type="single_label_classification", trust_remote_code=True
    )

    if args.model_type == "first":
        model = LongHorizonBertForTokenClassification.from_pretrained(
            args.model_name_or_path,
            num_labels=num_token_labels,
        )
    elif args.model_type == "second":
        model = LongHorizonBertForTokenClassificationTheSecond.from_pretrained(
            args.model_name_or_path,
            num_labels=num_token_labels,
        )
    elif args.model_type == "baseline":
        model = BertForTokenClassification.from_pretrained(
            args.model_name_or_path,
            num_labels=num_token_labels,
        )
    elif args.model_type == "third":
        model = LongHorizonBertForTokenClassificationTheThird.from_pretrained(
            args.model_name_or_path,
            num_labels=num_token_labels,
        )
    elif args.model_type == "forth":
        model = LongHorizonBertForTokenClassificationTheForth.from_pretrained(
            args.model_name_or_path,
            num_labels=num_token_labels,
        )
    elif args.model_type == "temporal":
        model = LongHorizonBertForTokenClassificationTemporal.from_pretrained(
            args.model_name_or_path,
            num_labels=num_token_labels,
        )
    elif args.model_type == "bloom":
        from peft import get_peft_model, LoraConfig, TaskType

        lora_r = 12
        model = BloomForTokenClassification.from_pretrained(
            args.model_name_or_path,
            num_labels=num_token_labels,
        )  # .bfloat16()
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    elif args.model_type == "auto":
        # from peft import get_peft_model, LoraConfig, TaskType
        if "gemma" in args.model_name_or_path:
            model = AutoModelForTokenClassification.from_pretrained(
                args.model_name_or_path,
                num_labels=num_token_labels,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        else:
            model = AutoModelForTokenClassification.from_pretrained(
                args.model_name_or_path,
                num_labels=num_token_labels,
            )
        # peft_config = LoraConfig(
        #     task_type=TaskType.TOKEN_CLS,
        #     inference_mode=False,
        #     r=8,
        #     lora_alpha=32,
        #     lora_dropout=0.1,
        # )
        # peft_config = LoraConfig(
        #     task_type=TaskType.TOKEN_CLS,
        #     inference_mode=False,
        #     # target_modules=["up_proj"],
        #     r=12,
        #     lora_alpha=32,
        #     lora_dropout=0.1,
        # )
        # model = get_peft_model(model, peft_config)
        # model.print_trainable_parameters()
    model = model.to(args.device)

    # model = torch.nn.DataParallel(model, device_ids=[0, 1])

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
        )
    else:
        logger.info(f"Resumed from checkpoint: {args.checkpoint}")
        # checkpoint = torch.load(args.checkpoint)
        # model.load_state_dict(checkpoint['model_state_dict'])
        config = AutoConfig.from_pretrained(
            args.checkpoint,
            problem_type="single_label_classification",
            local_files_only=True,
        )

        if args.model_type == "first":
            model = LongHorizonBertForTokenClassification.from_pretrained(
                args.checkpoint,
                num_labels=num_token_labels,
            )
        elif args.model_type == "second":
            model = LongHorizonBertForTokenClassificationTheSecond.from_pretrained(
                args.checkpoint,
                num_labels=num_token_labels,
            )
        elif args.model_type == "baseline":
            model = BertForTokenClassification.from_pretrained(
                args.checkpoint,
                num_labels=num_token_labels,
            )
        model = model.to(args.device)

        tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint, local_files_only=True
        )

        # dev data
        results, words_list, preds_list, _, report_class = evaluate(
            args, model, dev_dataset, label_map, tokenizer=tokenizer
        )

        write_predictions(
            args.output_dir, dev_dataset.get_filename(), words_list, preds_list
        )

        results_devset = dict()
        results_devset["global"] = results
        results_devset["token-level"] = report_class

        # test data
        results, words_list, preds_list, _, report_class = evaluate(
            args, model, test_dataset, label_map, tokenizer=tokenizer
        )

        write_predictions(
            args.output_dir, test_dataset.get_filename(), words_list, preds_list
        )

        results_testset = dict()
        results_testset["global"] = results
        results_testset["token-level"] = report_class

        # results to json
        all_results = {"dev": results_devset, "test": results_testset}
        if "-de" in test_dataset.get_filename():
            with open(os.path.join(args.output_dir, "all_results_de.json"), "w") as f:
                json.dump(all_results, f)
        elif "-fr" in test_dataset.get_filename():
            with open(os.path.join(args.output_dir, "all_results_fr.json"), "w") as f:
                json.dump(all_results, f)
        else:
            logger.info(
                f"Was not able to deduct language from filename of testset, thus no metrics were saved."
            )
