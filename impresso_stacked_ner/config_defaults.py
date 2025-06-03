import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='hipe2020')
    parser.add_argument('--model', type=str, default='stacked',
                        choices=['bert', 'stacked'])
    parser.add_argument('--language', type=str, default='english')
    parser.add_argument('--n_heads', type=int, default=12)
    parser.add_argument('--head_dims', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--attn_type', type=str, default='adatrans')
    parser.add_argument('--trans_dropout', type=float, default=0.45)
    parser.add_argument('--pool_method', type=str, default='last')

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--after_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--warmup_steps', type=float, default=0.01)
    parser.add_argument('--fc_dropout', type=float, default=0.4)
    parser.add_argument('--pos_embed', type=str, default='sin')  # or fix
    parser.add_argument('--encoding_type', type=str, default='bioes')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--layers', type=str, default='-1')
    parser.add_argument('--no_cpu', type=int, default=10)
    parser.add_argument("--lower",
                        action='store_true',
                        help="Whether to lowercase the input.")
    # for elaborate predictions of multiple files
    parser.add_argument('--dataset_dir', type=str)  # input directory files
    parser.add_argument('--output_dir', type=str)  # output predictions directory
    parser.add_argument('--extension', type=str, default='txt')
    # for elaborate predictions of multiple files

    parser.add_argument('--train_dataset', type=str)
    parser.add_argument('--test_dataset', type=str)
    parser.add_argument('--dev_dataset', type=str)

    parser.add_argument('--pre_trained_model', type=str)

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--continue_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval or not.")
    parser.add_argument("--eval_script_path", default="../HIPE-scorer/clef_evaluation.py")
    # in case of do_eval, load model from saved dir/best
    parser.add_argument('--saved_model', type=str)

    return parser.parse_args()
