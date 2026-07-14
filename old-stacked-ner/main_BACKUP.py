# -*- coding: utf-8 -*-

from functools import partial
from tqdm import tqdm
from models.models import StackedTransformersCRF, BertCRF
from fastNLP import cache_results
from fastNLP import GradientClipCallback, WarmupCallback, CheckPointCallback
from fastNLP import SpanFPreRecMetric
from modules.pipe_main import DataReader
from modules.callbacks import EvaluateCallback
from modules.utils import set_rng_seed
from embeddings import BertEmbedding
import csv
import json
import os
import torch
import argparse
from transformers import AdamW
from predictor import Predictor
from trainer import Trainer
import multiprocessing
import pandas as pd
import subprocess

set_rng_seed(rng_seed=42)


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
parser.add_argument('--pos_embed', type=str, default='sin') #or fix
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

args = parser.parse_args()

output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

pre_trained_model = args.pre_trained_model
pre_trained_model = pre_trained_model.split('/')[-1]

if args.model in ['stacked']:
    output_dir = os.path.join(output_dir,
                              f'{args.dataset}_model-{args.model}_{args.language}_{pre_trained_model}_num_layers-{args.num_layers}_attn_type_{args.attn_type}_n_heads-{args.n_heads}_head_dims-{args.head_dims}_pos_embed_{args.pos_embed}_trans_dropout_{args.trans_dropout}_fc_dropout{args.fc_dropout}_pool_method_{args.pool_method}')
else:
    output_dir = os.path.join(output_dir,
                              f'{args.dataset}_model-{args.model}_{args.language}_{pre_trained_model}')

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print(f'{output_dir} created.')

dataset = args.dataset
n_heads = args.n_heads
head_dims = args.head_dims
num_layers = args.num_layers
attn_type = args.attn_type
trans_dropout = args.trans_dropout
batch_size = args.batch_size
lr = args.lr
pos_embed = args.pos_embed
warmup_steps = args.warmup_steps
after_norm = args.after_norm
fc_dropout = args.fc_dropout
no_cpu = args.no_cpu
normalize_embed = True

encoding_type = args.encoding_type
name = output_dir + '/{}_{}_{}_normalize_{}.pkl'.format(args.model, dataset, encoding_type, normalize_embed)
d_model = n_heads * head_dims
dim_feedforward = int(2 * d_model)
if output_dir:
    dataset_dir = args.dataset_dir
    # change if other extension
    files = [
        os.path.join(
            path,
            f) for path,
        directories,
        files in os.walk(dataset_dir) for f in files if f.endswith(
            "." +
            args.extension) and not os.path.exists(
                os.path.join(
                    path.replace(
                        dataset_dir,
                        output_dir),
                    f))]
#    import pdb
#    pdb.set_trace()
paths = {'test': args.test_dataset,
         'train': args.train_dataset,
         'dev': args.dev_dataset}


# Convert args namespace to dictionary
config = vars(args)

# Build the path to the config file
config_path = os.path.join(output_dir, 'config.json')

# Write the config dictionary to file as JSON
with open(config_path, 'w') as config_file:
    json.dump(config, config_file, indent=4)

@cache_results(name, _refresh=False)
def load_data(paths, load_embed=True):
    data = DataReader(
        encoding_type=encoding_type,
        model_dir_or_name=args.pre_trained_model).process_from_file(paths)

    if load_embed:

        embed = BertEmbedding(
            data.get_vocab('words'),
            model_dir_or_name=args.pre_trained_model,
            pool_method=args.pool_method,
            requires_grad=True,
            layers=args.layers,
            include_cls_sep=False,
            dropout=0.5,
            auto_truncate=True,
            word_dropout=0.01)

        return data, embed, embed
    return data


data_bundle, embed, embed_doc = load_data(paths, load_embed=True)
print(data_bundle.get_dataset('test')[:10])


def predict(path, data_bundle, predictor, predict_on='test', do_eval=False):

    if do_eval:
        paths = {'train': path}
        data_bundle_test = DataReader(
            encoding_type=encoding_type,
            vocabulary=data_bundle.get_vocab('words')).process_from_file(paths)
        dataset_test = data_bundle_test.get_dataset('train')
        predictions = predictor.predict(dataset_test)
        predictions_path = path.replace(dataset_dir, output_dir)
    else:
        print('Predicting on {}:'.format(predict_on))
        dataset_test = data_bundle.get_dataset(predict_on)
        predictions = predictor.predict(dataset_test)
        predictions_path = path

    with open(predictions_path, 'w') as f:
        f.write('TOKEN	NE-COARSE-LIT	NE-COARSE-METO	NE-FINE-LIT	NE-FINE-METO	NE-FINE-COMP	NE-NESTED	NEL-LIT	NEL-METO	MISC\n')
        for i, j, j1, j2, j3, j4, j5 in zip(dataset_test, predictions['pred'], predictions['pred1'], predictions['pred2'],
                                            predictions['pred3'], predictions['pred4'], predictions['pred5']):
            if isinstance(j[0], int):
                f.write(str(i['raw_words'][0]) +
                        '\tO\tO\tO\tO\tO\tO\t_\t_\t_\n')
            else:
                labels = list([data_bundle.get_vocab(
                    'target').idx2word[x] for x in j[0]])
                labels += ['O'] * len(i['raw_words'])
                labels1 = list([data_bundle.get_vocab(
                    'target1').idx2word[x] for x in j1[0]])
                labels1 += ['O'] * len(i['raw_words'])
                labels2 = list([data_bundle.get_vocab(
                    'target2').idx2word[x] for x in j2[0]])
                labels2 += ['O'] * len(i['raw_words'])
                labels3 = list([data_bundle.get_vocab(
                    'target3').idx2word[x] for x in j3[0]])
                labels3 += ['O'] * len(i['raw_words'])
                labels4 = list([data_bundle.get_vocab(
                    'target4').idx2word[x] for x in j4[0]])
                labels4 += ['O'] * len(i['raw_words'])
                labels5 = list([data_bundle.get_vocab(
                    'target5').idx2word[x] for x in j5[0]])
                labels5 += ['O'] * len(i['raw_words'])

                for word, label, label1, label2, label3, label4, label5 in zip(
                        i['raw_words'], labels, labels1, labels2, labels3, labels4, labels5):
                    f.write(
                        str(word) +
                        '\t' +
                        str(label) +
                        '\t' +
                        str(label1) +
                        '\t' +
                        str(label2) +
                        '\t' +
                        str(label3) +
                        '\t' +
                        str(label4) +
                        '\t' +
                        str(label5) +
                        '\t_\t_\t_\n')
            f.write('\n')
    return predictions


def main():
    if args.do_eval:
        torch.multiprocessing.set_start_method('spawn', force=True)

    if args.model == 'bert':

        model = BertCRF(embed,  # , data_bundle.get_vocab('target1')
                        [data_bundle.get_vocab('target'),
                         data_bundle.get_vocab('target1'),
                         data_bundle.get_vocab('target2'),
                         data_bundle.get_vocab('target3'),
                         data_bundle.get_vocab('target4'),
                         data_bundle.get_vocab('target5')],
                        encoding_type='bioes')

    else:
        model = StackedTransformersCRF(
            tag_vocabs=[
                data_bundle.get_vocab('target'),
                data_bundle.get_vocab('target1'),
                data_bundle.get_vocab('target2'),
                data_bundle.get_vocab('target3'),
                data_bundle.get_vocab('target4'),
                data_bundle.get_vocab('target5')],
            embed=embed,
            embed_doc=embed_doc,
            num_layers=num_layers,
            d_model=d_model,
            n_head=n_heads,
            feedforward_dim=dim_feedforward,
            dropout=trans_dropout,
            after_norm=after_norm,
            attn_type=attn_type,
            bi_embed=None,
            fc_dropout=fc_dropout,
            pos_embed=pos_embed,
            scale=attn_type == 'transformer')
        model = torch.nn.DataParallel(model)

    if args.do_eval:
        if os.path.exists(os.path.expanduser(args.saved_model)):
            print("Load checkpoint from {}".format(
                os.path.expanduser(args.saved_model)))
            model = torch.load(args.saved_model)
            model.to('cuda')
            print('model to CUDA')

    from torch import optim
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    # optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99))

    callbacks = []
    clip_callback = GradientClipCallback(clip_type='value', clip_value=5)
    # , batch_size=8, use_cuda=False)
    evaluate_callback = EvaluateCallback(data_bundle.get_dataset('test'))
    checkpoint_callback = CheckPointCallback(
        os.path.join(output_dir, 'model.pth'), delete_when_train_finish=False,
        recovery_fitlog=True)

    if warmup_steps > 0:
        warmup_callback = WarmupCallback(warmup_steps, schedule='linear')
        callbacks.append(warmup_callback)
    callbacks.extend([clip_callback, checkpoint_callback, evaluate_callback])

    print('-'*20)
    num_of_gpus = torch.cuda.device_count()
    print(num_of_gpus)
    # device = list(range(num_of_gpus))
    print('-'*20)


    if not args.do_eval:
        if not os.path.exists(os.path.join(output_dir, 'predictions_test.tsv')):
            trainer = Trainer(data_bundle.get_dataset('train'), model, optimizer,
                              batch_size=batch_size, #sampler=SortedSampler(),
                              num_workers=no_cpu, n_epochs=args.n_epochs,
                              dev_data=data_bundle.get_dataset('dev'),
                              metrics=SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'),
                                                        encoding_type=encoding_type),
                              dev_batch_size=batch_size,
                              callbacks=callbacks,
                              device=args.device,
                              test_use_tqdm=True,
                              use_tqdm=True,
                              print_every=1,
                              # fp16=True,
                              save_path=os.path.join(output_dir, 'best'))

            trainer.train(load_best_model=True)

            predictor = Predictor(model)
            # predict(os.path.join(directory, f'dev_{args.dataset}_{args.languge}_{args.model}_{args.num_layers}_{args.n_head}_{args.head_dims}_{args.pos_embed}_predictions.tsv'),
            predict(os.path.join(output_dir,
                                 f'predictions_dev.tsv'),
                    data_bundle, predictor, 'dev')
            predict(os.path.join(output_dir, f'predictions_test.tsv'),
                    data_bundle, predictor, 'test')
        print('Computing results.')
        if 'stacked' in args.model:
            line = {'dataset': args.dataset,
                    'model': args.model,
                    'language': args.language,
                    'pre_trained_model': pre_trained_model,
                    '_num_layers': args.num_layers,
                    'attn_type': args.attn_type,
                    'n_heads': args.n_heads,
                    'head_dims': args.head_dims,
                    'pos_embed': args.pos_embed,
                    'trans_dropout': args.trans_dropout,
                    'fc_dropout': args.fc_dropout,
                    'pool_method': args.pool_method,
                    'learning_rate': args.lr,
                    }
        else:
            line = {'dataset': args.dataset,
                    'model': args.model,
                    'language': args.language,
                    'pre_trained_model': pre_trained_model,
                    '_num_layers': '-',
                    'attn_type': '-',
                    'n_heads': '-',
                    'head_dims': '-',
                    'pos_embed': '-',
                    'trans_dropout': '-',
                    'fc_dropout': '-',
                    'pool_method': '-',
                    'learning_rate': args.lr,
                    }
        with open(os.path.join('experiments', 'results_models.csv'), 'a') as f_out:
            for task in ['nerc_coarse', 'nerc_fine']:
                # Run the external evaluation script
                eval_cmd = f"python {args.eval_script_path} --ref {args.test_dataset} --pred {os.path.join(output_dir, 'predictions_test.tsv')} --skip-check --outdir {output_dir} --hipe_edition hipe-2022 --log logs --task {task}"
                subprocess.run(eval_cmd, shell=True, check=True)
                # Read the results using pandas
                results_path = os.path.join(output_dir, f"predictions_test_{task}.tsv")
                df = pd.read_csv(results_path, sep="\t")

                # Filter the desired results
                if 'coarse' in task:
                    desired_filters = [
                        ("NE-COARSE-LIT-micro-fuzzy-TIME-ALL-LED-ALL", "ALL"),
                        ("NE-COARSE-LIT-micro-strict-TIME-ALL-LED-ALL", "ALL")
                    ]
                else:
                    desired_filters = [
                        ("NE-FINE-LIT-micro-fuzzy-TIME-ALL-LED-ALL", "ALL"),
                        ("NE-FINE-LIT-micro-strict-TIME-ALL-LED-ALL", "ALL")
                    ]

                for system, label in desired_filters:
                    filtered_row = df[(df['Evaluation'] == system) & (df['Label'] == label)][['Evaluation', 'Label', 'P', 'R', 'F1']]
                    print(filtered_row.to_string(index=False))
                    line[task + '-' + filtered_row['Evaluation'].iloc[0].split('-')[4] + '-P'] = filtered_row['P'].iloc[0]
                    line[task + '-' + filtered_row['Evaluation'].iloc[0].split('-')[4] + '-R'] = filtered_row['P'].iloc[0]
                    line[task + '-' + filtered_row['Evaluation'].iloc[0].split('-')[4] + '-F1'] = filtered_row['P'].iloc[0]

            writer = csv.DictWriter(f_out, fieldnames=line.keys(), delimiter='\t')
            # import pdb;pdb.set_trace()
            print(line)
            writer.writerow(line)

    else:
        print('Predicting')
        # predictions of multiple files
        torch.multiprocessing.freeze_support()
        model.share_memory()
        predictor = Predictor(model)

        if len(files) > multiprocessing.cpu_count():
            with torch.multiprocessing.Pool(processes=no_cpu) as p:
                with tqdm(total=len(files)) as pbar:
                    for i, _ in enumerate(p.imap_unordered(partial(predict,
                                                                   data_bundle=data_bundle,
                                                                   predictor=predictor,
                                                                   predict_on='train',
                                                                   do_eval=args.do_eval), files)):
                        pbar.update()
        else:
            for file in tqdm(files):
                predict(file, data_bundle, predictor, 'train', args.do_eval)


if __name__ == '__main__':
    main()
