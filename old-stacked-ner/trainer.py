__all__ = [
    "Trainer"
]

import os
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn

try:
    from tqdm.auto import tqdm
except BaseException:
    from fastNLP.utils import _pseudo_tqdm as tqdm
import warnings
from pkg_resources import parse_version
from itertools import chain
from fastNLP.core.batch import DataSetIter, BatchIter
from fastNLP.core.callback import CallbackManager, CallbackException, Callback
from fastNLP.core.dataset import DataSet
from fastNLP.core.losses import _prepare_losser
from fastNLP.core.metrics import _prepare_metrics
from fastNLP.core.optimizer import Optimizer
from fastNLP.core.sampler import Sampler, RandomSampler
from fastNLP.core.tester import Tester
from fastNLP.core.utils import (
    _CheckError,
    _build_args,
    _check_forward_error,
    _check_loss_evaluate,
    _move_dict_value_to_device,
    _get_func_signature,
    _get_model_device,
    _move_model_to_device,
    _build_fp16_env,
    _can_use_fp16)
from fastNLP.core._parallel_utils import _model_contains_inner_module
from fastNLP.core._logger import logger


class ConstTokenNumSampler(Sampler):

    def __init__(self, seq_len_field_name, max_token=4096, max_sentence=-1, need_be_multiple_of=1, num_bucket=-1):

        assert (max_sentence!=-1 and max_sentence>=need_be_multiple_of) or max_sentence<1
        self.seq_len_field_name = seq_len_field_name
        self.num_bucket = num_bucket
        self.max_token = max_token
        self._max_sentence = max_sentence
        self.need_be_multiple_of = need_be_multiple_of

    def __call__(self, data_set):
        assert len(data_set)>self.num_bucket, "The number of samples should be larger than buckets."
        seq_len = data_set.get_field(self.seq_len_field_name)
        self.seq_len = seq_len
        seq_len_indice = [(length, i) for i, length in enumerate(seq_len)]
        seq_len_indice.sort(key=lambda x: x[0])
        indice_in_buckets = []
        if self.num_bucket>0:
            sample_per_bucket = len(seq_len_indice)//self.num_bucket
            i = 0
            while len(indice_in_buckets)<len(seq_len_indice):
                indice_in_buckets.append(seq_len_indice[i*sample_per_bucket:(i+1)*sample_per_bucket])
                i += 1
        else:
            indice_in_buckets = [seq_len_indice]
        self.indice_in_buckets = indice_in_buckets
        self.get_new_order()

    @property
    def max_sentence(self):
        if self._max_sentence<1:
            return 100000000
        return self._max_sentence

    @max_sentence.setter
    def max_sentence(self, max_sentence):
        self._max_sentence = max_sentence

    def get_new_order(self):
        np.random.shuffle(self.indice_in_buckets)
        for bucket in self.indice_in_buckets:
            np.random.shuffle(bucket)
        indices = list(chain(*self.indice_in_buckets))
        batches = []
        cur_max_len = 0
        batch = []
        for length, i in indices:
            max_len = max(length, cur_max_len)
            if max_len*(len(batch)+1)>self.max_token or len(batch)>=self.max_sentence:
                left_sample = len(batch) % self.need_be_multiple_of
                add_samples = batch.copy()
                cur_max_len =length
                if left_sample!=0:
                    add_samples = add_samples[:-left_sample]
                    batch = batch[-left_sample:]
                    cur_max_len = max(cur_max_len, max(batch))
                else:
                    batch = []
                if len(add_samples)==0:
                    raise RuntimeError(f"The sample `{i}` is too long to make a batch with {self.need_be_multiple_of} samples.")
                batches.append(add_samples)
            else:
                cur_max_len = max_len
            batch.append(i)
        if batch:
            left_sample = len(batch) % self.need_be_multiple_of
            add_samples = batch.copy()
            if left_sample != 0:
                add_samples = add_samples[:-left_sample].copy()
            if add_samples:
                batches.append(add_samples)
        np.random.shuffle(batches)
        self.batches = batches

    def __iter__(self):
        for batch in self.batches:
            yield batch
        self.get_new_order()

    def __len__(self):
        return len(self.batches)

class Trainer(object):
    r"""
         Trainer is used in fastNLP to organize the training process of a single task, which can avoid users to write repeatedly in different training tasks.
             (1) epoch cycle;
             (2) Divide the data into different Batch;
             (3) Pad the Batch;
             (4) Verify the verification set at the end of each epoch or after a certain step;
             (5) Save models that achieve better verification performance, etc.

         For a detailed introduction, see :mod:`fastNLP.core.trainer`
         """

    def __init__(self, train_data, model, optimizer=None, loss=None,
                 batch_size=32, sampler=None, drop_last=False, update_every=1,
                 num_workers=0, n_epochs=10, print_every=5,
                 dev_data=None, metrics=None, metric_key=None,
                 validate_every=-1, save_path=None, use_tqdm=True, device=None,
                 callbacks=None, check_code_level=0, fp16=False, **kwargs):
        r"""
                 :param train_data: training set, :class:`~fastNLP.DataSet` type or a subclass of :class:`~fastNLP.BatchIter`
                 :param nn.modules model: model to be trained
                 :param optimizer: `torch.optim.Optimizer` optimizer. If it is None, the Trainer uses the default Adam(model.parameters(), lr=4e-3) optimizer
                 :param int batch_size: batch size during training and verification.
                 :param loss: The :class:`~fastNLP.core.losses.LossBase` object used. When None, the default is :class:`~fastNLP.LossInForward`
                 :param sampler: The order in which Batch data is generated, :class:`~fastNLP.Sampler` type. If None, defaults to :class:`~fastNLP.RandomSampler`
                 :param drop_last: If the last batch does not have exactly as much data as batch_size, drop the last batch.
                 :param num_workers: int, how many threads are used for data pad processing.
                 :param update_every: int, how many steps to update the gradient. Used for scenarios where you want to accumulate gradients. For example, a batch_size of 128 is required, but it is directly set to 128.
                     It will cause insufficient memory. This can be achieved by setting batch_size=32, update_every=4. When optimizer is None, this parameter has no effect.
                 :param int n_epochs: How many optimization iterations are needed.
                 :param int print_every: How many times backpropagation updates the loss displayed by tqdm; if use_tqdm=False, how many times backpropagation prints loss.
                 :param dev_data: DataSet used for verification, :class:`~fastNLP.DataSet` type.
                 :param metrics: Validation evaluation function. You can use just one :class:`Metric<fastNLP.core.metrics.MetricBase>` ,
                     You can also use multiple :class:`Metric<fastNLP.core.metrics.MetricBase>` , passed in through a list.
                     If better verification results are obtained during verification (if there are multiple metrics, the first metric in the list shall prevail), and save_path is not None,
                     Then save the current model. For details on Metric types, see :mod:`metrics module <fastNLP.core.metrics>` . Only valid when dev_data is passed in.
                 :param str,None metric_key: :class:`Metric<fastNLP.core.metrics.MetricBase>` Sometimes there are multiple indicators,
                     For example: class:`~fastNLP.core.metrics.SpanFPreRecMetric` contains 'f', 'pre', 'rec'. At this time it is necessary
                     To specify which indicator should be used. In addition, there are some indicators that the smaller the effect, the better, such as the perplexity of the language model. In this case, add a '-' in front of the key to indicate
                     When verifying explicitly, the smaller the value, the better (for example: "-ppl"). Only valid when dev_data is passed in.
                 :param int validate_every: How many steps are verified once on the validation set; if it is -1, the verification is completed once every epoch. Only valid when dev_data is passed in.
                 :param str,None save_path: Save the model path. If the path does not exist, the folder will be automatically created. If None, the model is not saved. If dev_data is None, save
                     The last iteration of the model. When saving, not only the parameters are saved, but also the model structure is saved. Even using DataParallel, only the model is saved here.
                 :param bool use_tqdm: Whether to use tqdm to display training progress; if False, loss will be printed in the terminal.
                 :param str,int,torch.device,list(int) device: Which device to load the model to. The default is None, that is, the Trainer is not suitable for the model
                     The calculation location is managed. The following inputs are supported:

                     1. str: ['cpu', 'cuda', 'cuda:0', 'cuda:1', ...] in order 'cpu', the first visible GPU, the first visible GPU middle,
                     Visible second GPU;

                     2. torch.device: Load the model onto torch.device.

                     3. int: The gpu whose device_id is this value will be used for training.

                     4. list(int): If there is more than 1 device, torch.nn.DataParallel will be used to wrap the model and the passed in device will be used.

                     5. None. If it is None, no processing will be performed on the model. If the incoming model is torch.nn.DataParallel, the value must be None.

                     Known possible problems: Adagrad optimizer may not be able to use this parameter normally, please manage the model position manually.

                 :param list(callbacks) callbacks: callback function used to adjust during the training process. For example, early stop, negative sampling, etc. can
                     Implemented through callback mechanism. For available callbacks, see :mod:`callback module <fastNLP.core.callback>`
                 :param int check_code_level: Model checking level. -1: No checking; 0: Stop only when an error occurs; 1: If a field is not used,
                     Report warning information; 2: If any field is not used, an error is reported. The principle of checking is to run the code by using a small batch (default 2 samples), but
                     In theory, this process will not modify any parameters, but will only check whether it can run. But if (1) there is a situation in the model where batch_size is written as a fixed value;
                     (2) If there are cumulative forward calculation times in the model, one more calculation may be performed. In the above situation, it is recommended to set check_code_level to -1.
                 :param bool fp16: Whether to use fp16 for training.
                 :param kwargs: supports configuring optional parameters
                     bool test_use_tqdm: whether to enable tqdm when verifying on dev
                     Sampler test_sampler: sampler used when evaluating
                     bool test_use_fp16: Whether to use fp16 test when evalute, the default value is the same as fp16.
                     bool set_grad_to_none: Whether to set gradient to None instead of zero during zero_grad
                     GradScaler grad_scaler: Only valid when fp16 is True. If you do not use the initialization parameters of torch.cuda.amp.GradScaler, you can pass in an initialized
                         grad_scaler.
                     bool pin_memory: Whether to use pin memory for the generated tensor, which may speed up the data speed.
                 """
        super(Trainer, self).__init__()
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"The type of model must be torch.nn.Module, got {type(model)}.")

        # check metrics and dev_data
        if (not metrics) and dev_data is not None:
            raise ValueError("No metric for dev_data evaluation.")
        if metrics and (dev_data is None):
            raise ValueError(
                "No dev_data for evaluations, pass dev_data or set metrics to None. ")

        # check update every
        assert update_every >= 1, "update_every must be no less than 1."
        self.update_every = int(update_every)

        # check save_path
        if not (save_path is None or isinstance(save_path, str)):
            raise ValueError("save_path can only be None or `str`.")
        # prepare evaluate
        metrics = _prepare_metrics(metrics)

        # parse metric_key
        # increase_better is True. It means the exp result gets better if the indicator increases.
        # It is true by default.
        self.increase_better = True
        if metric_key is not None:
            self.increase_better = False if metric_key[0] == "-" else True
            self.metric_key = metric_key[1:] if metric_key[0] == "+" or metric_key[0] == "-" else metric_key
        else:
            self.metric_key = None
        # prepare loss
        losser = _prepare_losser(loss)

        if isinstance(train_data, BatchIter):
            if sampler is not None:
                warnings.warn(
                    "sampler is ignored when train_data is a BatchIter.")
            if num_workers > 0:
                warnings.warn(
                    "num_workers is ignored when train_data is BatchIter.")
            if drop_last:
                warnings.warn(
                    "drop_last is ignored when train_data is BatchIter.")
        # concerning issue from https://github.com/pytorch/pytorch/issues/57273
        self.pin_memory = kwargs.get(
            'pin_memory', False if parse_version(
                torch.__version__) == parse_version('1.9') else True)
        if isinstance(model, nn.parallel.DistributedDataParallel):
            if device is not None:
                warnings.warn(
                    "device is ignored when model is nn.parallel.DistributedDataParallel.")
                device = None
            if sampler is None:
                sampler = torch.utils.data.DistributedSampler(train_data)
            elif not isinstance(sampler, torch.utils.data.DistributedSampler):
                raise TypeError(
                    "When using nn.parallel.DistributedDataParallel, "
                    "sampler must be None or torch.utils.data.DistributedSampler.")
            if save_path:
                raise RuntimeError(
                    "Saving model in Distributed situation is not allowed right now.")
        else:
            # sampler check
            # if sampler is not None and not isinstance(
            #         sampler, (Sampler, torch.utils.data.Sampler)):
            #     raise ValueError(
            #         f"The type of sampler should be fastNLP.BaseSampler or pytorch's Sampler, got {type(sampler)}")
            if sampler is None:
                sampler = RandomSampler()
            elif hasattr(sampler, 'set_batch_size'):
                sampler.set_batch_size(batch_size)
            if isinstance(sampler, ConstTokenNumSampler):
                assert isinstance(train_data,
                                  DataSet), f"When sampler is `ConstTokenNumSampler`, the train_data must" \
                                            f" be `DataSet`."
                sampler(train_data)
                train_data = DataSetIter(
                    train_data,
                    batch_size=1,
                    sampler=None,
                    as_numpy=False,
                    num_workers=num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=drop_last,
                    timeout=0,
                    worker_init_fn=None,
                    batch_sampler=sampler)

        if isinstance(train_data, DataSet):
            self.data_iterator = DataSetIter(
                dataset=train_data,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                drop_last=drop_last,
                pin_memory=self.pin_memory)
        elif isinstance(train_data, BatchIter):
            self.data_iterator = train_data
            train_data = train_data.dataset
            check_code_level = -1
        else:
            raise TypeError(
                "train_data type {} not support".format(
                    type(train_data)))

        model.train()
        self.model = _move_model_to_device(model, device=device)
        if _model_contains_inner_module(self.model):
            self._forward_func = self.model.module.forward
        else:
            self._forward_func = self.model.forward

        self.fp16 = fp16
        self.verbose = kwargs.get('verbose', 0)

        self.auto_cast, _grad_scaler = _build_fp16_env(dummy=not fp16)
        self.grad_scaler = _grad_scaler()
        if self.fp16:
            _can_use_fp16(device=device, model=model, func=self._forward_func)
            grad_scaler = kwargs.get('grad_scaler', None)
            if grad_scaler is not None:
                self.grad_scaler = grad_scaler
            else:
                self.grad_scaler = _grad_scaler()
        self.test_use_fp16 = kwargs.get('test_use_fp16', fp16)
        self.set_grad_to_none = kwargs.get('set_grad_to_none', True)

        if check_code_level > -1:
            dev_dataset = dev_data
            if isinstance(dev_data, BatchIter):
                dev_dataset = None
                warnings.warn(
                    "dev_data is of BatchIter type, ignore validation checking.")
            check_batch_size = min(batch_size, DEFAULT_CHECK_BATCH_SIZE)
            if isinstance(self.model, nn.DataParallel):
                _num_devices = len(self.model.device_ids)
                if batch_size // _num_devices > 1:
                    check_batch_size = max(
                        len(self.model.device_ids) * 2, check_batch_size)
                else:
                    check_batch_size = max(
                        len(self.model.device_ids), check_batch_size)
            _check_code(
                dataset=train_data,
                model=self.model,
                losser=losser,
                forward_func=self._forward_func,
                metrics=metrics,
                dev_data=dev_dataset,
                metric_key=self.metric_key,
                check_level=check_code_level,
                batch_size=check_batch_size)

        self.train_data = train_data
        self.dev_data = dev_data  # If None, No validation.
        self.losser = losser
        self.metrics = metrics
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.save_path = save_path
        self.print_every = int(print_every)
        self.validate_every = int(
            validate_every) if validate_every != 0 else -1
        self.best_metric_indicator = None
        self.best_dev_epoch = None
        self.best_dev_step = None
        self.best_dev_perf = None
        self.n_steps = len(self.data_iterator) * self.n_epochs

        if isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        elif isinstance(optimizer, Optimizer):
            self.optimizer = optimizer.construct_from_pytorch(
                self.model.parameters())
        elif optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=4e-3)
        else:
            if not (hasattr(optimizer, 'step') and callable(optimizer.step)):
                raise TypeError(
                    "optimizer must have a callable step() function.")
            else:
                self.optimizer = optimizer

        self.logger = logger

        self.use_tqdm = use_tqdm
        self.test_use_tqdm = kwargs.get('test_use_tqdm', self.use_tqdm)
        self.pbar = None
        self.print_every = abs(self.print_every)
        self.kwargs = kwargs
        if self.dev_data is not None:
            self.tester = Tester(
                model=self.model,
                data=self.dev_data,
                metrics=self.metrics,
                batch_size=kwargs.get(
                    "dev_batch_size",
                    self.batch_size),
                device=None,
                verbose=0,
                use_tqdm=self.test_use_tqdm,
                sampler=kwargs.get(
                    'test_sampler',
                    None),
                fp16=self.test_use_fp16,
                num_workers=num_workers,
                pin_memory=self.pin_memory)

        self.start_time = None  # start timestamp

        if isinstance(callbacks, Callback):
            callbacks = [callbacks]

        self.callback_manager = CallbackManager(env={"trainer": self},
                                                callbacks=callbacks)

    def train(self, load_best_model=True, on_exception='auto', **kwargs):
        r"""
                 Use this function to start the Trainer training.

                 :param bool load_best_model: This parameter is only valid if dev_data is provided during initialization. If True, the trainer will reload the dev performance before returning.
                         best model parameters.
                 :param str on_exception: Whether to continue throwing exceptions after encountering an exception during the training process and being handled by on_exception() of :py:class:Callback.
                         Supports 'ignore', 'raise', 'auto': 'ignore' will catch the exception, and the code written after Trainer.train() will continue to run; 'raise' will throw the exception;
                         'auto' will ignore the following two Exceptions: CallbackException and KeyboardInterrupt, and raise other exceptions.
                :param kwargs:
                         int verbose: When it is 1, when an exception occurs, the index of the data in the batch in the dataset when the exception occurs will be printed.
                 :return dict: Returns a dictionary type data,
                         Contains the following content::

                             seconds: float, represents the training time
                             The following three contents will only exist if dev_data is provided.
                             best_eval: Dict of Dict, represents the result of evaluation. The key of the first layer is the name of the metric.
                                         The key of the second layer is a specific Metric
                             best_epoch: int, the best value obtained in epoch
                             best_step: int, the best value obtained in the step (batch) update

                 """
        results = {}
        verbose = kwargs.get('verbose', 0)
        if self.n_epochs <= 0:
            self.logger.info(
                f"training epoch is {self.n_epochs}, nothing was done.")
            results['seconds'] = 0.
            return results
        try:
            self._model_device = _get_model_device(self.model)
            self._mode(self.model, is_test=False)
            self._load_best_model = load_best_model
            # 加上millsecond，防止两个太接近的保存
            self.start_time = str(
                datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f'))
            start_time = time.time()
            self.logger.info("training epochs started " + self.start_time)
            self.step = 0
            self.epoch = 1
            try:
                self.callback_manager.on_train_begin()
                self._train()
                self.callback_manager.on_train_end()

            except BaseException as e:
                self.callback_manager.on_exception(e)
                if verbose > 0:
                    self.logger.info(
                        f"The data indices for current batch are: {self.data_iterator.cur_batch_indices}.")
                if on_exception == 'auto':
                    if not isinstance(
                            e, (CallbackException, KeyboardInterrupt)):
                        raise e
                elif on_exception == 'raise':
                    raise e

            if self.dev_data is not None and self.best_dev_perf is not None and load_best_model:
                model_name = "best_" + \
                    "_".join([self.model.__class__.__name__, self.metric_key, self.start_time])
                load_succeed = self._load_model(self.model, model_name)
                if load_succeed:
                    self.logger.info("Reloaded the best model.")
                else:
                    self.logger.info("Fail to reload best model.")

            if self.dev_data is None and self.save_path is not None:
                model_name = "_".join(
                    [self.model.__class__.__name__, self.start_time])
                self._save_model(self.model, model_name)

        finally:
            if self.dev_data is not None and self.best_dev_perf is not None:
                self.logger.info(
                    "\nIn Epoch:{}/Step:{}, got best dev performance:".format(
                        self.best_dev_epoch, self.best_dev_step))
                self.logger.info(
                    self.tester._format_eval_results(
                        self.best_dev_perf))
                results['best_eval'] = self.best_dev_perf
                results['best_epoch'] = self.best_dev_epoch
                results['best_step'] = self.best_dev_step

        results['seconds'] = round(time.time() - start_time, 2)

        return results

    def _train(self):
        if not self.use_tqdm:
            from .utils import _pseudo_tqdm as inner_tqdm
        else:
            inner_tqdm = tqdm
        start = time.time()

        # pbar = tqdm(colour="blue", desc=f"Training Epoch: {self.n_steps + 1}", total=total_length, dynamic_ncols=True)

        with inner_tqdm(colour="blue", total=self.n_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True,
                        initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0
            self.batch_per_epoch = self.data_iterator.num_batches
            for epoch in range(self.epoch, self.n_epochs + 1):
                self.epoch = epoch
                pbar.set_description_str(
                    desc="Epoch {}/{}".format(epoch, self.n_epochs))
                # early stopping
                self.callback_manager.on_epoch_begin()
                for batch_x, batch_y in self.data_iterator:
                    self.step += 1

                    _move_dict_value_to_device(
                        batch_x, batch_y, device=self._model_device)
                    indices = self.data_iterator.get_batch_indices()
                    # negative sampling; replace unknown; re-weight batch_y
                    self.callback_manager.on_batch_begin(
                        batch_x, batch_y, indices)
                    prediction = self._data_forward(self.model, batch_x)

                    # edit prediction
                    self.callback_manager.on_loss_begin(batch_y, prediction)
                    with self.auto_cast():
                        loss = self._compute_loss(prediction, batch_y).mean()
                    loss = loss / self.update_every
                    avg_loss += loss.item()

                    # Is loss NaN or inf? requires_grad = False
                    self.callback_manager.on_backward_begin(loss)
                    self._grad_backward(loss)
                    self.callback_manager.on_backward_end()

                    self._update()
                    self.callback_manager.on_step_end()

                    if self.step % self.print_every == 0:
                        avg_loss = float(avg_loss) / self.print_every
                        if self.use_tqdm:
                            print_output = "loss:{:<6.5f}".format(avg_loss)
                            pbar.update(self.print_every)
                        else:
                            end = time.time()
                            diff = timedelta(seconds=round(end - start))
                            print_output = "[epoch: {:>3} step: {:>4}] train loss: {:>4.6} time: {}".format(
                                epoch, self.step, avg_loss, diff)
                        pbar.set_postfix_str(print_output)
                        avg_loss = 0
                    self.callback_manager.on_batch_end()

                    if (self.validate_every > 0 and self.step %
                            self.validate_every == 0) and self.dev_data is not None:
                        eval_res = self._do_validation(
                            epoch=epoch, step=self.step)
                        eval_str = "Evaluation on dev at Epoch {}/{}. Step:{}/{}: ".format(
                            epoch, self.n_epochs, self.step, self.n_steps)
                        # pbar.write(eval_str + '\n')
                        self.logger.info(eval_str)
                        self.logger.info(
                            self.tester._format_eval_results(eval_res) + '\n')
                # ================= mini-batch end ==================== #
                if self.validate_every < 0 and self.dev_data is not None:
                    eval_res = self._do_validation(epoch=epoch, step=self.step)
                    eval_str = "Evaluation on dev at Epoch {}/{}. Step:{}/{}: ".format(
                        epoch, self.n_epochs, self.step, self.n_steps)
                    # pbar.write(eval_str + '\n')
                    self.logger.info(eval_str)
                    self.logger.info(
                        self.tester._format_eval_results(eval_res) + '\n')
                # lr decay; early stopping
                self.callback_manager.on_epoch_end()
            # =============== epochs end =================== #
            if self.dev_data is not None and (
                    self.validate_every > 0 and self.n_steps %
                    self.validate_every != 0):
                eval_res = self._do_validation(epoch=epoch, step=self.step)
                eval_str = "Evaluation on dev at Epoch {}/{}. Step:{}/{}: ".format(
                    epoch, self.n_epochs, self.step, self.n_steps)
                # pbar.write(eval_str + '\n')
                self.logger.info(eval_str)
                self.logger.info(
                    self.tester._format_eval_results(eval_res) + '\n')
            pbar.close()
            self.pbar = None
        # ============ tqdm end ============== #

    def _do_validation(self, epoch, step):
        self.callback_manager.on_valid_begin()
        res = self.tester.test()

        is_better_eval = False
        if self._better_eval_result(res):
            if self.save_path is not None:
                self._save_model(self.model,
                                 "best_" + "_".join([self.model.__class__.__name__,
                                                     self.metric_key,
                                                     self.start_time]))
            elif self._load_best_model:
                self._best_model_states = {
                    name: param.cpu().clone() for name,
                    param in self.model.state_dict().items()}
            self.best_dev_perf = res
            self.best_dev_epoch = epoch
            self.best_dev_step = step
            is_better_eval = True
        # get validation results; adjust optimizer
        self.callback_manager.on_valid_end(
            res, self.metric_key, self.optimizer, is_better_eval)
        return res

    def _mode(self, model, is_test=False):
        r"""Train mode or Test mode. This is for PyTorch currently.

        :param model: a PyTorch model
        :param bool is_test: whether in test mode or not.

        """
        if is_test:
            model.eval()
        else:
            model.train()

    def _update(self):
        r"""Perform weight update on a model.

        """
        if self.step % self.update_every == 0:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

    def _data_forward(self, network, x):
        x = _build_args(self._forward_func, **x)
        with self.auto_cast():
            y = network(**x)
        if not isinstance(y, dict):
            raise TypeError(
                f"The return value of {_get_func_signature(self._forward_func)} should be dict, got {type(y)}.")
        return y

    def _grad_backward(self, loss):
        r"""Compute gradient with link rules.

        :param loss: a scalar where back-prop starts

        For PyTorch, just do "loss.backward()"
        """
        if (self.step - 1) % self.update_every == 0:
            self._clear_grad(self.optimizer, self.set_grad_to_none)
        self.grad_scaler.scale(loss).backward()

    def _clear_grad(self, optimizer, set_to_none=True):
        param_groups = optimizer.param_groups
        for group in param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()

    def _compute_loss(self, predict, truth):
        r"""Compute loss given prediction and ground truth.

        :param predict: prediction dict, produced by model.forward
        :param truth: ground truth dict, produced by batch_y
        :return: a scalar
        """
        return self.losser(predict, truth)

    def _save_model(self, model, model_name, only_param=False):
        r""" stores state_dict or model that does not contain graphics card information
        :param model:
        :param model_name:
        :param only_param:
        :return:
        """
        if self.save_path is not None:
            model_path = os.path.join(self.save_path, model_name)
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path, exist_ok=True)
            if _model_contains_inner_module(model):
                model = model.module
            if only_param:
                state_dict = model.state_dict()
                for key in state_dict:
                    state_dict[key] = state_dict[key].cpu()
                torch.save(state_dict, model_path)
            else:
                model.cpu()
                torch.save(model, model_path)
                model.to(self._model_device)

    def _load_model(self, model, model_name, only_param=False):
        if self.save_path is not None:
            model_path = os.path.join(self.save_path, model_name)
            if only_param:
                states = torch.load(model_path)
            else:
                states = torch.load(model_path).state_dict()
            if _model_contains_inner_module(model):
                model.module.load_state_dict(states)
            else:
                model.load_state_dict(states)
        elif hasattr(self, "_best_model_states"):
            model.load_state_dict(self._best_model_states)
        else:
            return False
        return True

    def _better_eval_result(self, metrics):
        r"""Check if the current epoch yields better validation results.

        :return bool value: True means current results on dev set is the best.
        """
        indicator, indicator_val = _check_eval_results(
            metrics, self.metric_key, self.metrics)
        if self.metric_key is None:
            self.metric_key = indicator
        is_better = True
        if self.best_metric_indicator is None:
            # first-time validation
            self.best_metric_indicator = indicator_val
        else:
            if self.increase_better is True:
                if indicator_val > self.best_metric_indicator:
                    self.best_metric_indicator = indicator_val
                else:
                    is_better = False
            else:
                if indicator_val < self.best_metric_indicator:
                    self.best_metric_indicator = indicator_val
                else:
                    is_better = False
        return is_better

    @property
    def is_master(self):
        return True


DEFAULT_CHECK_BATCH_SIZE = 2
DEFAULT_CHECK_NUM_BATCH = 2


def _get_value_info(_dict):
    # given a dict value, return information about this dict's value. Return
    # list of str
    strs = []
    for key, value in _dict.items():
        _str = ''
        if isinstance(value, torch.Tensor):
            _str += "\t{}: (1)type:torch.Tensor (2)dtype:{}, (3)shape:{} ".format(key,
                                                                                  value.dtype, value.size())
        elif isinstance(value, np.ndarray):
            _str += "\t{}: (1)type:numpy.ndarray (2)dtype:{}, (3)shape:{} ".format(
                key, value.dtype, value.shape)
        else:
            _str += "\t{}: type:{}".format(key, type(value))
        strs.append(_str)
    return strs


def _check_code(
        dataset,
        model,
        losser,
        metrics,
        forward_func,
        batch_size=DEFAULT_CHECK_BATCH_SIZE,
        dev_data=None,
        metric_key=None,
        check_level=0):
    # check get_loss
    model_device = _get_model_device(model=model)
    _iter = DataSetIter(dataset, batch_size=batch_size, sampler=None)

    for batch_count, (batch_x, batch_y) in enumerate(_iter):
        _move_dict_value_to_device(batch_x, batch_y, device=model_device)
        # forward check
        if batch_count == 0:
            info_str = ""
            input_fields = _get_value_info(batch_x)
            target_fields = _get_value_info(batch_y)
            if len(input_fields) > 0:
                info_str += "input fields after batch(if batch size is {}):\n".format(
                    batch_size)
                info_str += "\n".join(input_fields)
                info_str += '\n'
            else:
                raise RuntimeError("There is no input field.")
            if len(target_fields) > 0:
                info_str += "target fields after batch(if batch size is {}):\n".format(
                    batch_size)
                info_str += "\n".join(target_fields)
                info_str += '\n'
            else:
                info_str += 'There is no target field.'
            logger.info(info_str)
            _check_forward_error(forward_func=forward_func, dataset=dataset,
                                 batch_x=batch_x, check_level=check_level)
        refined_batch_x = _build_args(forward_func, **batch_x)
        pred_dict = model(**refined_batch_x)
        func_signature = _get_func_signature(forward_func)
        if not isinstance(pred_dict, dict):
            raise TypeError(
                f"The return value of {func_signature} should be `dict`, not `{type(pred_dict)}`.")

        # loss check
        try:
            loss = losser(pred_dict, batch_y)
            # check loss output
            if batch_count == 0:
                if not isinstance(loss, torch.Tensor):
                    raise TypeError(
                        f"The return value of {_get_func_signature(losser.get_loss)} should be `torch.Tensor`, "
                        f"but got `{type(loss)}`.")
                if len(loss.size()) != 0:
                    raise ValueError(
                        f"The size of return value of {_get_func_signature(losser.get_loss)} is {loss.size()}, "
                        f"should be torch.size([])")
            loss.backward()
        except _CheckError as e:
            # TODO: another error raised if _CheckError caught
            pre_func_signature = _get_func_signature(forward_func)
            _check_loss_evaluate(
                prev_func_signature=pre_func_signature,
                func_signature=e.func_signature,
                check_res=e.check_res,
                pred_dict=pred_dict,
                target_dict=batch_y,
                dataset=dataset,
                check_level=check_level)
        model.zero_grad()
        if batch_count + 1 >= DEFAULT_CHECK_NUM_BATCH:
            break

    if dev_data is not None:
        tester = Tester(data=dev_data[:batch_size * DEFAULT_CHECK_NUM_BATCH],
                        model=model,
                        metrics=metrics,
                        batch_size=batch_size,
                        verbose=-1,
                        use_tqdm=False)
        evaluate_results = tester.test()
        _check_eval_results(
            metrics=evaluate_results,
            metric_key=metric_key,
            metric_list=metrics)


def _check_eval_results(metrics, metric_key, metric_list):
    if isinstance(metrics, tuple):
        loss, metrics = metrics

    if isinstance(metrics, dict):
        metric_dict = list(metrics.values())[0]

        if metric_key is None:
            indicator_val, indicator = list(
                metric_dict.values())[0], list(
                metric_dict.keys())[0]
        else:
            # metric_key is set
            if metric_key not in metric_dict:
                raise RuntimeError(
                    f"metric key {metric_key} not found in {metric_dict}")
            indicator_val = metric_dict[metric_key]
            indicator = metric_key
    else:
        raise RuntimeError(
            "Invalid metrics type. Expect {}, got {}".format(
                (tuple, dict), type(metrics)))
    return indicator, indicator_val
