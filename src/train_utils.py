# -*-coding:utf-8 -*-
import numpy as np
import torch
import random
import os
import operator
from collections import deque
from glob import glob


class TrainParams(dict):
    """
    Train Parameter used to initialize Model
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @property
    def total_train_steps(self):
        return int(self.num_train_steps * self.epoch_size)


def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def get_torch_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


class ModelSave():
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, ckpt_path, keep_ckpt_max=3, continue_train=False, save_best=True):
        self.best_valid_loss = float('inf')
        self.continue_train = continue_train
        self.ckpt_path = ckpt_path
        self.save_best = save_best
        self.keep_ckpt_max = keep_ckpt_max  # 最多保存多少ckpt
        self.ckpt_list = deque()
        self.counter = 0

    @staticmethod
    def remove_dir(model_dir):
        import shutil
        try:
            shutil.rmtree(model_dir)
        except Exception as e:
            print(f'Warning: {e} not exists')
        else:
            print(f'{model_dir} model cleaned')

    def init(self):
        """
        Create model dir or rm model dir
        """
        if not self.continue_train:
            self.remove_dir(self.ckpt_path)

        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

    def __call__(self, train_loss, valid_loss, epoch, global_step, model, optimizer, scheduler=None):
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss
        }

        if scheduler:
            checkpoint.update({'scheduler_state_dict': scheduler.state_dict()})

        ckpt_file = f'{self.ckpt_path}/ckpt_{global_step}.pth'
        torch.save(checkpoint, ckpt_file)
        if ckpt_file not in self.ckpt_list:
            self.ckpt_list.append(ckpt_file)

        if len(self.ckpt_list) > self.keep_ckpt_max:
            os.remove(self.ckpt_list.popleft())  # remove earliest ckpt

        if self.save_best:
            best_ckpt = f'{self.ckpt_path}/best_ckpt.pth'
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                torch.save(checkpoint, best_ckpt)


class EarlyStop(object):
    """
    Run Early Stop on the end of each valid evaluation
    """
    mode_dict = {'min': operator.lt, 'max': operator.gt}

    def __init__(self, monitor, mode='min', min_delta=0.0, patience=3, verbose=False):
        self.monitor = monitor
        self.mode = mode # monitor metric the lower/bigger the better
        self.monitor_op = self.mode_dict[mode]
        self.min_delta = min_delta if self.monitor_op == torch.gt else -min_delta
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = -float('inf') if self.monitor_op == operator.gt else float('inf')

    def check(self, valid_metrics):
        assert self.monitor in valid_metrics, f'monitor metric {self.monitor} not in valid metrics'
        cur_score = valid_metrics[self.monitor]
        should_stop=False
        if self.monitor_op(cur_score - self.min_delta, self.best_score):
            self.counter = 0
            self.best_score = cur_score
        else:
            self.counter +=1
            if self.counter>=self.patience:
                should_stop=True
                reason = f"{self.monitor} did not improve in the last {self.patience} evaluation"
                if self.verbose:
                    print(reason)
        return should_stop


def load_checkpoint(ckpt):
    """
    Checkpoint Loading, should be used along with ModelSave
    1. try to locate *best.pth checkpoint
    2. if not find the *.pth with biggest epoch
    3. load state dict with different environment
    """
    files = glob(os.path.join(ckpt, '*best.pth'))
    if not files:
        file = sorted(glob(os.path.join(ckpt, '*.pth')))[-1]
    else:
        file = files[0]
    if torch.cuda.is_available():
        # Use saving environment
        map_location = lambda storage, loc: storage.cuda()
    else:
        # pytorch0.4.0及以上版本
        map_location = torch.device('cpu')

    return torch.load(file, map_location=map_location)


