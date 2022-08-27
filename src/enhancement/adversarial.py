# -*-coding:utf-8 -*-

import torch


class FGM():
    def __init__(self, model, epsilon=1, emb_name='embedding'):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}
        self.attack_params = []
        self.locate_params()

    def locate_params(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.attack_params.append(name)
        print(f'Attack parameters {self.attack_params}')

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    def __init__(self, model, epsilon=1, alpha=0.3, emb_name='embedding'):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_name = emb_name
        self.emb_backup = {}
        self.grad_backup = {}
        self.attack_params = []
        self.locate_params()

    def locate_params(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.attack_params.append(name)
        print(f'Attack parameters {self.attack_params}')

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]