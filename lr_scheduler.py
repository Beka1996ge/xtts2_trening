#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
სასწავლო სიჩქარის მართვის კლასები XTTS-v2 მოდელის ტრენინგისთვის
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class NoamLRScheduler(_LRScheduler):
    """
    Transformer-ებისთვის ოპტიმიზირებული Noam სასწავლო სიჩქარის დამგეგმავი.
    წყარო: "Attention Is All You Need" ნაშრომი
    """

    def __init__(self, optimizer, model_dim, warmup_steps, last_epoch=-1, min_lr=1e-5):
        """
        Args:
            optimizer: პითონის ოპტიმიზატორი
            model_dim (int): მოდელის განზომილება
            warmup_steps (int): გათბობის ნაბიჯების რაოდენობა
            last_epoch (int, optional): ბოლო ეპოქა გასაგრძელებლად. Default: -1
            min_lr (float, optional): მინიმალური სასწავლო სიჩქარე. Default: 1e-5
        """
        self.model_dim = model_dim
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        super(NoamLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """გამოთვლის ახალ სასწავლო სიჩქარეს Noam ფორმულის მიხედვით"""
        step = max(1, self._step_count)

        # Noam ფორმულა: lr = factor * (model_dim ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
        scale = self.model_dim ** (-0.5) * min(
            step ** (-0.5), step * self.warmup_steps ** (-1.5)
        )

        return [max(self.min_lr, base_lr * scale) for base_lr in self.base_lrs]


class WarmupCosineLR(_LRScheduler):
    """
    გათბობის ფაზის მქონე კოსინუსური სასწავლო სიჩქარის დამგეგმავი.
    """

    def __init__(
            self,
            optimizer,
            max_steps,
            warmup_steps=0,
            min_lr=1e-5,
            last_epoch=-1
    ):
        """
        Args:
            optimizer: პითონის ოპტიმიზატორი
            max_steps (int): ნაბიჯების მაქსიმალური რაოდენობა
            warmup_steps (int, optional): გათბობის ნაბიჯების რაოდენობა. Default: 0
            min_lr (float, optional): მინიმალური სასწავლო სიჩქარე. Default: 1e-5
            last_epoch (int, optional): ბოლო ეპოქა გასაგრძელებლად. Default: -1
        """
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """გამოთვლის ახალ სასწავლო სიჩქარეს კოსინუსური ფორმულის მიხედვით"""
        step = max(1, self._step_count)

        # გათბობის ფაზაში
        if step <= self.warmup_steps:
            # წრფივი ზრდა 0-დან base_lr-მდე
            scale = float(step) / float(max(1, self.warmup_steps))
            return [base_lr * scale for base_lr in self.base_lrs]

        # კოსინუსური სასწავლო სიჩქარის შემცირება
        progress = float(step - self.warmup_steps) / float(
            max(1, self.max_steps - self.warmup_steps)
        )
        # კოსინუსური შემცირება 1-დან 0-მდე
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        # min_lr-სა და base_lr-ს შორის ინტერპოლაცია
        return [
            self.min_lr + (base_lr - self.min_lr) * scale
            for base_lr in self.base_lrs
        ]


class CyclicLR(_LRScheduler):
    """
    ციკლური სასწავლო სიჩქარის დამგეგმავი, როგორც ეს აღწერილია
    "Cyclical Learning Rates for Training Neural Networks" ნაშრომში.
    """

    def __init__(
            self,
            optimizer,
            base_lr,
            max_lr,
            step_size_up=2000,
            step_size_down=None,
            mode='triangular',
            gamma=1.0,
            scale_fn=None,
            scale_mode='cycle',
            last_epoch=-1
    ):
        """
        Args:
            optimizer: პითონის ოპტიმიზატორი
            base_lr (float or list): საწყისი სასწავლო სიჩქარე, რომელიც ასევე არის მინიმალური
            max_lr (float or list): მაქსიმალური სასწავლო სიჩქარე
            step_size_up (int): ნაბიჯების რაოდენობა მინიმუმიდან მაქსიმუმამდე
            step_size_down (int): ნაბიჯების რაოდენობა მაქსიმუმიდან მინიმუმამდე
            mode (str): რეჟიმი: 'triangular', 'triangular2' ან 'exp_range'
            gamma (float): გამოიყენება exp_range რეჟიმისთვის
            scale_fn (function): საკუთარი სკალირების ფუნქცია
            scale_mode (str): სკალირების რეჟიმი: 'cycle' ან 'iterations'
        """
        if step_size_down is None:
            step_size_down = step_size_up

        self.base_lrs = [base_lr] if isinstance(base_lr, float) else base_lr
        self.max_lrs = [max_lr] if isinstance(max_lr, float) else max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down
        self.mode = mode
        self.gamma = gamma

        # სკალირების ფუნქციის განსაზღვრა
        if scale_fn is None:
            if mode == 'triangular':
                self.scale_fn = lambda x: 1.0
                self.scale_mode = 'cycle'
            elif mode == 'triangular2':
                self.scale_fn = lambda x: 1.0 / (2.0 ** (x - 1))
                self.scale_mode = 'cycle'
            elif mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        super(CyclicLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """გამოთვლის ახალ სასწავლო სიჩქარეს ციკლური ფორმულის მიხედვით"""
        cycle_size = self.step_size_up + self.step_size_down
        cycle = math.floor(1 + self._step_count / cycle_size)
        x = 1. + self._step_count - cycle_size * (cycle - 1)

        # ციკლის ფაზის გამოთვლა
        if x <= self.step_size_up:
            scale_factor = x / self.step_size_up
        else:
            scale_factor = 1. - (x - self.step_size_up) / self.step_size_down

        # სკალირების ფაქტორის გამოყენება
        if self.scale_mode == 'cycle':
            lr_scale = self.scale_fn(cycle)
        else:
            lr_scale = self.scale_fn(self._step_count)

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            # სასწავლო სიჩქარის ინტერპოლაცია
            lr = base_lr + (max_lr - base_lr) * scale_factor * lr_scale
            lrs.append(lr)

        return lrs