from statistics import mean

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from holisticai.bias.mitigation.inprocessing.commons import Logging


class TrainArgs:
    """
    training configuration for Trainer Class
    """

    def __init__(
        self,
        initial_epoch=0,
        epochs=10,
        adversary_loss_weight=0.1,
        initial_lr=0.001,
        verbose=0,
        print_interval=1,
        batch_size=32,
        shuffle_data=True,
    ):
        self.epochs = epochs
        self.adversary_loss_weight = adversary_loss_weight
        self.initial_lr = initial_lr
        self.initial_epoch = initial_epoch
        self.verbose = verbose
        self.print_interval = print_interval
        self.batch_size = batch_size
        self.shuffle_data = shuffle_data


class Trainer:
    """
    Trainer class support traditional classifier training and adverarial training.
    """

    def __init__(self, model, dataset, train_args, use_debias):
        self.train_args = train_args
        self.model = model
        self.use_debias = use_debias
        self.build(dataset)
        log_params = [
            ("iteration", int),
            ("classifier loss", float),
            ("adversarial loss", float),
        ]
        self.steps_per_epoch = (
            len(dataset) + self.train_args.batch_size - 1
        ) // self.train_args.batch_size
        self.total_iterations = self.steps_per_epoch * self.train_args.epochs
        self.logger = Logging(
            log_params=log_params,
            total_iterations=self.total_iterations,
            logger_format="epochs",
            epochs=self.train_args.epochs,
        )

    def build(self, dataset):
        """Build loss function and optimizers"""

        self.trainloader = DataLoader(
            dataset=dataset,
            batch_size=self.train_args.batch_size,
            shuffle=self.train_args.shuffle_data,
        )

        self.loss_cls_fn = nn.BCELoss()
        self.loss_adv_fn = nn.BCELoss()

        self.optimizer_cls = optim.SGD(
            self.model.classifier.parameters(),
            lr=self.train_args.initial_lr,
            momentum=0.9,
        )
        self.optimizer_adv = optim.SGD(
            self.model.adversarial.parameters(),
            lr=self.train_args.initial_lr,
            momentum=0.9,
        )

        self.scheduler_cls = torch.optim.lr_scheduler.LinearLR(
            self.optimizer_cls, start_factor=0.99, total_iters=20
        )
        self.scheduler_adv = torch.optim.lr_scheduler.LinearLR(
            self.optimizer_adv, start_factor=0.99, total_iters=20
        )

    def train(self):
        if self.use_debias:
            train_step = self._train_step_with_debias
        else:
            train_step = self._train_step_without_debias

        """Traditional one epoch"""
        for epoch in range(self.train_args.epochs):
            cls_loss, adv_loss = [], []
            for it, data in enumerate(self.trainloader, 1):

                loss_cls, loss_adv = train_step(data)

                cls_loss.append(loss_cls.item())
                adv_loss.append(0 if loss_adv is None else loss_adv.item())

                self._logging_progress(epoch, it, mean(cls_loss), mean(adv_loss))

            self.scheduler_cls.step()
            self.scheduler_adv.step()

    def _train_step_with_debias(self, data):
        """one step advesarial training classifier"""
        inputs, labels = data
        outputs = self.model(*inputs)

        # grad classifier Loss - grad adversarial Loss
        loss_adv = self.loss_adv_fn(outputs[1], labels[1])
        classifier_params = list(self.model.classifier.parameters())
        dloss_adv = torch.autograd.grad(
            outputs=loss_adv, inputs=classifier_params, retain_graph=True
        )

        loss_cls = self.loss_cls_fn(outputs[0], labels[0])
        dloss_cls = torch.autograd.grad(outputs=loss_cls, inputs=classifier_params)

        self._update_gradients(classifier_params, dloss_cls, dloss_adv)
        self.optimizer_cls.step()

        # grad adversarial Loss
        outputs = self.model(*inputs)
        loss_adv = self.loss_adv_fn(outputs[1], labels[1])
        adversarial_params = list(self.model.adversarial.parameters())
        dloss_adv = torch.autograd.grad(
            outputs=loss_adv, inputs=adversarial_params, retain_graph=True
        )
        for param, grad in zip(adversarial_params, dloss_adv):
            param.grad = grad
        self.optimizer_adv.step()

        return loss_cls, loss_adv

    def _train_step_without_debias(self, data):
        """Traditional classifier tran step"""
        inputs, labels = data
        outputs = self.model(*inputs)

        # Classifier
        loss_cls = self.loss_cls_fn(outputs[0], labels[0])
        classifier_params = list(self.model.classifier.parameters())
        dloss_cls = torch.autograd.grad(outputs=loss_cls, inputs=classifier_params)
        for param, grad in zip(classifier_params, dloss_cls):
            param.grad = grad
        self.optimizer_cls.step()

        return loss_cls, None

    def _update_gradients(self, classifier_params, dloss_cls, dloss_adv):
        """update classifier gradients with adversarial model"""
        normalize = lambda x: x / (torch.norm(x) + np.finfo(np.float32).tiny)
        for param, grad, grad_adv in zip(classifier_params, dloss_cls, dloss_adv):
            unit_adversary_grad = normalize(grad_adv)
            grad -= torch.sum(grad * unit_adversary_grad) * unit_adversary_grad
            grad -= self.train_args.adversary_loss_weight * grad_adv
            param.grad = grad

    def _logging_progress(self, epoch, it, cls_loss, adv_loss):
        iteration = epoch * self.steps_per_epoch + it
        if self.train_args.verbose > 0:
            if (iteration % self.train_args.print_interval) == 0 or (
                iteration % self.total_iterations
            ) == 0:
                self.logger.update(iteration, cls_loss, adv_loss)
