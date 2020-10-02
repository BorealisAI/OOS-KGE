# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch import nn

from model.DisMultOutKG import DisMultOutKG
from utils import save_model


class OutKGTrainer:
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DisMultOutKG(self.dataset, self.args, self.device)
        self.model = nn.DataParallel(self.model)
        self.predict_loss = nn.Softplus()

    def l2_loss(self):
        return self.model.module.l2_loss()

    def train(self, save=True):
        self.model.train()
        if self.args.use_acc:
            initial_accumulator_value = 0.1
        else:
            initial_accumulator_value = 0.0

        if self.args.use_custom_reg:
            weight_decay = 0.0
        else:
            weight_decay = self.args.reg_lambda

        if self.args.opt == "adagrad":
            print("using adagrad")
            optimizer = torch.optim.Adagrad(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=weight_decay,
                initial_accumulator_value=initial_accumulator_value,
                # this is added because of the consistency to the original tensorflow code
            )

        else:
            print("using adam")
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.reg_lambda,
            )

        iters_per_update = self.args.simulated_batch_size // self.args.batch_size

        if iters_per_update < 1:
            raise ("Actual batch size smaller than batch size to be simulated.")
        else:
            print("iterations before the gradient step : ", iters_per_update)

        for epoch in range(self.args.ne):
            optimizer.zero_grad()
            last_batch = False
            total_loss = 0.0
            num_iters = 1
            while not last_batch:

                triples, l, new_ent_mask = self.dataset.next_batch(
                    self.args.batch_size,
                    neg_ratio=self.args.neg_ratio,
                    device=self.device,
                )
                last_batch = self.dataset.was_last_batch()
                scores, predicted_emb = self.model(triples, new_ent_mask)
                predict_loss = torch.sum(self.predict_loss(-l * scores))
                if self.args.use_custom_reg:
                    if num_iters % iters_per_update == 0 or last_batch == True:
                        l2_loss = (
                                self.args.reg_lambda
                                * self.l2_loss()
                                / (
                                    self.dataset.num_batch_simulated(
                                        self.args.simulated_batch_size
                                    )
                                )
                        )
                        loss = predict_loss + l2_loss

                    else:
                        loss = predict_loss
                else:
                    loss = predict_loss

                loss.backward()
                if num_iters % iters_per_update == 0 or last_batch == True:
                    if last_batch:
                        print("last batch triggered gradient update.")
                        print(
                            "remaining iters for gradient update :",
                            num_iters % iters_per_update,
                        )
                    optimizer.step()
                    optimizer.zero_grad()
                total_loss += loss
                num_iters += 1

            print(
                "Loss in iteration "
                + str(epoch)
                + ": "
                + str(total_loss.item())
                + "("
                + self.dataset.dataset_name
                + ")"
            )
            if epoch % self.args.save_each == 0 and save:
                print('save model...')
                save_model(
                    self.model,
                    self.args.model_name,
                    self.args.emb_method,
                    self.dataset.dataset_name,
                    epoch,
                    self.args.lr,
                    self.args.reg_lambda,
                    self.args.neg_ratio,
                    self.args.emb_dim,
                )
