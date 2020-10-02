# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import time

import torch

from common.dataset import Dataset
from tester import OutKGTester
from trainer import OutKGTrainer


def get_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset", default="WN18RR", type=str, help="dataset name")
    parser.add_argument(
        "-model_name", default="DisMult", type=str, help="initial embedding model"
    )
    parser.add_argument(
        "-opt",
        default="adagrad",
        type=str,
        help="optimizer. Currenty only adagrad and adam are supported",
    )
    parser.add_argument(
        "-emb_method",
        default="ERAverage",
        type=str,
        help="method to find new enitity's embedding",
    )
    parser.add_argument("-emb_dim", default=200, type=int, help="embedding dimension")
    parser.add_argument(
        "-neg_ratio",
        default=1,
        type=int,
        help="number of negative examples per positive example",
    )
    parser.add_argument("-batch_size", default=1000, type=int, help="batch size")
    parser.add_argument(
        "-simulated_batch_size",
        default=1000,
        type=int,
        help="batch size to be simulated",
    )
    parser.add_argument(
        "-save_each", default=5, type=int, help="validate every k epochs"
    )
    parser.add_argument("-ne", default=1000, type=int, help="number of epochs")
    parser.add_argument("-lr", default=0.1, type=float, help="learning rate")
    parser.add_argument(
        "-reg_lambda", default=0.01, type=float, help="l2 regularization parameter"
    )
    parser.add_argument(
        "-reg_ls",
        default=0.01,
        type=float,
        help="l2 regularization parameter (for Least Squares)",
    )
    parser.add_argument(
        "-val", default=False, type=bool, help="start validation after training"
    )
    parser.add_argument(
        "-use_custom_reg", default=True, type=bool, help="use custom regularisation"
    )
    parser.add_argument("-use_acc", default=False, type=bool, help="use_acc flag")
    parser.add_argument(
        "-cons_mask", default=False, type=bool, help="Use consistent masking"
    )
    parser.add_argument(
        "-mask_prob",
        default=0.5,
        type=float,
        help="The probability of observed entities",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_parameters()
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))
    dataset = Dataset(args.dataset, args.cons_mask, args.mask_prob)
    outKG_trainer = OutKGTrainer(dataset, args)

    print("~~~~ Training ~~~~")
    outKG_trainer.train()

    if args.val or True:
        with torch.no_grad():
            print("~~~~ Select best epoch on validation set ~~~~")
            epochs2test = [
                str(int(args.save_each * i))
                for i in range(args.ne // args.save_each)
            ]
            best_mrr = -1.0
            best_epoch = "0"
            valid_performance = None
            for epoch in epochs2test:
                start = time.time()
                print("epoch: ", epoch)
                outKG_tester = OutKGTester(dataset)
                model_path = (
                        "logs/"
                        + args.model_name
                        + "/"
                        + args.emb_method
                        + "/"
                        + args.dataset
                        + "/"
                        + epoch
                        + "_"
                        + str(args.lr)
                        + "_"
                        + str(args.reg_lambda)
                        + "_"
                        + str(args.neg_ratio)
                        + "_"
                        + str(args.emb_dim)
                        + ".chkpnt"
                )
                mrr = outKG_tester.test(model_path, "valid")
                if mrr > best_mrr:
                    best_mrr = mrr
                    best_epoch = epoch
                    valid_performance = [
                        best_epoch,
                        outKG_tester.measure.hit1,
                        outKG_tester.measure.hit3,
                        outKG_tester.measure.hit10,
                        outKG_tester.measure.mr,
                        outKG_tester.measure.mrr,
                    ]
                print("validation time: ", time.time() - start)

            print("Best epoch: " + best_epoch)
            if valid_performance:
                print("\nValidation Performance : ")
                print("\tBest epoch : ", valid_performance[0])
                print("\tHit@1 : ", valid_performance[1])
                print("\tHit@3 : ", valid_performance[2])
                print("\tHit@10 : ", valid_performance[3])
                print("\tMR : ", valid_performance[4])
                print("\tMRR : ", valid_performance[5])

            print("~~~~ Testing on the best epoch ~~~~")
            best_model_path = (
                    "logs/"
                    + args.model_name
                    + "/"
                    + args.emb_method
                    + "/"
                    + args.dataset
                    + "/"
                    + best_epoch
                    + "_"
                    + str(args.lr)
                    + "_"
                    + str(args.reg_lambda)
                    + "_"
                    + str(args.neg_ratio)
                    + "_"
                    + str(args.emb_dim)
                    + ".chkpnt"
            )
            outKG_tester = OutKGTester(dataset)
            start = time.time()
            outKG_tester.test(best_model_path, "test")
            print("Inference time: ", time.time() - start)

        print("Done :) ")
