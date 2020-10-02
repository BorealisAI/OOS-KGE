# Out-of-Sample Representation Learning for Multi-Relational Graphs

This repo containts the PyTorch implementation of the model presented in [Out-of-Sample Representation Learning for Multi-Relational Graphs](https://arxiv.org/pdf/2004.13230.pdf) accepted to findings of EMNLP 2020.

## Dependencies

* `Python` version 3.6
* `Numpy` version 1.16.0
* `PyTorch` version 1.5.0


## Running a model

To train the model run `python main.py` from the `src` directory, but first you need to specify a few parameters.

Here is a list of important parameters:
```
-dataset            	dataset to use (WN18RR or FB15K-237)
-model_name         	embedding model (currently only DisMult is supported)
-emb_method         	aggregation functions to compute unobserved representations
-mask_prob              The probability of observed entities (equivalent to (1-psi) in the paper)
-opt                	optimizer to use. Currenty only adagrad and adam are supported
-lr                     learning rate
-reg_lambda         	l2 regularization parameter
-reg_ls             	l2 regularization parameter for least square
-ne                 	number of epochs
-save_each          	validation frequency
-batch_size         	batch size
-simulated_batch_size   batch size to be simulated
-neg_ratio          	number of negative examples per positive example
```


## Reproducing the Results in the Paper

To reproduce results of `oDistMult-ERAvg` models, run the following commands.

### WN18RR dataset

```bash
python main.py -dataset "WN18RR" -model_name "DisMult" -emb_method "ERAverage" -mask_prob 0.5 -ne 1000 -lr 0.1 -reg_lambda 0.01  -emb_dim 200 -neg_ratio 1 -batch_size 250 -simulated_batch_size 1000 -save_each 100
```


### FB15K-237

```bash
python main.py -dataset "FB15k-237" -model_name "DisMult" -emb_method "ERAverage" -mask_prob 0.5 -ne 1000 -lr 0.01 -reg_lambda 0.0001  -emb_dim 200 -neg_ratio 1 -batch_size 250 -simulated_batch_size 1000 -save_each 100
```


## Cite

If you found this codebase or our work useful, please cite:
```text
@article{albooyeh2020out,
  title={Out-of-Sample Representation Learning for Multi-Relational Graphs},
  author={Albooyeh, Marjan and Goel, Rishab and Kazemi, Seyed Mehran},
  journal={arXiv preprint arXiv:2004.13230},
  year={2020}
}
```


## License

Licensed under Creative Commons Attribution-NonCommercial-ShareALike (CC BY-NC-SA). For more information please read
https://creativecommons.org/licenses/by-nc-sa/4.0/
