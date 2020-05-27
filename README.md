# Improving BERT with Self-Supervised Attention

Codes and corpora for paper "Improving BERT with Self-Supervised Attention" 
[https://arxiv.org/abs/2004.03808](https://arxiv.org/abs/2004.03808).

## Requirement

* pytorch: 1.4.0
* python: 3.5.2
* numpy: 1.16.4

## Trained Checkpoints

You can download **ssa-BERT-base**, **ssa-BERT-large**, **ssa-RoBERTa-base** and **ssa-RoBERTa-large** from here: url:https://pan.baidu.com/s/1x-Whii8ZmntxUXUbf-Qltg  password:00bg

After that, you can reproduce the results using specific checkpoint and related parameters. 

For example, reproduce **ssa-BERT-base** results:
```
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/ssa_base_re.sh &> log/ssa_base_re.out &
```

## Step 1: prepare GLUE datasets

Before running this code you must download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpack it to some directory.


## Step 2: train with ssa-BERT

For example, **ssa-BERT-base** model on **RTE** dataset:

```
CUDA_VISIBLE_DEVICES=0 nohup python -u run_ssa.py --data_dir=./glue_data/ --task_name=RTE --num_train_epochs=5.0 --use_saved=0 &> log/ssa_rte_base.out &
```

Note:

* There are several important parameters need to be fine-tuned, such as: `cls_weight`, `attention_threshold`, 
`aug_loss_weight`, `aug_threshold`, `rm_threshold`, `use_saved`, `share_weight`. The parameter interval can refer to the paper.

**ssa-RoBERTa-large** model on **RTE** dataset:
```
CUDA_VISIBLE_DEVICES=0,1 nohup python -u Roberta/run_ssa.py --data_dir=./glue_data/ --model_name_or_path=roberta-large --task_name=RTE --num_train_epochs=3.0 --use_saved=0 &> log/ssa_ro_rte_large.out &
```

You can only run with vanilla BERT or RoBERTa, for example:
```
CUDA_VISIBLE_DEVICES=0 nohup python -u run_ssa.py --data_dir=./glue_data/ --task_name=RTE --num_train_epochs=5.0 --only_bert=1 &> log/rte_bert.out &

```


## Citation

```
@article{kou2020improving,
  title={Improving BERT with Self-Supervised Attention},
  author={Kou, Xiaoyu and Yang, Yaming and Wang, Yujing and Zhang, Ce and Chen, Yiren and Tong, Yunhai and Zhang, Yan and Bai, Jing},
  journal={arXiv preprint arXiv:2004.03808},
  year={2020}
}
```
