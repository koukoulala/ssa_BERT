#!/usr/bin/env bash

#MRPC
echo "it's: MRPC"
echo
echo
python -u run_ssa.py \
  --task_name='MRPC' \
  --data_dir=./glue_data \
  --output_dir='./results_final/base' \
  --bert_model='bert-base-uncased' \
  --ckpt='./ssa_ckpt_base/MRPC/ssa_base' \
  --do_first_eval \
  --cls_weight=0.5 \
  --attention_threshold=0.3 \
  --share_weight=1 \
"$@"

#CoLA
echo "it's: CoLA"
echo
echo
python -u run_ssa.py \
  --task_name='CoLA' \
  --data_dir=./glue_data \
  --output_dir='./results_final/base' \
  --bert_model='bert-base-uncased' \
  --ckpt='./ssa_ckpt_base/CoLA/ssa_base' \
  --do_first_eval \
  --cls_weight=0.999 \
  --attention_threshold=0.3 \
  --share_weight=1 \
"$@"

#RTE
echo "it's: RTE"
echo
echo
python -u run_ssa.py \
  --task_name='RTE' \
  --data_dir=./glue_data \
  --output_dir='./results_final/base' \
  --bert_model='bert-base-uncased' \
  --ckpt='./ssa_ckpt_base/RTE/ssa_base' \
  --do_first_eval \
  --cls_weight=0.5 \
  --attention_threshold=0.3 \
  --share_weight=0 \
"$@"

#STS-B
echo "it's: STS-B"
echo
echo
python -u run_ssa.py \
  --task_name='STS-B' \
  --data_dir=./glue_data \
  --output_dir='./results_final/base' \
  --bert_model='bert-base-uncased' \
  --ckpt='./ssa_ckpt_base/STS-B/ssa_base' \
  --do_first_eval \
  --cls_weight=0.9 \
  --attention_threshold=0.3 \
  --share_weight=0 \
"$@"

#SST-2
echo "it's: SST-2"
echo
echo
python -u run_ssa.py \
  --task_name='SST-2' \
  --data_dir=./glue_data \
  --output_dir='./results_final/base' \
  --bert_model='bert-base-uncased' \
  --ckpt='./ssa_ckpt_base/SST-2/ssa_base' \
  --do_first_eval \
  --cls_weight=0.5 \
  --attention_threshold=0.3 \
  --share_weight=1 \
"$@"

#QNLI
echo "it's: QNLI"
echo
echo
python -u run_ssa.py \
  --task_name='QNLI' \
  --data_dir=./glue_data \
  --output_dir='./results_final/base' \
  --bert_model='bert-base-uncased' \
  --ckpt='./ssa_ckpt_base/QNLI/ssa_base' \
  --do_first_eval \
  --cls_weight=0.7 \
  --attention_threshold=0.3 \
  --share_weight=1 \
"$@"

#QQP
echo "it's: QQP"
echo
echo
python -u run_ssa.py \
  --task_name='QQP' \
  --data_dir=./glue_data \
  --output_dir='./results_final/base' \
  --bert_model='bert-base-uncased' \
  --ckpt='./ssa_ckpt_base/QQP/ssa_base' \
  --do_first_eval \
  --cls_weight=0.9 \
  --attention_threshold=0.3 \
  --share_weight=1 \
"$@"

#MNLI
echo "it's: MNLI"
echo
echo
python -u run_ssa.py \
  --task_name='MNLI' \
  --data_dir=./glue_data \
  --output_dir='./results_final/base' \
  --bert_model='bert-base-uncased' \
  --ckpt='./ssa_ckpt_base/MNLI/ssa_base' \
  --do_first_eval \
  --cls_weight=0.999 \
  --attention_threshold=0.3 \
  --share_weight=1 \
"$@"

