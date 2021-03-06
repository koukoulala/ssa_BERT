#!/usr/bin/env bash

#MRPC
echo "it's: MRPC"
echo
echo
python -u run_ssa.py \
  --task_name='MRPC' \
  --data_dir=./glue_data/ \
  --output_dir='./results_final/large' \
  --bert_model='bert-large-uncased' \
  --ckpt='./ssa_ckpt_large/MRPC/ssa_base' \
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
  --data_dir=./glue_data/ \
  --output_dir='./results_final/large' \
  --bert_model='bert-large-uncased' \
  --ckpt='./ssa_ckpt_large/CoLA/ssa_base' \
  --do_first_eval \
  --cls_weight=0.9 \
  --attention_threshold=0.3 \
  --share_weight=1 \
"$@"

#RTE
echo "it's: RTE"
echo
echo
python -u run_ssa.py \
  --task_name='RTE' \
  --data_dir=./glue_data/ \
  --output_dir='./results_final/large' \
  --bert_model='bert-large-uncased' \
  --ckpt='./ssa_ckpt_large/RTE/ssa_base' \
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
  --data_dir=./glue_data/ \
  --output_dir='./results_final/large' \
  --bert_model='bert-large-uncased' \
  --ckpt='./ssa_ckpt_large/STS-B/ssa_base' \
  --do_first_eval \
  --cls_weight=0.5 \
  --attention_threshold=0.3 \
  --share_weight=1 \
"$@"

#SST-2
echo "it's: SST-2"
echo
echo
python -u run_ssa.py \
  --task_name='SST-2' \
  --data_dir=./glue_data/ \
  --output_dir='./results_final/large' \
  --bert_model='bert-large-uncased' \
  --ckpt='./ssa_ckpt_large/SST-2/ssa_base' \
  --do_first_eval \
  --cls_weight=0.999 \
  --attention_threshold=0.3 \
  --share_weight=1 \
"$@"

#QNLI
echo "it's: QNLI"
echo
echo
python -u run_ssa.py \
  --task_name='QNLI' \
  --data_dir=./glue_data/ \
  --output_dir='./results_final/large' \
  --bert_model='bert-large-uncased' \
  --ckpt='./ssa_ckpt_large/QNLI/ssa_base' \
  --do_first_eval \
  --cls_weight=0.9 \
  --attention_threshold=0.3 \
  --share_weight=1 \
"$@"

#QQP
echo "it's: QQP"
echo
echo
python -u run_ssa.py \
  --task_name='QQP' \
  --data_dir=./glue_data/ \
  --output_dir='./results_final/large' \
  --bert_model='bert-large-uncased' \
  --ckpt='./ssa_ckpt_large/QQP/ssa_base' \
  --do_first_eval \
  --cls_weight=0.999 \
  --attention_threshold=0.3 \
  --share_weight=1 \
"$@"

#MNLI
echo "it's: MNLI"
echo
echo
python -u run_ssa.py \
  --task_name='MNLI' \
  --data_dir=./glue_data/ \
  --output_dir='./results_final/large' \
  --bert_model='bert-large-uncased' \
  --ckpt='./ssa_ckpt_large/MNLI/ssa_base' \
  --do_first_eval \
  --cls_weight=0.2 \
  --attention_threshold=0.1 \
  --share_weight=1 \
"$@"

