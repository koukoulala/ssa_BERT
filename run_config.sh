
# 5.11
CUDA_VISIBLE_DEVICES=5 nohup python -u run_ssa.py --data_dir=/data/data5/kxy/glue_data/ --task_name=RTE --num_train_epochs=5.0 --use_saved=0 &> log/try_rte.out &
