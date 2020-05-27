
# 5.11
CUDA_VISIBLE_DEVICES=5 nohup python -u run_ssa.py --data_dir=/data/data5/kxy/glue_data/ --task_name=RTE --num_train_epochs=5.0 --use_saved=0 &> log/try_rte.out &
CUDA_VISIBLE_DEVICES=5 nohup python -u run_ssa.py --data_dir=/data/data5/kxy/glue_data/ --task_name=RTE --num_train_epochs=5.0 --only_bert=1 &> log/try_rte_bert.out &

# 5.27
CUDA_VISIBLE_DEVICES=0,1 nohup python -u Roberta/run_ssa.py --data_dir=/data/data5/kxy/glue_data/ --task_name=RTE --num_train_epochs=3.0 --use_saved=0 &> log/try_ro_rte.out &
CUDA_VISIBLE_DEVICES=1,2 nohup python -u Roberta/run_ssa.py --data_dir=/data/data5/kxy/glue_data/ --task_name=RTE --num_train_epochs=3.0 --only_bert=1 &> log/try_ro_rte_bert.out &
CUDA_VISIBLE_DEVICES=0 nohup python -u run_ssa.py --data_dir=/data/data5/kxy/glue_data/ --task_name=RTE --num_train_epochs=5.0 --use_saved=0 &> log/try_rte.out &

