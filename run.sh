# 05/22/2024
python classification_train.py --model_name all --dataset_choice all --CUDA cuda:0 --start_per 0.0 --end_per 0.25
python classification_train.py --model_name all --dataset_choice all --CUDA cuda:1 --start_per 0.25 --end_per 0.50
python classification_train.py --model_name all --dataset_choice all --CUDA cuda:2 --start_per 0.50 --end_per 0.75
python classification_train.py --model_name all --dataset_choice all --CUDA cuda:3 --start_per 0.75 --end_per 1.0