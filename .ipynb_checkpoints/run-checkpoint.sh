# 05/22/2024
python classification_train.py --model_name all --dataset_choice all --CUDA cuda:0 --start_per 0.0 --end_per 0.25
python classification_train.py --model_name all --dataset_choice all --CUDA cuda:1 --start_per 0.25 --end_per 0.50
python classification_train.py --model_name all --dataset_choice all --CUDA cuda:2 --start_per 0.50 --end_per 0.75
python classification_train.py --model_name all --dataset_choice all --CUDA cuda:3 --start_per 0.75 --end_per 1.0


python AE_train.py --model_name FCN_AE --dataset_choice all --CUDA cuda:5 --start_per 0.0 --end_per 1.0


# Sanity check if the low acc is coming form high variance
python classification_train.py  --model_name FCN --dataset_choice all --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python classification_train.py  --model_name ResNet --dataset_choice all --CUDA cuda:1 --start_per 0.0 --end_per 1.0
python classification_train.py  --model_name InceptionTime --dataset_choice all --CUDA cuda:2 --start_per 0.0 --end_per 1.0

#05/27/2024
# repeatly train until reach 0.95 reported acc Sanity Check:
python classification_train.py --model_name MLP --dataset_choice uni --CUDA cuda:0 --start_per 0.0 --end_per 0.25

python classification_train.py --model_name all --dataset_choice all --CUDA cuda:0 --start_per 0.0 --end_per 0.2
python classification_train.py --model_name all --dataset_choice all --CUDA cuda:1 --start_per 0.2 --end_per 0.4
python classification_train.py --model_name all --dataset_choice all --CUDA cuda:2 --start_per 0.4 --end_per 0.6
python classification_train.py --model_name all --dataset_choice all --CUDA cuda:3 --start_per 0.6 --end_per 0.8
python classification_train.py --model_name all --dataset_choice all --CUDA cuda:4 --start_per 0.8 --end_per 1.0

# batch size are too large for some models retrain them. Some datasets are not giving good accs for 5 repeat, deal with it by prolong the training epochs maybe.

python classification_train.py --model_name MLP --dataset_choice all --CUDA cuda:4 --start_per 0.9 --end_per 1.0