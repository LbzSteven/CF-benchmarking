python CF_generate.py  --CF_name NG --model_name FCN --dataset_choice uni --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name NG --model_name ResNet --dataset_choice uni --CUDA cuda:1 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name NG --model_name InceptionTime --dataset_choice uni --CUDA cuda:2 --start_per 0.0 --end_per 1.0