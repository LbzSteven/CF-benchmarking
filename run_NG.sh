#python CF_generate.py  --CF_name NG --model_name FCN --dataset_choice selected_uni --CUDA cuda:1 --start_per 0.0 --end_per 0.5 &
#python CF_generate.py  --CF_name NG --model_name InceptionTime --dataset_choice selected_uni --CUDA cuda:1 --start_per 0.0 --end_per 0.5 &
#
#python CF_generate.py  --CF_name NG --model_name FCN --dataset_choice selected_uni --CUDA cuda:2 --start_per 0.5 --end_per 1.0 &
#python CF_generate.py  --CF_name NG --model_name InceptionTime --dataset_choice selected_uni --CUDA cuda:2 --start_per 0.5 --end_per 1.0 &
python CF_generate.py  --CF_name NG --model_name FCN --dataset_choice multiclass_uni --CUDA cuda:1 --start_per 0.0 --end_per 0.5 &
python CF_generate.py  --CF_name NG --model_name FCN --dataset_choice multiclass_uni --CUDA cuda:1 --start_per 0.5 --end_per 1.0 &
#python CF_generate.py  --CF_name NG --model_name MLP --dataset_choice multiclass_uni --CUDA cuda:1 --start_per 0.0 --end_per 0.5 &
#python CF_generate.py  --CF_name NG --model_name MLP --dataset_choice multiclass_uni --CUDA cuda:1 --start_per 0.5 --end_per 1.0 &
python CF_generate.py  --CF_name NG --model_name InceptionTime --dataset_choice multiclass_uni --CUDA cuda:3 --start_per 0.0 --end_per 0.5 &
python CF_generate.py  --CF_name NG --model_name InceptionTime --dataset_choice multiclass_uni --CUDA cuda:3 --start_per 0.5 --end_per 1.0 &

python CF_generate.py  --CF_name NUN_CF --model_name FCN --dataset_choice multiclass_uni --CUDA cuda:4 --start_per 0.0 --end_per 0.5 &
python CF_generate.py  --CF_name NUN_CF --model_name FCN --dataset_choice multiclass_uni --CUDA cuda:4 --start_per 0.5 --end_per 1.0 &
python CF_generate.py  --CF_name NUN_CF --model_name MLP --dataset_choice multiclass_uni --CUDA cuda:6 --start_per 0.0 --end_per 0.5 &
python CF_generate.py  --CF_name NUN_CF --model_name MLP --dataset_choice multiclass_uni --CUDA cuda:6 --start_per 0.5 --end_per 1.0 &
python CF_generate.py  --CF_name NUN_CF --model_name InceptionTime --dataset_choice multiclass_uni --CUDA cuda:7 --start_per 0.0 --end_per 0.5 &
python CF_generate.py  --CF_name NUN_CF --model_name InceptionTime --dataset_choice multiclass_uni --CUDA cuda:7 --start_per 0.5 --end_per 1.0 &
wait