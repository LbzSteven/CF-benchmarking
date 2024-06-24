#python CF_generate.py  --CF_name wCF --model_name MLP --dataset_choice selected_uni --CUDA cuda:1 --start_per 0.0 --end_per 0.1 &
#python CF_generate.py  --CF_name wCF --model_name MLP --dataset_choice selected_uni --CUDA cuda:1 --start_per 0.1 --end_per 0.2 &
#python CF_generate.py  --CF_name wCF --model_name MLP --dataset_choice selected_uni --CUDA cuda:2 --start_per 0.2 --end_per 0.3 &
#python CF_generate.py  --CF_name wCF --model_name MLP --dataset_choice selected_uni --CUDA cuda:2 --start_per 0.3 --end_per 0.4 &
#python CF_generate.py  --CF_name wCF --model_name MLP --dataset_choice selected_uni --CUDA cuda:3 --start_per 0.4 --end_per 0.5 &
#python CF_generate.py  --CF_name wCF --model_name MLP --dataset_choice selected_uni --CUDA cuda:3 --start_per 0.5 --end_per 0.6 &
#python CF_generate.py  --CF_name wCF --model_name MLP --dataset_choice selected_uni --CUDA cuda:5 --start_per 0.6 --end_per 0.7 &
#python CF_generate.py  --CF_name wCF --model_name MLP --dataset_choice selected_uni --CUDA cuda:5 --start_per 0.7 --end_per 0.8 &
#python CF_generate.py  --CF_name wCF --model_name MLP --dataset_choice selected_uni --CUDA cuda:6 --start_per 0.8 --end_per 0.9 &
#python CF_generate.py  --CF_name wCF --model_name MLP --dataset_choice selected_uni --CUDA cuda:6 --start_per 0.9 --end_per 1.0 &
#python CF_generate.py  --CF_name TSEvo --model_name FCN --dataset_choice selected_uni --CUDA cuda:4 --start_per 0.0 --end_per 1.0 &
#python CF_generate.py  --CF_name NUN_CF --model_name FCN --dataset_choice selected_uni --CUDA cuda:4 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name NUN_CF --model_name InceptionTime --dataset_choice selected_uni --CUDA cuda:4 --start_per 0.0 --end_per 0.25
python CF_generate.py  --CF_name NUN_CF --model_name InceptionTime --dataset_choice selected_uni --CUDA cuda:4 --start_per 0.25 --end_per 0.5
python CF_generate.py  --CF_name NUN_CF --model_name InceptionTime --dataset_choice selected_uni --CUDA cuda:4 --start_per 0.50 --end_per 0.75
python CF_generate.py  --CF_name NUN_CF --model_name InceptionTime --dataset_choice selected_uni --CUDA cuda:4 --start_per 0.75 --end_per 1.0

wait