python CF_generate.py  --CF_name SETS --model_name MLP --dataset_choice selected_uni --CUDA cuda:1 --start_per 0.0 --end_per 0.33 &
python CF_generate.py  --CF_name SETS --model_name MLP --dataset_choice selected_uni --CUDA cuda:1 --start_per 0.33 --end_per 0.67 &
python CF_generate.py  --CF_name SETS --model_name MLP --dataset_choice selected_uni --CUDA cuda:1 --start_per 0.67 --end_per 1.0 &
python CF_generate.py  --CF_name wCF --model_name MLP --dataset_choice selected_uni --CUDA cuda:2 --start_per 0.25 --end_per 0.50 &
python CF_generate.py  --CF_name wCF --model_name MLP --dataset_choice selected_uni --CUDA cuda:2 --start_per 0.75 --end_per 1.0 &

wait