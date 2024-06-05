python CF_generate.py  --CF_name NG --model_name FCN --dataset_choice uni --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name NG --model_name ResNet --dataset_choice uni --CUDA cuda:1 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name NG --model_name InceptionTime --dataset_choice uni --CUDA cuda:2 --start_per 0.0 --end_per 1.0
# 0604
# python CF_generate.py  --CF_name NG --model_name MLP --dataset_choice selected_uni --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name NUN_CF --model_name MLP --dataset_choice selected_uni --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name TSEvo --model_name MLP --dataset_choice selected_uni --CUDA cuda:7 --start_per 0.0 --end_per 0.5
python CF_generate.py  --CF_name TSEvo --model_name MLP --dataset_choice selected_uni --CUDA cuda:7 --start_per 0.5 --end_per 0.75
python CF_generate.py  --CF_name TSEvo --model_name MLP --dataset_choice selected_uni --CUDA cuda:7 --start_per 0.75 --end_per 1.0
python CF_generate.py  --CF_name SETS --model_name MLP --dataset_choice selected_uni --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name wCF --model_name MLP --dataset_choice selected_uni --CUDA cuda:6 --start_per 0.0 --end_per 0.5
python CF_generate.py  --CF_name wCF --model_name MLP --dataset_choice selected_uni --CUDA cuda:6 --start_per 0.5 --end_per 1.0

python CF_generate.py  --CF_name NG --model_name InceptionTime --dataset_choice selected_uni --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name NUN_CF --model_name InceptionTime --dataset_choice selected_uni --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name TSEvo --model_name InceptionTime --dataset_choice selected_uni --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name SETS --model_name InceptionTime --dataset_choice selected_uni --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name wCF --model_name InceptionTime --dataset_choice selected_uni --CUDA cuda:0 --start_per 0.0 --end_per 1.0

python CF_generate.py  --CF_name NG --model_name FCN --dataset_choice selected_uni --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name NUN_CF --model_name FCN --dataset_choice selected_uni --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name TSEvo --model_name FCN --dataset_choice selected_uni --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name SETS --model_name FCN --dataset_choice selected_uni --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name wCF --model_name FCN --dataset_choice selected_uni --CUDA cuda:0 --start_per 0.0 --end_per 1.0

python CF_generate.py  --CF_name COMTE --model_name MLP --dataset_choice selected_mul --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name NUN_CF --model_name MLP --dataset_choice selected_mul --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name TSEvo --model_name MLP --dataset_choice selected_mul --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name SETS --model_name MLP --dataset_choice selected_mul --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name wCF --model_name MLP --dataset_choice selected_mul --CUDA cuda:0 --start_per 0.0 --end_per 1.0

python CF_generate.py  --CF_name COMTE --model_name FCN --dataset_choice selected_mul --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name NUN_CF --model_name FCN --dataset_choice selected_mul --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name TSEvo --model_name FCN --dataset_choice selected_mul --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name SETS --model_name FCN --dataset_choice selected_mul --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name wCF --model_name FCN --dataset_choice selected_mul --CUDA cuda:0 --start_per 0.0 --end_per 1.0

python CF_generate.py  --CF_name COMTE --model_name InceptionTime --dataset_choice selected_mul --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name NUN_CF --model_name InceptionTime --dataset_choice selected_mul --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name TSEvo --model_name InceptionTime --dataset_choice selected_mul --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name SETS --model_name InceptionTime --dataset_choice selected_mul --CUDA cuda:0 --start_per 0.0 --end_per 1.0
python CF_generate.py  --CF_name wCF --model_name InceptionTime --dataset_choice selected_mul --CUDA cuda:0 --start_per 0.0 --end_per 1.0