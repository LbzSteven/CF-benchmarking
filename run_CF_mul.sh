model="FCN"
method="SETS"
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Libras --CUDA cuda:1 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice UWaveGestureLibrary --CUDA cuda:1 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice PenDigits --CUDA cuda:1 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Phoneme --CUDA cuda:1 --start_per 0.0 --end_per 1.0 &

method="COMTE"
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Libras --CUDA cuda:1 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice UWaveGestureLibrary --CUDA cuda:1 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice PenDigits --CUDA cuda:2 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Phoneme --CUDA cuda:2 --start_per 0.0 --end_per 1.0 &

method="NUN_CF"
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Libras --CUDA cuda:2 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice UWaveGestureLibrary --CUDA cuda:2 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice PenDigits --CUDA cuda:2 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Phoneme --CUDA cuda:2 --start_per 0.0 --end_per 1.0 &

model="MLP"
method="SETS"
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Libras --CUDA cuda:3 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice UWaveGestureLibrary --CUDA cuda:3 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice PenDigits --CUDA cuda:3 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Phoneme --CUDA cuda:3 --start_per 0.0 --end_per 1.0 &

method="COMTE"
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Libras --CUDA cuda:3 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice UWaveGestureLibrary --CUDA cuda:3 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice PenDigits --CUDA cuda:7 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Phoneme --CUDA cuda:7 --start_per 0.0 --end_per 1.0 &

method="NUN_CF"
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Libras --CUDA cuda:7 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice UWaveGestureLibrary --CUDA cuda:7 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice PenDigits --CUDA cuda:7 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Phoneme --CUDA cuda:7 --start_per 0.0 --end_per 1.0 &

wait
echo "Finish"

model="InceptionTime"
method="SETS"
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Libras --CUDA cuda:1 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice UWaveGestureLibrary --CUDA cuda:1 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice PenDigits --CUDA cuda:1 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Phoneme --CUDA cuda:1 --start_per 0.0 --end_per 1.0 &

method="COMTE"
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Libras --CUDA cuda:2 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice UWaveGestureLibrary --CUDA cuda:2 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice PenDigits --CUDA cuda:2 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Phoneme --CUDA cuda:2 --start_per 0.0 --end_per 1.0 &

method="NUN_CF"
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Libras --CUDA cuda:3 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice UWaveGestureLibrary --CUDA cuda:3 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice PenDigits --CUDA cuda:3 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Phoneme --CUDA cuda:3 --start_per 0.0 --end_per 1.0 &

wait
echo "Finish InceptionTime"