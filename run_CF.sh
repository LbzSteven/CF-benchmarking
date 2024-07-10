#method="wCF"
#model="InceptionTime"
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Computers --CUDA cuda:6 --start_per 0.0 --end_per 1.0 &
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice ElectricDevices --CUDA cuda:6 --start_per 0.0 --end_per 1.0 &
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice UWaveGestureLibraryAll --CUDA cuda:6 --start_per 0.0 --end_per 1.0 &
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice PowerCons --CUDA cuda:6 --start_per 0.0 --end_per 1.0 &
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice FordA --CUDA cuda:6 --start_per 0.0 --end_per 1.0 &
#
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice HandOutlines --CUDA cuda:7 --start_per 0.0 --end_per 1.0 &
#model="MLP"
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice HandOutlines --CUDA cuda:7 --start_per 0.0 --end_per 1.0 &
#
#method="TSEvo"
#model="InceptionTime"
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice PowerCons --CUDA cuda:7 --start_per 0.0 --end_per 1.0 &


#model="FCN"
#method="TSEvo"
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Libras --CUDA cuda:6 --start_per 0.0 --end_per 1.0 &
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice UWaveGestureLibrary --CUDA cuda:6 --start_per 0.0 --end_per 1.0 &
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice PenDigits --CUDA cuda:6 --start_per 0.0 --end_per 1.0 &
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Phoneme --CUDA cuda:6 --start_per 0.0 --end_per 1.0 &
#
#model="MLP"
#method="TSEvo"
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Libras --CUDA cuda:4 --start_per 0.0 --end_per 1.0 &
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice UWaveGestureLibrary --CUDA cuda:4 --start_per 0.0 --end_per 1.0 &
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice PenDigits --CUDA cuda:4 --start_per 0.0 --end_per 1.0 &
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Phoneme --CUDA cuda:4 --start_per 0.0 --end_per 1.0 &
#
#model="InceptionTime"
#method="TSEvo"
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Libras --CUDA cuda:5 --start_per 0.0 --end_per 1.0 &
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice UWaveGestureLibrary --CUDA cuda:5 --start_per 0.0 --end_per 1.0 &
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice PenDigits --CUDA cuda:5 --start_per 0.0 --end_per 1.0 &
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Phoneme --CUDA cuda:5 --start_per 0.0 --end_per 1.0 &

model="FCN"
method="wCF"
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Libras --CUDA cuda:6 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice UWaveGestureLibrary --CUDA cuda:6 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice PenDigits --CUDA cuda:6 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Phoneme --CUDA cuda:6 --start_per 0.0 --end_per 1.0 &

model="MLP"
method="wCF"
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Libras --CUDA cuda:4 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice UWaveGestureLibrary --CUDA cuda:4 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice PenDigits --CUDA cuda:4 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Phoneme --CUDA cuda:4 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice EigenWorms --CUDA cuda:6 --start_per 0.0 --end_per 1.0 &

model="InceptionTime"
method="wCF"
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Libras --CUDA cuda:5 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice UWaveGestureLibrary --CUDA cuda:5 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice PenDigits --CUDA cuda:5 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Phoneme --CUDA cuda:5 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice EigenWorms --CUDA cuda:6 --start_per 0.0 --end_per 1.0 &



wait
echo "Finish"

#method="TSEvo"
#model="FCN"


#Libras
#AtrialFibrillation
#Phoneme