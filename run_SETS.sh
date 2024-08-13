#python CF_generate.py  --CF_name SETS --model_name MLP --dataset_choice selected_uni --CUDA cuda:6 --start_per 0.0 --end_per 0.2 &
#python CF_generate.py  --CF_name SETS --model_name MLP --dataset_choice selected_uni --CUDA cuda:6 --start_per 0.2 --end_per 0.4 &
#python CF_generate.py  --CF_name SETS --model_name MLP --dataset_choice selected_uni --CUDA cuda:6 --start_per 0.4 --end_per 0.6 &
#python CF_generate.py  --CF_name SETS --model_name MLP --dataset_choice selected_uni --CUDA cuda:6 --start_per 0.6 --end_per 0.8 &
#python CF_generate.py  --CF_name SETS --model_name MLP --dataset_choice selected_uni --CUDA cuda:6 --start_per 0.8 --end_per 1.0 &
#
#wait

#python CF_generate.py  --CF_name SETS --model_name FCN --dataset_choice selected_uni --CUDA cuda:6 --start_per 0.0 --end_per 0.2 &
#python CF_generate.py  --CF_name SETS --model_name FCN --dataset_choice selected_uni --CUDA cuda:6 --start_per 0.2 --end_per 0.4 &
#python CF_generate.py  --CF_name SETS --model_name FCN --dataset_choice selected_uni --CUDA cuda:6 --start_per 0.4 --end_per 0.6 &
#python CF_generate.py  --CF_name SETS --model_name FCN --dataset_choice selected_uni --CUDA cuda:6 --start_per 0.6 --end_per 0.8 &
#python CF_generate.py  --CF_name SETS --model_name FCN --dataset_choice selected_uni --CUDA cuda:6 --start_per 0.8 --end_per 1.0 &
#
#wait
#
#python CF_generate.py  --CF_name SETS --model_name InceptionTime --dataset_choice selected_uni --CUDA cuda:2 --start_per 0.0 --end_per 0.2 &
#python CF_generate.py  --CF_name SETS --model_name InceptionTime --dataset_choice selected_uni --CUDA cuda:2 --start_per 0.2 --end_per 0.4 &
#python CF_generate.py  --CF_name SETS --model_name InceptionTime --dataset_choice selected_uni --CUDA cuda:3 --start_per 0.4 --end_per 0.6 &
#python CF_generate.py  --CF_name SETS --model_name InceptionTime --dataset_choice selected_uni --CUDA cuda:3 --start_per 0.6 --end_per 0.8 &
#python CF_generate.py  --CF_name SETS --model_name InceptionTime --dataset_choice selected_uni --CUDA cuda:3 --start_per 0.8 --end_per 1.0 &
#
#wait

#python CF_generate.py  --CF_name SETS --model_name MLP --dataset_choice selected_mul --CUDA cuda:7 --start_per 0.0 --end_per 0.2 &
#python CF_generate.py  --CF_name SETS --model_name MLP --dataset_choice selected_mul --CUDA cuda:7 --start_per 0.2 --end_per 0.4 &
#python CF_generate.py  --CF_name SETS --model_name MLP --dataset_choice selected_mul --CUDA cuda:7 --start_per 0.4 --end_per 0.6 &
#python CF_generate.py  --CF_name SETS --model_name MLP --dataset_choice selected_mul --CUDA cuda:7 --start_per 0.6 --end_per 0.8 &
#python CF_generate.py  --CF_name SETS --model_name MLP --dataset_choice selected_mul --CUDA cuda:7 --start_per 0.8 --end_per 1.0 &
#
#wait

#python CF_generate.py  --CF_name SETS --model_name FCN --dataset_choice selected_mul --CUDA cuda:1 --start_per 0.0 --end_per 0.2 &
#python CF_generate.py  --CF_name SETS --model_name FCN --dataset_choice selected_mul --CUDA cuda:1 --start_per 0.2 --end_per 0.4 &
#python CF_generate.py  --CF_name SETS --model_name FCN --dataset_choice selected_mul --CUDA cuda:1 --start_per 0.4 --end_per 0.6 &
#python CF_generate.py  --CF_name SETS --model_name FCN --dataset_choice selected_mul --CUDA cuda:1 --start_per 0.6 --end_per 0.8 &
#python CF_generate.py  --CF_name SETS --model_name FCN --dataset_choice selected_mul --CUDA cuda:1 --start_per 0.8 --end_per 1.0 &
#
#python CF_generate.py  --CF_name SETS --model_name InceptionTime --dataset_choice selected_mul --CUDA cuda:4 --start_per 0.0 --end_per 0.2 &
#python CF_generate.py  --CF_name SETS --model_name InceptionTime --dataset_choice selected_mul --CUDA cuda:4 --start_per 0.2 --end_per 0.4 &
#python CF_generate.py  --CF_name SETS --model_name InceptionTime --dataset_choice selected_mul --CUDA cuda:5 --start_per 0.4 --end_per 0.6 &
#python CF_generate.py  --CF_name SETS --model_name InceptionTime --dataset_choice selected_mul --CUDA cuda:5 --start_per 0.6 --end_per 0.8 &
#python CF_generate.py  --CF_name SETS --model_name InceptionTime --dataset_choice selected_mul --CUDA cuda:5 --start_per 0.8 --end_per 1.0 &
#method="SETS"
#model="InceptionTime"
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Heartbeat --CUDA cuda:4 --start_per 0.0 --end_per 1.0 &
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice StandWalkJump --CUDA cuda:4 --start_per 0.0 --end_per 1.0 &
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice SelfRegulationSCP1 --CUDA cuda:4 --start_per 0.0 --end_per 1.0 &
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Cricket --CUDA cuda:4 --start_per 0.0 --end_per 1.0 &
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice BasicMotions --CUDA cuda:5 --start_per 0.0 --end_per 1.0 &
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice Epilepsy --CUDA cuda:5 --start_per 0.0 --end_per 1.0 &
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice NATOPS --CUDA cuda:5 --start_per 0.0 --end_per 1.0 &
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice RacketSports --CUDA cuda:5 --start_per 0.0 --end_per 1.0 &
#python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice EigenWorms --CUDA cuda:5 --start_per 0.0 --end_per 1.0 &
#
#
#wait
#echo "Finish"

#python shapelet_mining_SETS.py --dataset_choice UWaveGestureLibrary &
#python shapelet_mining_SETS.py --dataset_choice PenDigits &
#
#wait
#echo "Shapelets mined"
method='SETS'
model='FCN'
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice UWaveGestureLibrary --CUDA cuda:1 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice PenDigits --CUDA cuda:1 --start_per 0.0 --end_per 1.0 &
model='MLP'
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice UWaveGestureLibrary --CUDA cuda:1 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice PenDigits --CUDA cuda:1 --start_per 0.0 --end_per 1.0 &
model='InceptionTime'
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice UWaveGestureLibrary --CUDA cuda:5 --start_per 0.0 --end_per 1.0 &
python CF_generate.py  --CF_name "$method" --model_name "$model" --dataset_choice PenDigits --CUDA cuda:5 --start_per 0.0 --end_per 1.0 &

wait
echo "SETS over"