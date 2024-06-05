# initialize start and end per
start_per=0.0
end_per=0.1

# define a function to clean all process
cleanup() {
    echo "terminate all the python program"

    pkill -P $$
    wait
    echo "All the CF program has been terminated"
    exit 1
}

# catch interrupt signal (Ctrl+C)
trap cleanup SIGINT

# parallel execute 10 programs
for i in $(seq 1 10); do
    python CF_generate.py  --CF_name NUN_CF --model_name MLP --dataset_choice selected_uni --CUDA cuda:0 --start_per $start_per" --end_per "$end_per" &

    start_per=$(echo "$start_per + 0.1" | bc)
    end_per=$(echo "$end_per + 0.1" | bc)

done


wait

echo "All CF program is completed"