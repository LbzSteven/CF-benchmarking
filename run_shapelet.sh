# 定义第一个参数和第二个参数的初始值
param1=0.0
end_per=0.1

# define a function to clean all process
cleanup() {
    echo "terminate all the python program"

    pkill -P $$
    wait
    echo "All the shapelets program has been terminated"
    exit 1
}

# catch interrupt signal (Ctrl+C)
trap cleanup SIGINT

# parallel execute 10 programs
for i in $(seq 1 10); do
    python your_python_script.py --dataset_choice all --start_per "$start_per" --end_per "$end_per" &
    

    start_per=$(echo "$start_per + 0.1" | bc)
    end_per=$(echo "$end_per + 0.1" | bc)
    

    if (( $(jobs -r | wc -l) >= 10 )); then
        wait -n
    fi
done


wait

echo "All shapelets program is completed"