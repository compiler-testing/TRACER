#!/usr/bin/env bash


process_name="./src/main.py --opt=fuzz  --path=/home/ty/$2/llvm-project-16 --sqlName=$1 --Mut=$3"
echo $process_name
start_time=$(date +%s)
end_time=$((start_time + 12*60*60))

python3 $process_name &
pid=$!

while true; do
    current_time=$(date +%s)
    if ps -p $pid > /dev/null; then
        echo "进程存在"
        if [ $current_time -ge $end_time ]; then
            echo "12小时已经过去，停止脚本执行。"
            kill -9 $pid
            exit
        fi
    else
        echo "进程不存在"
        python3 $process_name &
        pid=$!
    fi
    sleep 600
done


