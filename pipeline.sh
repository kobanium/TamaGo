for i in `seq 1 100` ; do
    python3.6 selfplay_main.py --save-dir archive --model model/rl-model.bin --use-gpu true
    python3.6 get_final_status.py
    python3.6 train.py --rl true --kifu-dir archive
done
