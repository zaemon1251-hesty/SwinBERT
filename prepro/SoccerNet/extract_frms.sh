python ./prepro/SoccerNet/extract_videos.py

python ./prepro/SoccerNet/select_valid_data.py


python ./prepro/extract_frames.py \
    --video_root_dir ./datasets/SoccerNet/raw_videos/val/ \
    --save_dir ./datasets/SoccerNet/ \
    --video_info_tsv ./datasets/SoccerNet/val.img.tsv \
    --num_frames 32

python ./prepro/create_image_frame_tsv.py \
    --dataset SoccerNet \
    --split val \
    --image_size 256 \
    --num_frames 32
