from __future__ import annotations
from SoccerNet.Downloader import getListGames
from SoccerNet.Evaluation.utils import getMetaDataTask
import json
import os
import pandas as pd
from tqdm import tqdm
import random
import string
from dataclasses import dataclass
import cv2
import sys
import logging

file_handler = logging.FileHandler(filename='20230713_val_result.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=handlers
)

logger = logging.getLogger(__name__)

_VIDDEO_BASENAME = (
    "1_224p.mkv",
    "2_224p.mkv",
)

RANDOM_NAMES = set()


def get_max_len_seconds(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)                  # 動画を読み込む
    video_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # フレーム数を取得する
    video_fps = cap.get(cv2.CAP_PROP_FPS)                 # フレームレートを取得する
    video_len_sec = video_frame_count / video_fps         # 長さ（秒）を計算する

    return video_len_sec


def random_name(n):
    randlst = [random.choice(string.ascii_letters + string.digits)
               for i in range(n)]
    name = ''.join(randlst)
    if name in RANDOM_NAMES:
        return random_name(n)

    RANDOM_NAMES.add(name)

    return name


def extract_timeinfo(time_string: str):
    half = int(time_string[0])

    minutes, seconds = time_string.split(' ')[-1].split(':')
    minutes, seconds = int(minutes), int(seconds)

    return half, minutes, seconds


@dataclass(frozen=True)
class SingleData:
    captions: list[str]
    videoPath: str
    videoID: str
    spotTime: int | float


class Stage1:
    def __init__(self) -> None:
        path = '/raid_elmo/home/lr/moriy/SoccerNet/'
        split = "val"
        splits = [split]
        if split == 'val':
            splits = ['valid']
        self.path = path
        self.listGames = getListGames(splits, task="caption")
        labels, num_classes, dict_event, _ = getMetaDataTask(
            "caption", "SoccerNet", 2)
        self.labels = labels
        self.num_classes = num_classes
        self.dict_event = dict_event

    def run(self):
        res = []
        for game_i, game in tqdm(enumerate(self.listGames)):
            label_path = os.path.join(self.path, game, self.labels)
            game_path = os.path.join(self.path, game)
            labels = json.load(open(label_path))

            for annotation in labels["annotations"]:
                time = annotation["gameTime"]
                half, minutes, seconds = extract_timeinfo(time)

                event = annotation["label"]
                if event not in self.dict_event or half > 2:
                    continue

                caption = annotation['anonymized']
                captions = [
                    {"caption": caption}
                ]
                videoPath = os.path.join(game_path, _VIDDEO_BASENAME[half-1])
                videoID = random_name(10)
                spotTime = minutes * 60 + seconds
                res.append(SingleData(captions, videoPath, videoID, spotTime))
        return res


class Stage2:
    """
    """

    def __init__(self, data: list[SingleData]) -> None:
        split = 'val'
        self.data = data
        self.window_size = 20
        self.video_type = "mkv"
        self.dst_dir = '/raid_elmo/home/lr/moriy/SwinBERT/datasets/SoccerNet'
        self.video_dir = os.path.join(self.dst_dir, f"raw_videos/{split}")
        self.img_file = os.path.join(self.dst_dir, f"{split}.img.tsv")
        self.label_file = os.path.join(self.dst_dir, f"{split}.label.tsv")
        self.caption_file = os.path.join(self.dst_dir, f"{split}.caption.tsv")

        os.makedirs(self.video_dir, exist_ok=True)

    def run(self):
        import csv
        import multiprocessing
        from multiprocessing import Pool

        img_data = []
        label_data = []
        extract_video_args = []
        invalid_data = []
        for data in self.data:
            captions = data.captions
            videoID = data.videoID
            src_video_path = data.videoPath
            spotTime = data.spotTime

            dst_video_path = os.path.join(
                self.video_dir, videoID + f".{self.video_type}")
            img_data.append((dst_video_path, dst_video_path))
            label_data.append((dst_video_path, captions))
            extract_video_args.append(
                (src_video_path, dst_video_path, spotTime))

        with open(self.img_file, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(img_data)

        with open(self.label_file, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(label_data)

        # この記述がないと関数をpickel化できない
        global _video_clipping_worker

        def _video_clipping_worker(args):
            status, message = self.extract_video(*args)
            if not status:
                logger.warning(f"Failed to extract video: {repr(args)}")
                logger.warning(message)
                # args = (src_video_path, dst_video_path, spotTime)
                # return dst_video_path, src_video_path, spotTime, message
                return args[1], args[0], args[2], message
            return None

        with Pool(multiprocessing.cpu_count()) as p:
            results = list(tqdm(p.imap(_video_clipping_worker,
                           extract_video_args), total=len(extract_video_args)))

        invalid_data = [res for res in results if res is not None]

        split = 'val'
        invalid_data_file = os.path.join(self.dst_dir, f"invalid_data_{split}.tsv")
        invalid_data = pd.DataFrame(invalid_data)
        invalid_data.to_csv(invalid_data_file, index=False,
                            header=False, sep='\t')

    def extract_video(self,
                      src_video_path: str,
                      dst_video_path: str,
                      spotTime: int | float):
        import subprocess

        if os.path.exists(dst_video_path):
            return True, f"Video file already exists: {dst_video_path}"

        if not os.path.exists(src_video_path):
            return False, f"Video file does not exist: {src_video_path}"

        maxLenSeconds = get_max_len_seconds(src_video_path)

        start_time = max(0, spotTime - self.window_size / 2)
        end_time = min(maxLenSeconds, spotTime + self.window_size / 2)
        duration = max(0, min(self.window_size, end_time - start_time))

        if end_time - start_time <= 0:
            return False, f"Invalid time range, duration:{duration}, start_time:{start_time}, end_time:{end_time}"

        command = ['ffmpeg',
                   '-i', '"%s"' % src_video_path,
                   '-ss', str(start_time),
                   '-t', str(duration),
                   '-c:v', 'copy',
                   '-c:a', 'copy',
                   '-threads', "1",
                   '-loglevel', 'panic',
                   '"%s"' % dst_video_path]
        command = ' '.join(command)
        try:
            _ = subprocess.check_output(command, shell=True,
                                        stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            return False, repr(err.output)

        # Check if the video was successfully saved.
        status = os.path.exists(dst_video_path)
        return status, "Succeeded"


def main():
    stage1 = Stage1()
    data = stage1.run()
    stage2 = Stage2(data)
    stage2.run()


if __name__ == "__main__":
    main()
