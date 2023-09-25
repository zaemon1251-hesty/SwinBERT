from dataclasses import dataclass
import os
import pandas as pd


@dataclass(frozen=True)
class Config:
    split = "val"
    base_dir: str = 'datasets/SoccerNet'
    img_path: str = os.path.join(base_dir, f'{split}.img.tsv')
    label_path: str = os.path.join(base_dir, f'{split}.label.tsv')
    invalid_data_path: str = os.path.join(base_dir, f'invalid_data_{split}.tsv')


def main(config: Config):

    img_df = pd.read_csv(config.img_path, sep='\t', header=None)
    label_df = pd.read_csv(config.label_path, sep='\t', header=None)
    invalid_data_df = pd.read_csv(config.invalid_data_path, sep='\t', header=None)

    img_df.columns = ['video_path', 'video_path_2']
    label_df.columns = ['video_path', 'captions']
    invalid_data_df.columns = ['video_path', 'src_video_path', 'spotTime', "message"]

    img_mask = (
        ~img_df['video_path']
        .isin(invalid_data_df['video_path'])
    )
    label_mask = (
        ~label_df['video_path']
        .isin(invalid_data_df['video_path'])
    )

    img_df = (
        img_df[img_mask]
        .reset_index(drop=True)
    )
    label_df = (
        label_df.loc[label_mask]
        .reset_index(drop=True)
    )

    img_df.to_csv(config.img_path, sep='\t', header=None, index=False)
    label_df.to_csv(config.label_path, sep='\t', header=None, index=False)


if __name__ == '__main__':
    # main()
    config = Config()
    main(config)
