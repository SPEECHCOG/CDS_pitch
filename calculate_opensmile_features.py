#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniil Kocharov (dan_ya)
"""
from argparse import ArgumentParser
from collections import defaultdict
import opensmile
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm


feature_levels = {'lowlevel': opensmile.FeatureLevel.LowLevelDescriptors,
                  'lldelta': opensmile.FeatureLevel.LowLevelDescriptors_Deltas,
                  'functional': opensmile.FeatureLevel.Functionals
                  }


def parse_args(parser):
    parser.add_argument("--data-dir", type=str, default=os.path.join('Corpora', 'Providence'),
                        help="Data directory.")
    return parser.parse_args()


def main():
    parser = ArgumentParser(description='Calculate openSMILE features', allow_abbrev=False)
    args = parse_args(parser)

    good_file_list_file = Path('providence_correct_files.csv')
    audio_dir = Path(args.data_dir, 'audio')
    dataset_dir = Path(args.data_dir, 'Providence_opensmile')

    assert audio_dir.is_dir(), f'No directory: {audio_dir}'
    os.makedirs(dataset_dir, exist_ok=True)

    good_files = read_file_list(good_file_list_file)

    for feature_level in feature_levels:
        print(feature_level)
        smile_processor = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=feature_levels[feature_level],
        )
        smile_data = defaultdict(list)
        for root, dirs, files in sorted(os.walk(audio_dir)):
            files = [file for file in files if file.endswith('.wav')]
            if len(files) == 0:
                continue
            print(f' - {root}')
            for file in tqdm(files):
                filename, _ = os.path.splitext(file)
                if filename not in good_files:
                    continue
                meta_info = filename.split('_')
                speaker_id = f"{meta_info[0]}_{meta_info[5]}"
                file = os.path.join(root, file)
                smile_data[speaker_id].append(smile_processor.process_file(file))
        for speaker_id in smile_data:
            print(speaker_id)
            print(' - concatenating DataFrames...')
            speaker_data = pd.concat(smile_data[speaker_id])
            print(' - sorting data...')
            speaker_data = speaker_data.sort_values('file')
            print(' - saving data to file...')
            file_path = Path(dataset_dir, f'{speaker_id}_{feature_level}.csv')
            speaker_data.to_csv(file_path, sep='\t')
    return True


def read_file_list(file):
    df = pd.read_csv(file, sep='\t', decimal=',')
    file_list = df['filename'].tolist()
    file_list = set(file_list)
    return file_list


main()
