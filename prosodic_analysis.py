#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniil Kocharov (dan_ya)
"""
from argparse import ArgumentParser
from collections import defaultdict
import copy
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from scipy.stats import spearmanr, sem
from scipy.signal import savgol_filter
from time import time
from tqdm import tqdm
from annotation_utils import TextGrid

vowel_symbols = {
                 # IPA (Montreal Forced Aligner)
                 'i', 'y', 'ɨ', 'ʉ', 'ɯ', 'u',
                 'ɪ', 'ʏ', 'ʊ',
                 'e', 'ø', 'ɘ', 'ɵ', 'ɤ', 'o',
                 'ə',
                 'ɛ', 'œ', 'ɜ', 'ɞ', 'ʌ', 'ɔ',
                 'æ', 'ɐ',
                 'a', 'ɶ', 'ɑ', 'ɒ',
                 # X-SAMPA: https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runMAUSGetInventar?LANGUAGE=eng-US
                 'A', 'a', 'E', 'e', 'I', 'i', 'O', 'o', 'U', 'u',
                 '3', 'V', 'Q', '6', '@', '{'
}

measurement_units = {
    'intonation_start': 'Hertz',
    'intonation_mean': 'semitones',
    'intonation_std': 'semitones',
    'intonation_max': 'semitones',
    'intonation_min': 'semitones',
    'intonation_declination': 'semitones',
    'intonation_accent': 'semitones',
    'pitch_mean': 'semitones',
    'pitch_std': 'semitones',
    'pitch_max': 'semitones',
    'pitch_min': 'semitones',
    'pitch_range': 'semitones',
    'pitch_fall': 'semitones',
    'pitch_rise': 'semitones',
    'prosodic_words': 'number of word per utterance',
    'speech_rate': 'number of sounds per second',
}


def parse_args(parser):
    # Main parameters
    parser.add_argument("--data-dir", type=str, default=os.path.join('Corpora', 'Providence'), help="Data directory.")
    parser.add_argument("--report-dir", type=str, default=os.path.join('report.12-42.lim.2'), help="Output directory.")
    parser.add_argument("--prosodic-features-file", type=str, default=os.path.join('prosodic_features.melody.txt'), help="Path to the file to save feature values.")
    parser.add_argument('--age-bounds-min', type=int, default=12, help="The lower bound for child age (in months).")
    parser.add_argument('--age-bounds-max', type=int, default=42, help="The lower bound for child age (in months). Set -1 - for no age limit.")
    parser.add_argument('--size-of-age-bin', type=int, default=3, help="The size of age bin (in months).")
    # Pipeline parameters
    parser.add_argument('--calculate-features', action='store_true', default=False, help="Calculate prosodic features.")
    parser.add_argument('--perform-statistical-analysis', action='store_true', default=False, help="Perform statistical analysis.")
    parser.add_argument('--plot-features', action='store_true', default=False, help="Plot figures.")
    # Feature analysis parameters
    parser.add_argument('--phone-tier', type=str, default='MAU', choices=['MAU', 'phones'], help="Name of the tier with phones.")
    parser.add_argument('--word-tier', type=str, default='ORT-MAU', choices=['ORT-MAU', 'words'], help="Name of the tier with words.")
    parser.add_argument('--relative-values', action='store_true', default=True, help="Calculate relative features (semitones) or linear (Hertz).")
    parser.add_argument('--stat-modes', nargs='+', default=['age', 'nword_age', 'speaker_age'], help="Modes of statistical analysis.")
    parser.add_argument('--feature-rate', type=int, default=100, help="The rate of feature value extraction.")
    parser.add_argument('--max-sound-length', type=float, default=0.3, help="Maximal duration of sound (in sec.) to be analyzed.")
    parser.add_argument('--min-voicing-part', type=float, default=0.6, help="Minimal duration of voicing segment (in sec.) to be taken into account.")
    parser.add_argument('--outlier-limit', type=float, default=2.5, help="The percentile to be removed to clean distributions.")
    parser.add_argument('--max-n-word-limit', type=int, default=7, help="Maximum number of words per utterance to be analyzed in 'nmord_age' mode.")
    # Plotting parameters
    parser.add_argument('--show-figures', action='store_true', default=False, help="Show figures.")
    parser.add_argument('--show-sem', action='store_true', default=True, help="Plot regions of standard error of mean on linear plots.")
    parser.add_argument('--paper-plots', action='store_true', default=True, help="Plots are prepared for the paper view.")
    parser.add_argument('--plot-distributions', action='store_true', default=False, help="Plot distributions.")
    parser.add_argument('--figure-file-ext', type=str, nargs='+',  default=['.png', '.pdf'], help="Types of files to be saved.")
    parser.add_argument('--violin_plot_with_quartiles', action='store_true', default=True, help="Plot distributions as violin plots.")
    parser.add_argument('--font_size', type=int, default=10, help="Size of the font in the plots.")

    return parser.parse_args()


arg_parser = ArgumentParser(description='Calculate openSMILE features', allow_abbrev=False)
args = parse_args(arg_parser)


def main():
    audio_dir = Path(args.data_dir, 'original dataset', 'audio')
    segmentation_dir = Path(args.data_dir, 'Providence_aligned')
    opensmile_dir = Path(args.data_dir, 'Providence_opensmile')
    stat_report_filename = Path(args.report_dir, 'statistical_report.melody')
    good_file_list_file = Path('providence_correct_files.csv')

    assert opensmile_dir.is_dir(), f'The given directory with audio files does not exist: {audio_dir}'
    assert opensmile_dir.is_dir(), f'The given directory with TextGrid segmentation does not exist: {segmentation_dir}'
    assert opensmile_dir.is_dir(), f'The given directory with prosodic features does not exist: {opensmile_dir}'
    os.makedirs(args.report_dir, exist_ok=True)

    if args.paper_plots:
        args.font_size = 12

    speech_data = None
    if args.calculate_features:
        print(f'\ncalculating features...')
        meta_info = load_meta_information(good_file_list_file)
        audio_files = load_filelist(audio_dir, '.wav')
        segmentation_data = load_segmentation(segmentation_dir)

        speech_data = defaultdict(dict)
        for filename in tqdm(sorted(audio_files)):
            if filename not in segmentation_data:
                continue
            filename_info = filename.split('_')
            speaker_id = f"{filename_info[0]}_{filename_info[5]}"
            speech_data[speaker_id][filename] = {'audio_file': audio_files[filename],
                                                 'segmentation': segmentation_data[filename],
                                                 'age': meta_info[filename]['child_age'],
                                                 'duration': meta_info[filename]['duration'],
                                                 'analysis': {}
                                                 }
        timestamp = time()
        for speaker_id in speech_data:
            print(f'\n{speaker_id}: loading opensmile data...')
            opensmile_datafile = os.path.join(opensmile_dir, f"{speaker_id}.pkl")
            if os.path.isfile(opensmile_datafile):
                with open(opensmile_datafile, 'rb') as file:
                    opensmile_speaker_data = pickle.load(file)
            else:
                opensmile_speaker_data = load_opensmile_data(speaker_id, speech_data[speaker_id], opensmile_dir)
            print(f'\n{speaker_id}: calculating intonation measures...')
            speech_data[speaker_id] = estimate_intonation(speech_data[speaker_id], opensmile_speaker_data, main_tier_name=args.word_tier, sub_tier_name=args.phone_tier)

            print(f'\n{speaker_id}: calculating pitch measures...')
            speech_data[speaker_id] = estimate_pitch(speech_data[speaker_id], opensmile_speaker_data, tier_name=args.phone_tier)
            speech_data = calculate_rise_ratio(speech_data)

            print(f'\n{speaker_id}: calculating speaker rate...')
            speech_data[speaker_id] = estimate_speech_rate(speech_data[speaker_id], tier_name=args.phone_tier)

            processing_time = time() - timestamp
            timestamp = time()
            ts_minutes, ts_seconds = divmod(processing_time, 60)
            ts_hours, ts_minutes = divmod(ts_minutes, 60)
            print(f"\n{speaker_id} proccesing time (h:min:sec) is {round(ts_hours)}:{round(ts_minutes)}:{int(ts_seconds)}")
        # clean and save the data
        for speaker_id in speech_data:
            for filename in speech_data[speaker_id]:
                del speech_data[speaker_id][filename]['segmentation']
        report = json.dumps(speech_data, ensure_ascii=False, indent=4, sort_keys=True)
        with open(args.prosodic_features_file, 'w', encoding='utf-8') as fo:
            fo.write(report)

    # load speech data, if necessary
    print(f'\nloading features...')
    if speech_data is None and os.path.isfile(args.prosodic_features_file):
        with open(args.prosodic_features_file, 'r') as fi:
            speech_data = json.load(fi)

    # reshape data in convenient way to analyze dependencies on age, speaker and number of words in the utterance.
    age_data = defaultdict(lambda: defaultdict(list))
    for speaker_id in speech_data:
        for filename in speech_data[speaker_id]:
            if speech_data[speaker_id][filename]['analysis']['prosodic_words'] in [None, 0]:
                continue
            for feature in speech_data[speaker_id][filename]['analysis']:
                if speech_data[speaker_id][filename]['analysis'][feature] is not None:
                    age_bin = args.size_of_age_bin * (speech_data[speaker_id][filename]['age'] // args.size_of_age_bin)
                    if args.age_limits_max != -1 and not (args.age_limits_min <= age_bin <= args.age_limits_max):
                        continue
                    age_data[feature][age_bin].append(speech_data[speaker_id][filename]['analysis'][feature])
    speaker_age_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for speaker_id in speech_data:
        for filename in speech_data[speaker_id]:
            if speech_data[speaker_id][filename]['analysis']['prosodic_words'] in [None, 0]:
                continue
            for feature in speech_data[speaker_id][filename]['analysis']:
                if speech_data[speaker_id][filename]['analysis'][feature] is not None:
                    age_bin = args.size_of_age_bin * (speech_data[speaker_id][filename]['age'] // args.size_of_age_bin)
                    if args.age_limits_max != -1 and not (args.age_limits_min <= age_bin <= args.age_limits_max):
                        continue
                    speaker_age_data[feature][speaker_id][age_bin].append(speech_data[speaker_id][filename]['analysis'][feature])
    nword_age_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for speaker_id in speech_data:
        for filename in speech_data[speaker_id]:
            if speech_data[speaker_id][filename]['analysis']['prosodic_words'] in [None, 0]:
                continue
            for feature in speech_data[speaker_id][filename]['analysis']:
                if speech_data[speaker_id][filename]['analysis'][feature] is not None:
                    age_bin = args.size_of_age_bin * (speech_data[speaker_id][filename]['age'] // args.size_of_age_bin)
                    nword_bin = speech_data[speaker_id][filename]['analysis']['prosodic_words']
                    if nword_bin > args.max_n_word_limit:
                        continue
                    if args.age_limits_max != -1 and not (args.age_limits_min <= age_bin <= args.age_limits_max):
                        continue
                    nword_age_data[feature][nword_bin][age_bin].append(speech_data[speaker_id][filename]['analysis'][feature])

    if args.plot_features:
        print(f'\nplotting features...')
        for stat_mode in args.stat_modes:
            if stat_mode == 'age':
                for feature in age_data:
                    local_figure_dir = os.path.join(args.report_dir, f'figures_{stat_mode}')
                    plot_analytics_with_lines(age_data[feature], label=feature, figure_path=local_figure_dir, file_types=args.figure_file_ext)
            elif stat_mode == 'nword_age':
                for feature in nword_age_data:
                    for nword_bin in nword_age_data[feature]:
                        if nword_bin == 1:
                            continue
                        local_figure_dir = os.path.join(args.report_dir, f'figures_{stat_mode}')
                        plot_label = f'{feature}_{nword_bin}'
                        plot_analytics_with_lines(nword_age_data[feature][nword_bin], label=plot_label, figure_path=local_figure_dir, file_types=args.figure_file_ext)
            elif stat_mode == 'speaker_age':
                for feature in speaker_age_data:
                    local_figure_dir = os.path.join(args.report_dir, f'figures_{stat_mode}')
                    plot_analytics_with_lines_by_speaker(speaker_age_data[feature], label=feature, figure_path=local_figure_dir, file_types=args.figure_file_ext)
                    if args.plot_distributions:
                        for speaker_id in speaker_age_data[feature]:
                            local_figure_dir = os.path.join(args.report_dir, f'figures_{stat_mode}', speaker_id)
                            plot_label = f'{speaker_id}_{feature}'
                            plot_analytics_with_violin_plots(speaker_age_data[feature][speaker_id], label=plot_label, figure_path=local_figure_dir, file_types=args.figure_file_ext)

    if args.perform_statistical_analysis:
        print(f'\nstatistical analysis...')
        for stat_mode in args.stat_modes:
            statistical_report = []
            if stat_mode == 'nword_age':
                for feature in nword_age_data:
                    for nword_bin in nword_age_data[feature]:
                        if nword_bin == 1:
                            continue
                        r_values = sorted(nword_age_data[feature][nword_bin].keys())
                        m_values = [get_mean_value(nword_age_data[feature][nword_bin][r]) for r in r_values]
                        r, p = spearmanr(r_values, m_values)
                        if p < 0.05:
                            unit = [feature, nword_bin, round(r, 2), round(p, 3)]
                            statistical_report.append(unit)
            elif stat_mode == 'speaker_age':
                for feature in speaker_age_data:
                    for speaker_id in speaker_age_data[feature]:
                        r_values = sorted(speaker_age_data[feature][speaker_id].keys())
                        m_values = [get_mean_value(speaker_age_data[feature][speaker_id][r]) for r in r_values]
                        r, p = spearmanr(r_values, m_values)
                        if p < 0.05:
                            unit = [feature, speaker_id, round(r, 2), round(p, 3)]
                            statistical_report.append(unit)
            elif stat_mode == 'age':
                for feature in age_data:
                    r_values = sorted(age_data[feature].keys())
                    m_values = [get_mean_value(age_data[feature][r]) for r in r_values]
                    r, p = spearmanr(r_values, m_values)
                    if p < 0.05:
                        unit = [feature, round(r, 2), round(p, 3)]
                        statistical_report.append(unit)
            stat_report_file = f'{stat_report_filename}_{stat_mode}.txt'
            with open(stat_report_file, 'w', encoding='utf-8') as fo:
                for unit in sorted(statistical_report):
                    line = '\t'.join([str(v) for v in unit]) + '\n'
                    fo.write(line)
    return None


def calculate_rise_ratio(speech_data):
    accent_data = defaultdict(lambda: defaultdict(list))
    for speaker_id in speech_data:
        for filename in speech_data[speaker_id]:
            if speech_data[speaker_id][filename]['analysis']['intonation_accent'] is not None:
                age_bin = args.size_of_age_bin * (speech_data[speaker_id][filename]['age'] // args.size_of_age_bin)
                accent_data[speaker_id][age_bin].append(speech_data[speaker_id][filename]['analysis']['intonation_accent'])
    for speaker_id in speech_data:
        for filename in speech_data[speaker_id]:
            speech_data[speaker_id][filename]['analysis']['intonation_rise_ratio'] = None
            age_bin = args.size_of_age_bin * (speech_data[speaker_id][filename]['age'] // args.size_of_age_bin)
            if age_bin in accent_data[speaker_id] and len(accent_data[speaker_id][age_bin]) > 0:
                rises = [v for v in accent_data[speaker_id][age_bin] if not v < 0]
                rise_ratio = len(rises) / len(accent_data[speaker_id][age_bin])
                speech_data[speaker_id][filename]['analysis']['intonation_rise_ratio'] = rise_ratio
    return speech_data


def estimate_intonation(speech_data, prosodic_data, main_tier_name='words', sub_tier_name='phones'):
    opensmile_feature_name = 'F0final_sma'
    speaker_stat_data = get_speaker_general_distribution(speech_data, prosodic_data, feature=opensmile_feature_name, tier_name=sub_tier_name)
    for filename in sorted(speech_data):
        raw_values = None
        if filename in prosodic_data:
            raw_values = prosodic_data[filename][opensmile_feature_name].to_numpy(dtype=float)
            raw_values = smooth_melodic_data(raw_values)

        if main_tier_name not in speech_data[filename]['segmentation']:
            continue
        if sub_tier_name not in speech_data[filename]['segmentation']:
            continue
        word_tier = speech_data[filename]['segmentation'][main_tier_name]
        sound_tier = speech_data[filename]['segmentation'][sub_tier_name]
        word_tier = [w for w in word_tier if w.mark not in {'', '<p:>'}]
        vowel_tier = [s for s in sound_tier if is_vowel(s.mark)]
        for word in word_tier:
            word_values = []
            vowels = [v for v in vowel_tier if word.overlaps(v)]
            for vowel in vowels:
                vowel_values = []
                if raw_values is not None:
                    vowel_values = raw_values[int(args.feature_rate*vowel.minTime):int(args.feature_rate*vowel.maxTime)]
                    vowel_values = [v for v in vowel_values if speaker_stat_data['limits'][0] < v < speaker_stat_data['limits'][1]]
                if is_valid_sound(vowel, vowel_values):
                    word_values.append(copy.deepcopy(vowel_values))
            word.max_pitch = None
            if len(word_values) > 0:
                word.max_pitch = np.max(np.concatenate(word_values))
        file_values = [w.max_pitch for w in word_tier if w.max_pitch is not None]

        speaker_ref_value = None
        if len(file_values) > 0:
            speaker_ref_value = file_values[0]

        speech_data[filename]['analysis']['intonation_start'] = calculate_first_value(file_values)
        speech_data[filename]['analysis']['intonation_accent'] = calculate_last_accent(file_values, to_semitones)
        speech_data[filename]['analysis']['intonation_mean'] = calculate_mean(file_values, speaker_ref_value, to_semitones)
        speech_data[filename]['analysis']['intonation_std'] = calculate_std(file_values, speaker_ref_value, to_semitones)
        speech_data[filename]['analysis']['intonation_max'] = calculate_max(file_values, speaker_ref_value, to_semitones)
        speech_data[filename]['analysis']['intonation_min'] = calculate_min(file_values, speaker_ref_value, to_semitones)
        speech_data[filename]['analysis']['intonation_declination'] = calculate_declination(file_values, to_semitones)
        speech_data[filename]['analysis']['prosodic_words'] = len(file_values)
        speech_data[filename]['analysis']['n_words'] = len(word_tier)
    return speech_data


def estimate_pitch(speech_data, prosodic_data, tier_name='phones'):
    opensmile_feature_name = 'F0final_sma'
    speaker_stat_data = get_speaker_general_distribution(speech_data, prosodic_data, feature=opensmile_feature_name, tier_name=tier_name)
    for filename in sorted(speech_data):
        raw_values = None
        if filename in prosodic_data:
            raw_values = prosodic_data[filename][opensmile_feature_name].to_numpy(dtype=float)
            raw_values = smooth_melodic_data(raw_values)

        if tier_name not in speech_data[filename]['segmentation']:
            continue
        tier_data = speech_data[filename]['segmentation'][tier_name]
        vowels = [v for v in tier_data if is_vowel(v.mark)]

        file_values = []
        for vowel in vowels:
            vowel_values = []
            if raw_values is not None:
                vowel_values = raw_values[int(args.feature_rate*vowel.minTime):int(args.feature_rate*vowel.maxTime)]
                vowel_values = [v for v in vowel_values if speaker_stat_data['limits'][0] < v < speaker_stat_data['limits'][1]]
            if is_valid_sound(vowel, vowel_values):
                file_values.append(copy.deepcopy(vowel_values))
        if len(file_values) > 0:
            file_values = np.concatenate(file_values)

        file_mean_value = calculate_mean(file_values)
        speech_data[filename]['analysis']['pitch_mean'] = calculate_mean(file_values)
        if speech_data[filename]['analysis']['pitch_mean'] is not None:
            speech_data[filename]['analysis']['pitch_mean'] = round(speech_data[filename]['analysis']['pitch_mean'])
        speech_data[filename]['analysis']['pitch_std'] = calculate_std(file_values, file_mean_value, to_semitones)
        speech_data[filename]['analysis']['pitch_max'] = calculate_max(file_values, file_mean_value, to_semitones)
        speech_data[filename]['analysis']['pitch_min'] = calculate_min(file_values, file_mean_value, to_semitones)
        speech_data[filename]['analysis']['pitch_range'] = calculate_range(file_values, to_semitones)
        rises, falls = get_rises_falls(file_values)
        speech_data[filename]['analysis']['pitch_fall'] = get_max_range(falls)
        speech_data[filename]['analysis']['pitch_rise'] = get_max_range(rises)
    if args.relative_values and speaker_stat_data is not None:
        for filename in speech_data:
            speech_data[filename]['analysis']['speaker_pitch_mean'] = speaker_stat_data['mean']
            file_mean_value = to_semitones(speech_data[filename]['analysis']['pitch_mean'], speaker_stat_data['mean'])
            if file_mean_value is not None:
                file_mean_value = round(file_mean_value)
            speech_data[filename]['analysis']['pitch_mean'] = file_mean_value
    return speech_data


def estimate_speech_rate(speech_data, tier_name):
    for filename in tqdm(sorted(speech_data)):
        if tier_name not in speech_data[filename]['segmentation']:
            continue
        tier_data = speech_data[filename]['segmentation'][tier_name]
        sounds = [s for s in tier_data if s.mark not in {'', '<p:>'}]
        mean_duration = np.mean([i.duration() for i in sounds])
        if mean_duration != 0:
            speech_data[filename]['analysis']['speech_rate'] = round(1/mean_duration, 2)
        else:
            speech_data[filename]['analysis']['speech_rate'] = None
    return speech_data


def calculate_declination(values, ref_function):
    if len(values) == 0:
        return None
    if args.relative_values:
        result = ref_function(values[-1], values[0])
        if result is not None:
            result = round(result)
    else:
        result = values[-1] - values[0]
    return result


def calculate_first_value(values, ref_value=None, ref_function=None):
    if len(values) == 0:
        return None
    if args.relative_values and ref_value is not None:
        result = ref_function(values[0], ref_value)
    else:
        result = values[0]
    return result


def calculate_last_accent(values, ref_function):
    if len(values) == 0:
        return None
    if len(values) == 1:
        return 0
    if args.relative_values:
        result = ref_function(values[-1], values[-2])
        if result is not None:
            result = round(result)
    else:
        result = values[-2] - values[-1]
    return result


def calculate_max(values, ref_value, ref_function):
    if len(values) == 0:
        return None
    result = np.max(values)
    if args.relative_values:
        result = ref_function(result, ref_value)
        if result is not None:
            result = round(result)
    return result


def calculate_mean(values, ref_value=None, ref_function=None):
    if len(values) == 0:
        return None
    if args.relative_values and ref_value is not None:
        result = ref_function(np.mean(values), ref_value)
    else:
        result = np.mean(values)
    return result


def calculate_min(values, ref_value, ref_function):
    if len(values) == 0:
        return None
    result = np.min(values)
    if args.relative_values:
        result = ref_function(result, ref_value)
        if result is not None:
            result = round(result)
    return result


def calculate_range(values, ref_function):
    if len(values) == 0:
        return None
    if args.relative_values:
        result = ref_function(np.max(values), np.min(values))
        if result is not None:
            result = round(result)
    else:
        result = np.max(values) - np.min(values)
    return result


def calculate_std(values, ref_value, ref_function):
    if len(values) == 0:
        return None
    if args.relative_values:
        result = np.round(np.std([abs(ref_function(v, ref_value)) for v in values]))
        if result is not None:
            result = round(result)
    else:
        result = np.std(values)
    return result


def is_vowel(label):
    for symbol in vowel_symbols:
        if symbol in label:
            return True
    return False


def is_valid_sound(sound, prosodic_data):
    min_feature_frames = args.min_voicing_part * sound.duration() * args.feature_rate
    if sound.duration() > args.max_sound_length:
        return False
    if len(prosodic_data) < min_feature_frames:
        return False
    return True


def get_max_range(arr):
    if len(arr) == 0:
        return None
    argmax_range = 0
    max_range = abs(get_melodic_movement_amplitude(arr[argmax_range]))
    for i, elem in enumerate(arr):
        if len(elem) == 0:
            continue
        current_range = abs(get_melodic_movement_amplitude(elem))
        if max_range is None or max_range < current_range:
            argmax_range = i
    result = get_melodic_movement_amplitude(arr[argmax_range])
    if result is not None:
        result = abs(round(result))
    return result


def get_melodic_movement_amplitude(arr):
    if len(arr) == 0:
        return None
    if args.relative_values:
        result = to_semitones(arr[-1], arr[0])
    else:
        result = abs(arr[-1] - arr[0])
    return result


def get_rises_falls(arr):
    rises = []
    falls = []
    if len(arr) < 2:
        return rises, falls
    current_rise = []
    current_fall = []
    current_trend = 0

    i = 1
    while i < len(arr):
        if arr[i] > arr[i - 1]:
            current_rise.append(arr[i-1])
            current_trend = 1
            if len(current_fall) > 0:
                current_fall.append(arr[i-1])
                falls.append(copy.deepcopy(current_fall))
                current_fall = []
        elif arr[i] < arr[i - 1]:
            current_fall.append(arr[i-1])
            current_trend = -1
            if len(current_rise) > 0:
                current_rise.append(arr[i-1])
                rises.append(copy.deepcopy(current_rise))
                current_rise = []
        else:
            if current_trend == 1:
                current_rise.append(arr[i-1])
            else:
                current_fall.append(arr[i-1])
        i += 1
    # assign the last value to the final rise or final fall
    if arr[-2] < arr[-1]:
        current_rise.append(arr[-1])
    elif arr[-2] > arr[-1]:
        current_fall.append(arr[-1])
    elif arr[-2] == arr[-1] and len(current_rise) > 0:
        current_rise.append(arr[-1])
    elif arr[-2] == arr[-1] and len(current_fall) > 0:
        current_fall.append(arr[-1])

    # put the final melodic movement to the list of rises or falls
    if len(current_rise) > 0:
        rises.append(copy.deepcopy(current_rise))
    elif len(current_fall) > 0:
        falls.append(copy.deepcopy(current_fall))
    return rises, falls


def get_speaker_general_distribution(speech_data, prosodic_data, feature, tier_name='phones'):
    speaker_values = []
    for filename in sorted(speech_data):
        if filename not in prosodic_data:
            continue
        if tier_name not in speech_data[filename]['segmentation']:
            continue
        raw_values = prosodic_data[filename][feature].to_numpy(dtype=float)
        tier_data = speech_data[filename]['segmentation'][tier_name]
        vowels = [v for v in tier_data if is_vowel(v.mark)]
        for vowel in vowels:
            vowel_values = raw_values[int(args.feature_rate * vowel.minTime):int(args.feature_rate * vowel.maxTime)]
            vowel_values = vowel_values[vowel_values != 0]
            if is_valid_sound(vowel, vowel_values):
                speaker_values.append(copy.deepcopy(vowel_values))
    if len(speaker_values) == 0:
        return None
    speaker_values = np.concatenate(speaker_values)
    speaker_values = speaker_values[speaker_values != 0]
    speaker_stat_data = {'mean': np.mean(speaker_values),
                         'limits': np.percentile(speaker_values, [args.outlier_limit, 100 - args.outlier_limit])
                         }
    return speaker_stat_data


def load_filelist(directory: Path, extension: str):
    filelist = {}
    for root, dirs, files in os.walk(directory):
        files = [Path(file) for file in files if file.endswith(extension)]
        for file in files:
            file_path = os.path.join(root, file.stem)
            filelist[file.stem] = file_path
    return filelist


def load_meta_information(file: Path):
    df = pd.read_csv(file, sep='\t', decimal=',')
    meta_info = df.to_dict(orient='records')
    meta_info = {v['filename']: v for v in meta_info}
    return meta_info


def load_opensmile_data(speaker_id, speech_data, opensmile_dir):
    data = dict()
    file = Path(opensmile_dir, f"{speaker_id}_lowlevel.csv")
    if not file.is_file():
        return None
    else:
        data = dict()
        loaded_data = pd.read_csv(file, sep='\t', decimal=',')
        for filename in tqdm(speech_data):
            condition = loaded_data['file'] == speech_data[filename]['audio_file']
            data[filename] = loaded_data[condition]
    return data


def load_segmentation(directory: Path):
    print(f'loading segmentation...')
    data = {}
    for root, dirs, files in sorted(os.walk(directory)):
        files = [Path(root, file) for file in files if file.endswith('.TextGrid')]
        for file in tqdm(files):
            tg = TextGrid.read_file(file)
            data[file.stem] = tg
    return data


def plot_analytics_with_lines(data, label, figure_path, file_types=None, show_data_size=True):
    if not os.path.isdir(figure_path):
        os.makedirs(figure_path)
    x_labels = sorted(list(data.keys()))
    x_ticks = list(range(len(x_labels)))
    values = [get_mean_value(data[k]) if k in data else None for k in x_labels]
    sems = [sem(data[k], nan_policy='omit') if k in data else None for k in x_labels]
    correct_values = [len(data[k]) if k in data else 0 for k in x_labels]
    plt.plot(values, linewidth=2, color='black')
    alpha_value = 0.1
    if args.paper_plots:
        alpha_value = 0.05
    if args.show_sem:
        plt.fill_between(x=list(range(len(values))),
                         y1=[values[i] - sems[i] if values[i] is not None else None for i in range(len(values))],
                         y2=[values[i] + sems[i] if values[i] is not None else None for i in range(len(values))],
                         color='black', alpha=alpha_value)
    if show_data_size and not args.paper_plots:
        for j in range(len(x_ticks)):
            if values[j] is not None:
                plt.text(x_ticks[j], values[j], correct_values[j])
    if not args.paper_plots:
        plt.title(f'{label}', fontweight='bold')
    plt.xticks(x_ticks, x_labels, rotation='vertical', fontsize=args.font_size)
    plt.yticks(fontsize=args.font_size)
    plt.xlabel('age (months)', fontsize=args.font_size)
    if args.show_figures:
        plt.show()
    figure_file = os.path.join(figure_path, f'{label}')
    if file_types is not None:
        if args.paper_plots and '.pdf' not in file_types:
            plt.savefig(figure_file + '.pdf', bbox_inches='tight')
        for ext in file_types:
            plt.savefig(figure_file + ext, bbox_inches='tight')
    plt.close()


def plot_analytics_with_lines_by_speaker(data, label, figure_path, file_types=None, show_data_size=True):
    if not os.path.isdir(figure_path):
        os.makedirs(figure_path)
    colours = ['black', 'orange', 'red', 'magenta', 'blue', 'green']
    speaker_labels = sorted(data)
    x_labels = []
    for speaker_id in data:
        x_labels += list(data[speaker_id].keys())
    x_labels = sorted(list(set(x_labels)))
    x_ticks = list(range(len(x_labels)))
    for i, speaker_id in enumerate(speaker_labels):
        values = [get_mean_value(data[speaker_id][k]) if k in data[speaker_id] else None for k in x_labels]
        x_values = [i for i in range(len(values)) if values[i] is not None]
        sems = [sem(data[speaker_id][k], nan_policy='omit') if k in data[speaker_id] else None for k in x_labels]
        correct_values = [len(data[speaker_id][k]) if k in data[speaker_id] else 0 for k in x_labels]
        speaker_id = speaker_id.replace('_MOT', '')
        if args.paper_plots:
            plt.plot(values, linewidth=2, color=colours[i], label=f'{speaker_id}')
        else:
            plt.plot(values, linewidth=2, color=colours[i], label=f'{speaker_id} ({sum(correct_values)})')
        alpha_value = 0.1
        if args.paper_plots:
            alpha_value = 0.05
        if args.show_sem:
            plt.fill_between(x=x_values,
                             y1=[values[i]-sems[i] for i in x_values],
                             y2=[values[i]+sems[i] for i in x_values],
                             color=colours[i], alpha=alpha_value)
        if show_data_size and not args.paper_plots:
            for j in range(len(x_ticks)):
                if values[j] is not None:
                    plt.text(x_ticks[j], values[j], correct_values[j], color=colours[i])
        if not args.paper_plots:
            plt.title(f'{label}', fontweight='bold')
        plt.xticks(x_ticks, x_labels, rotation='vertical', fontsize=args.font_size)
    plt.xlabel('age (months)', fontsize=args.font_size)
    plt.yticks(fontsize=args.font_size)
    plt.legend(fontsize=args.font_size)
    if args.show_figures:
        plt.show()
    figure_file = os.path.join(figure_path, f'{label}')
    if file_types is not None:
        if args.paper_plots and '.pdf' not in file_types:
            plt.savefig(figure_file + '.pdf', bbox_inches='tight')
        for ext in file_types:
            plt.savefig(figure_file+ext, bbox_inches='tight')
    plt.close()


def plot_analytics_with_violin_plots(data, label, figure_path, file_types=None):
    if not os.path.isdir(figure_path):
        os.makedirs(figure_path)
    labels = sorted(data)
    if args.violin_plot_with_quartiles:
        data = copy.deepcopy(data)
        for k in labels:
            cutoff = 20
            limits = np.percentile(data[k], [cutoff, 100-cutoff])
            data[k] = [v for v in data[k] if limits[0] <= v <= limits[1]]
    plt.violinplot([data[k] for k in labels], showmeans=True)
    plt.title(f'{label}', fontweight='bold')
    x_ticks = [x+1 for x in range(len(labels))]
    x_labels = [k for k in labels]
    plt.xticks(x_ticks, x_labels, rotation='vertical')
    plt.xlabel('age')
    plt.grid(True)
    if args.show_figures:
        plt.show()
    figure_file = os.path.join(figure_path, f'{label}')
    if file_types is not None:
        for ext in file_types:
            plt.savefig(figure_file+ext, bbox_inches='tight')
    plt.close()


def get_mean_value(data):
    limits = np.percentile(data, [args.outlier_limit, 100 - args.outlier_limit])
    data = [v for v in data if limits[0] <= v <= limits[1]]
    result = np.mean(data)
    return result


def smooth_melodic_data(melodic_data):
    non_zero_indices = np.nonzero(melodic_data)[0]
    if len(non_zero_indices) == 0:
        return melodic_data
    zeros_prefix = np.linspace(0, non_zero_indices[0], num=non_zero_indices[0]+1, dtype=int)[:-1]
    zeros_suffix = np.linspace(non_zero_indices[-1], len(melodic_data)-1, num=len(melodic_data) - non_zero_indices[-1], dtype=int)[1:]
    zero_indices = np.concatenate((zeros_prefix, zeros_suffix))
    interpolated_values = np.interp(np.arange(len(melodic_data)), non_zero_indices, melodic_data[non_zero_indices])
    smoothed_values = savgol_filter(interpolated_values, window_length=5, polyorder=3)
    smoothed_values[zero_indices] = 0
    return smoothed_values


def to_semitones(first, second):
    if first is None or first == 0:
        return None
    if second is None or second == 0:
        return None
    value = 12 * math.log2(first/second)
    return value


if __name__ == '__main__':
    main()
