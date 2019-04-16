import sys
from pathlib import Path
if str(Path(__file__, '..', '..').resolve()) not in sys.path:
    sys.path.append(str(Path(__file__, '..', '..').resolve()))
if str(Path(__file__, '..').resolve()) not in sys.path:
    sys.path.append(str(Path(__file__, '..').resolve()))
import argparse
from collections import Counter
from typing import Iterable, Callable, Optional, List
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns

import helper
from data_analysis_helper import yelp_text_generator, text_generator_func_mapper

def yelp_review_generator(data_dir: Path, 
                          split_names: Optional[List[str]] = None
                          ) -> Iterable[Path]:
    split_names = split_names or ['train.json', 'val.json', 'test.json']
    for split_name in split_names:
        data_fp = Path(data_dir, split_name).resolve()
        yield data_fp

def yelp_sentence_generator(data_dir: Path, 
                            split_names: Optional[List[str]] = None
                            ) -> Iterable[Path]:
    split_names = split_names or ['split_train.txt', 'split_val.txt', 'split_test.txt']
    for split_name in split_names:
        data_fp = Path(data_dir, split_name).resolve()
        yield data_fp

def one_billion_generator(data_dir: Path) -> Iterable[Path]:
    test_data = Path(data_dir, 'heldout-monolingual.tokenized.shuffled', 
                     'news.en-00000-of-00100')
    train_data_dir = Path(data_dir, 'training-monolingual.tokenized.shuffled')
    all_fps = [test_data]
    for train_fp in train_data_dir.iterdir():
        if helper.valid_text_file_name_prefix('news', train_fp.name):
            all_fps.append(train_fp)
        else:
            print(f'Not including this file {train_fp} within the training '
                  'directory of the one billion word generator')
    for data_fp in all_fps:
        yield data_fp

def data_generator_mapper(data_name: str, data_dir: Path) -> Iterable[Path]:
    if data_name == 'yelp':
        return yelp_review_generator(data_dir)
    elif data_name == 'yelp_sentences':
        return yelp_sentence_generator(data_dir)
    elif data_name == 'billion_word_corpus':
        return one_billion_generator(data_dir)
    else:
        raise ValueError('Do not have a text generator function for the '
                         f'{data_name} review data')

if __name__ == '__main__':
    data_dir_help = "Directory that contains the train, val, and test splits,"\
                    " this will also be the directory where the new data "\
                    "will be stored."

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=helper.parse_path, 
                        help=data_dir_help)
    parser.add_argument("dataset_name", help="Name of the dataset e.g. yelp", 
                        choices=["yelp", "yelp_sentences", "billion_word_corpus"], 
                        type=str)
    parser.add_argument("--sentence_length_distribution", type=helper.parse_path)
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.is_dir():
        raise ValueError(f'Data directory {data_dir} is not a directory.')


    text_generator_func = text_generator_func_mapper(args.dataset_name)
    data_generator = data_generator_mapper(args.dataset_name, data_dir)

    sentence_lengths = []
    sentence_length_counter = Counter()
    token_frequencies = Counter()

    for data_fp in data_generator:
        for text in text_generator_func(data_fp):
            tokens = text.split()
            token_frequencies.update(tokens)
            num_tokens = len(tokens)
            sentence_lengths.append(num_tokens)
            sentence_length_counter.update([num_tokens])
    print(f'Max sentence length {max(sentence_lengths)}')
    print(f'Min sentence length {min(sentence_lengths)}')
    print(f'Mean sentence length {mean(sentence_lengths)}')
    for i in range(2, 11):
        token_counts_i = [token_frequency for token_frequency 
                          in token_frequencies.values() if token_frequency > i]
        print(f'Number of tokens with counts greater than {i} {len(token_counts_i)}')
    for i in range(1, 11):
        print(f'Number of sentences {sentence_length_counter[i]} at this length {i}')
    if args.sentence_length_distribution:
        fig, ax = plt.subplots()
        ax = sns.distplot(sentence_lengths, ax=ax)
        fig.savefig(str(args.sentence_length_distribution.resolve()))
    
