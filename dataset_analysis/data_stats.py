import sys
from pathlib import Path
if str(Path(__file__, '..', '..').resolve()) not in sys.path:
    sys.path.append(str(Path(__file__, '..', '..').resolve()))
if str(Path(__file__, '..').resolve()) not in sys.path:
    sys.path.append(str(Path(__file__, '..').resolve()))
import argparse
from collections import Counter
from typing import Iterable, Callable, Optional, List, Dict, Any, Union
from statistics import mean, stdev
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

def sent_len_greater_than_filter(count_dict: Dict[Any, Union[int,float]], 
                                 filter_number: Union[int,float]) -> int:
    count = 0
    for length, number_in_length in count_dict.items():
        if length > filter_number:
            count += number_in_length
    return count

def token_greater_than_filter(count_dict: Dict[Any, Union[int,float]], 
                              filter_number: Union[int,float]) -> int:
    count = 0
    for value in count_dict.values():
        if value > filter_number:
            count += 1
    return count


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
    file_path_help = "File path that contains data on each new line. Where "\
                     "the data is parsed based on the dataset_name flag"\
                     "This can also be a directory that contains the train, "\
                     "val, and test splits."
    dataset_name_help = "Name of the dataset e.g. yelp"

    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=helper.parse_path, 
                        help=file_path_help)
    parser.add_argument("--is_dir", action="store_true", 
                        help="Whether or not the file path given is a directory")
    parser.add_argument("dataset_name", help=dataset_name_help, 
                        choices=["yelp", "yelp_sentences", "billion_word_corpus"], 
                        type=str)
    parser.add_argument("--sentence_length_distribution", type=helper.parse_path)
    args = parser.parse_args()


    text_generator_func = text_generator_func_mapper(args.dataset_name)
    data_generator = [args.file_path]
    if args.is_dir:
        if not args.file_path.is_dir():
            raise ValueError(f'Data directory {args.file_path} is not a directory.')
        data_generator = data_generator_mapper(args.dataset_name, args.file_path)

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
    total_num_sentences = len(sentence_lengths)
    print(f'Total number of sentences: {len(sentence_lengths)}')
    print(f'Max sentence length {max(sentence_lengths)}')
    print(f'Min sentence length {min(sentence_lengths)}')
    print(f'Mean sentence length {mean(sentence_lengths)}')
    stdev_sentences = stdev(sentence_lengths)
    print(f'Standard Deviation {stdev_sentences}')
    for i in range(2, 11):
        token_counts_i = token_greater_than_filter(token_frequencies, i)
        print(f'Number of tokens with counts greater than {i} {token_counts_i}')
    mean_plus_stdev = mean(sentence_lengths) + stdev_sentences
    sentence_counts_mstdev = sent_len_greater_than_filter(sentence_length_counter, 
                                                          mean_plus_stdev)
    print('Number of sentences of length greater than the mean plus standard deviation'
          f' ({mean_plus_stdev}) is {sentence_counts_mstdev}'
          f' ({(sentence_counts_mstdev/total_num_sentences) * 100}%)')
    for i in [40,50,60,70,80,100,120,150,200,300]:
        sentence_counts_i = sent_len_greater_than_filter(sentence_length_counter, i)
        print(f'Number of sentence of length greater than {i} {sentence_counts_i}'
              f' ({(sentence_counts_i/total_num_sentences) * 100}%)')
    for i in range(1, 11):
        print(f'Number of sentences {sentence_length_counter[i]} at this length {i}')
    print(f'Total number of tokens {sum(token_frequencies.values())}')
    if args.sentence_length_distribution:
        fig, ax = plt.subplots()
        ax = sns.distplot(sentence_lengths, ax=ax)
        fig.savefig(str(args.sentence_length_distribution.resolve()))
    
