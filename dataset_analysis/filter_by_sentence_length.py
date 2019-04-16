import sys
from pathlib import Path
if str(Path(__file__, '..', '..').resolve()) not in sys.path:
    sys.path.append(str(Path(__file__, '..', '..').resolve()))
if str(Path(__file__, '..').resolve()) not in sys.path:
    sys.path.append(str(Path(__file__, '..').resolve()))

import argparse

import helper
from data_analysis_helper import text_generator_func_mapper


if __name__ == '__main__':
    data_dir_help = "Directory that contains the train, val, and test splits,"\
                    " this will also be the directory where the new data "\
                    "will be stored."

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=helper.parse_path, 
                        help=data_dir_help)
    parser.add_argument("dataset_name", help="Name of the dataset e.g. yelp_sentences", 
                        choices=["yelp_sentences"], 
                        type=str)
    parser.add_argument("min_sentence_length", type=int, 
                        help='Minimum sentence length if -1 then no minimum')
    parser.add_argument("max_sentence_length", type=int, 
                        help='Minimum sentence length if -1 then no maximum')
    args = parser.parse_args()

    min_sentence_length = args.min_sentence_length
    max_sentence_length = args.max_sentence_length

    data_dir = args.data_dir
    if not data_dir.is_dir():
        raise ValueError(f'Data directory {data_dir} is not a directory.')

    text_generator_func = text_generator_func_mapper(args.dataset_name)
    split_names = ["split_train.txt", "split_val.txt", "split_test.txt"]
    new_split_names = ["filtered_split_train.txt", "filtered_split_val.txt", 
                       "filtered_split_test.txt"]
    for split_name, new_split_name in zip(split_names, new_split_names):
        split_fp = Path(data_dir, split_name)
        new_split_fp = Path(data_dir, new_split_name)

        count = 0
        with new_split_fp.open('w+') as new_split_file:
            for text in text_generator_func(split_fp):
                tokens = text.split()
                sentence_length = len(tokens)
                if min_sentence_length != -1:
                    if sentence_length < min_sentence_length:
                        continue
                if max_sentence_length != -1:
                    if sentence_length > max_sentence_length:
                        continue
                if count != 0:
                    text = f'\n{text}'
                new_split_file.write(text)
                count += 1
        