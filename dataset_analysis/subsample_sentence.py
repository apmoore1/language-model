import argparse
import sys
from pathlib import Path
if str(Path(__file__, '..', '..').resolve()) not in sys.path:
    sys.path.append(str(Path(__file__, '..', '..').resolve()))
if str(Path(__file__, '..').resolve()) not in sys.path:
    sys.path.append(str(Path(__file__, '..').resolve()))
import random

import helper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=helper.parse_path, 
                        help='File Path to the sentences to subsample from')
    parser.add_argument("subsampled_fp", type=helper.parse_path,
                        help="File path to store subsampled sentences")
    parser.add_argument("number_sentences", type=int, 
                        help="Number of sentences to subsample")
    args = parser.parse_args()

    count = 0
    with args.file_path.open('r') as data_file:
        for line in data_file:
            count += 1
    print(f'Number sentences in larger file {count}')
    first_sentence = True
    probability_sentence = args.number_sentences / count
    not_probability = 1 - probability_sentence
    with args.subsampled_fp.open('w+') as subsampled_file:
        with args.file_path.open('r') as data_file:
            for line in data_file:
                line = line.strip()
                if line:
                    choice = random.choices([0, 1], [not_probability, 
                                                     probability_sentence])[0]
                    if choice and first_sentence:
                        first_sentence = False
                        subsampled_file.write(line)
                    elif choice and not first_sentence:
                        subsampled_file.write(f'\n{line}')
    num_subsampled_lines = 0
    with args.subsampled_fp.open('r') as subsampled_file:
        for line in subsampled_file:
            num_subsampled_lines += 1
    print(f'Wrote {num_subsampled_lines} to {args.subsampled_fp}')
