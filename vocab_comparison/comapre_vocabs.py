import sys
from pathlib import Path
if str(Path(__file__, '..', '..').resolve()) not in sys.path:
    sys.path.append(str(Path(__file__, '..', '..').resolve()))

import argparse
import json
from typing import Set

from helper import parse_path

def load_vocab(vocab_fp: Path) -> Set[str]:
    with vocab_fp.open('r') as vocab_file:
        vocab = json.load(vocab_file)
    return set(vocab)

if __name__ == '__main__':
    '''
    This script will write two a seperate text file the differences in words 
    between two json vocabulary files. It will also print the number of words 
    that are different and the number of words in the two vocabulary files.
    '''
    vocab_fp_1_help = 'File path the first vocabulary file that was created '\
                      'from the create_vocab.py script'
    vocab_fp_2_help = 'File path the second vocabulary file that was created '\
                      'from the create_vocab.py script'
    vocab_fp_diff_help = 'File path to the file that will store the words '\
                         'that are different between the two vocabularys, '\
                         'each word will be written on a new line'
    not_symmetric_help = "Difference between vocab 1 and vocab 2 e.g. words "\
                         "that occur in vocab 1 but not in vocab 2 the default"\
                         " is the difference in vocab 1 and vocab 2"
    parser = argparse.ArgumentParser()
    parser.add_argument("vocab_fp_1", type=parse_path, 
                        help=vocab_fp_1_help)
    parser.add_argument("vocab_fp_2", type=parse_path, 
                        help=vocab_fp_2_help)
    parser.add_argument("vocab_fp_difference", type=parse_path, 
                        help=vocab_fp_diff_help)
    parser.add_argument("--not_symmetric", action="store_true", 
                        help=not_symmetric_help)
    parser.add_argument("--json", action="store_true", 
                        help='Whether or not the output should be in json format.')
    args = parser.parse_args()

    vocab_1 = load_vocab(args.vocab_fp_1)
    vocab_2 = load_vocab(args.vocab_fp_2)
    diff_vocab = vocab_2.symmetric_difference(vocab_1)
    if args.not_symmetric:
        diff_vocab = vocab_1.difference(vocab_2)
    print(f'Number of words in vocabulary 1 {len(vocab_1)}')
    print(f'Number of words in vocabulary 2 {len(vocab_2)}')
    print(f'Number of difference words between the two vocabularies {len(diff_vocab)}')

    vocab_diff_fp = args.vocab_fp_difference
    with vocab_diff_fp.open('w+') as vocab_diff_file:
        if args.json:
            json.dump(list(diff_vocab), vocab_diff_file)
        else:
            for index, token in enumerate(diff_vocab):
                if index == 0:
                    vocab_diff_file.write(f'{token}')
                    continue
                vocab_diff_file.write(f'\n{token}')
    print('The difference between the two vocabularies has been written to '
          f'{vocab_diff_fp}')




