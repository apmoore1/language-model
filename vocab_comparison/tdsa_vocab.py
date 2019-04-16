import sys
from pathlib import Path
if str(Path(__file__, '..', '..').resolve()) not in sys.path:
    sys.path.append(str(Path(__file__, '..', '..').resolve()))

import argparse
import json
from typing import Set, List, Callable

from bella.data_types import TargetCollection

from helper import parse_path, get_tokeniser, load_dataset_splits

def get_tokens(data: TargetCollection, tokeniser: Callable[[str], List[str]],
               field: str = 'text') -> Set[str]:
    unique_tokens = set()
    for sentence_id, targets in data.grouped_sentences.items():
        if len(targets) == 0:
            raise ValueError('There should be at least one sentence per id')
        target = targets[0]
        text = target[field]
        for token in tokeniser(text):
            unique_tokens.add(token)
    return unique_tokens

def dataset_vocab(directory: Path, dataset_name: str, 
                  tokeniser: Callable[[str], List[str]], 
                  field: str = 'text') -> Set[str]:
    all_tokens = set()
    for data in load_dataset_splits(directory, dataset_name):
        data_tokens = get_tokens(data, tokeniser, field)
        all_tokens = all_tokens.union(data_tokens)
    return all_tokens

if __name__ == '__main__':
    tdsa_name_help = 'Name of the TDSA dataset that you wish the vocab will be created for'
    vocab_fp_help = 'File Path to store the created vocabulary'
    tokeniser_help = 'Which tokeniser to use, default is spaCy'
    data_dir_help = 'The directory where the TDSA data splits are stored e.g.'\
                    ' if you followed the README.md instruction this would be '\
                    'at the following path ./tdsa_data/splits'

    parser = argparse.ArgumentParser()
    parser.add_argument("tdsa_name", type=str, help=tdsa_name_help, 
                        choices=['laptop', 'restaurant', 'election'])
    parser.add_argument("vocab_fp", type=parse_path, help=vocab_fp_help)
    parser.add_argument("tokeniser", type=str, choices=['spacy', 'whitespace'], 
                        default='spacy', help=tokeniser_help)
    parser.add_argument("data_dir", help=data_dir_help, type=parse_path)
    parser.add_argument("--targets_only", help='Only get the target words', 
                        action="store_true")
    args = parser.parse_args()

    tdsa_name_mapping = {'laptop': 'Laptop', 'restaurant': 'Restaurant', 
                         'election': 'Election'}
    tdsa_name = tdsa_name_mapping[args.tdsa_name]

    vocab_fp = args.vocab_fp
    tokeniser = get_tokeniser(args.tokeniser)
    data_dir = args.data_dir
    if not data_dir.is_dir():
        raise ValueError(f'The data directory path given {data_dir} is '
                         'not a directory')
    if args.targets_only:
        print('Getting only the Target words vocabularly')
        tdsa_vocab = list(dataset_vocab(data_dir, tdsa_name, tokeniser, 
                                        field='target'))
    else:
        print('Getting the whole TDSA datasets vocabularly')
        tdsa_vocab = list(dataset_vocab(data_dir, tdsa_name, tokeniser))
    with vocab_fp.open('w+') as vocab_file:
        json.dump(tdsa_vocab, vocab_file)
    print(f'TDSA vocabularly for {tdsa_name} has been created at {vocab_fp}')