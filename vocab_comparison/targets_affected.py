import sys
from pathlib import Path
if str(Path(__file__, '..', '..').resolve()) not in sys.path:
    sys.path.append(str(Path(__file__, '..', '..').resolve()))

import argparse
from collections import Counter
from typing import Optional, Iterable, List

from bella.data_types import TargetCollection

from helper import parse_path, get_tokeniser

def load_vocab(vocab_fp: Path) -> List[str]:
    vocab = []
    with vocab_fp.open('r') as vocab_file:
        for line in vocab_file:
            line = line.strip()
            if line:
                vocab.append(line)
    return vocab

def load_dataset_splits(directory: Path, dataset_name: str,
                        split_names: Optional[List[str]] = None
                        ) -> Iterable[TargetCollection]:
    split_names = split_names or ['Train', 'Val', 'Test']
    for split_name in split_names:
        data_split_fp = Path(directory, f'{dataset_name} {split_name}').resolve()
        if not data_split_fp.exists():
            raise FileNotFoundError('Cannot find the following TDSA data '
                                    f'split file {data_split_fp}')
        yield TargetCollection.load_from_json(data_split_fp)

if __name__ == '__main__':
    tdsa_name_help = 'Name of the TDSA dataset that you wish the vocab will be created for'
    vocab_fp_help = 'File Path to the created vocab'
    tokeniser_help = 'Which tokeniser to use, default is spaCy'
    data_dir_help = 'The directory where the TDSA data splits are stored e.g.'\
                    ' if you followed the README.md instruction this would be '\
                    'at the following path ./tdsa_data/splits'
    unique_help = 'The returned affected count only includes each target '\
                  'once rather than relative count'

    parser = argparse.ArgumentParser()
    parser.add_argument("tdsa_name", type=str, help=tdsa_name_help, 
                        choices=['laptop', 'restaurant', 'election'])
    parser.add_argument("vocab_fp", type=parse_path, help=vocab_fp_help)
    parser.add_argument("tokeniser", type=str, choices=['spacy', 'whitespace'], 
                        default='spacy', help=tokeniser_help)
    parser.add_argument("data_dir", help=data_dir_help, type=parse_path)
    parser.add_argument("--targets_affected_count", type=int, default=5, 
                        help='How many targets that are affected to print out.')
    parser.add_argument("--unique", help=unique_help, action="store_true")
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

    not_in_model_vocab = load_vocab(vocab_fp)

    if not not_in_model_vocab:
        raise ValueError("All the vocab words are in the model's vocab")
    
    targets_affected = Counter()
    total_targets = Counter()
    for data in load_dataset_splits(data_dir, tdsa_name):
        for target in data.data_dict():
            target_word = target['target']
            total_targets.update([target_word])
            target_tokens = tokeniser(target_word)
            for token in target_tokens:
                if token in not_in_model_vocab:
                    targets_affected.update([target_word])
                    break
    if args.unique:
        print(f'Unique target count: {len(targets_affected)}')
    else:
        print(f'Relative target count: {sum(targets_affected.values())}')
    print(f'Out of {sum(total_targets.values())} relatively')
    print(f'Out of {len(total_targets)} uniquely')
    print('Sample of the targets affected')
    targets_affected_keys = list(targets_affected.keys())
    for i in range(args.targets_affected_count):
        print(f'{targets_affected_keys[i]}')

            