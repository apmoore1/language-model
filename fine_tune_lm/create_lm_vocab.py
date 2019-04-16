import argparse
from pathlib import Path

from allennlp.common import from_params, Params
from allennlp.data import Vocabulary, DatasetReader

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    '''
    This will allow you to create a pre-computed vocab file for your 
    language model.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("model_config", type=parse_path, 
                        help='File Path to your language models config file.')
    parser.add_argument("vocab_dir", type=parse_path, 
                        help='Directory to save your vocabulary files to.')
    args = parser.parse_args()

    vocab_dir = args.vocab_dir
    if not vocab_dir.is_dir():
        raise ValueError(f'Path for the vocabulary directory {vocab_dir} '
                         'is not a directory')
    # Loads the json config file
    model_config_fp = args.model_config
    params = Params.from_file(model_config_fp)
    # Gets the dataset reader and creates a generator of instances
    reader = DatasetReader.from_params(params['dataset_reader']['base_reader'])
    instances = reader.read(params['train_data_path'])
    # creates the vocabulary and saves it to the given directory
    vocab_params = params['vocabulary']
    vocab = Vocabulary.from_params(params=vocab_params, instances=instances)
    vocab.save_to_files(str(vocab_dir))