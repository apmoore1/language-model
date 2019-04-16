import argparse
from pathlib import Path

from allennlp.common import from_params, Params
from allennlp.data import Vocabulary, DatasetReader

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def write_to_file(text: str, text_fp: Path) -> None:
    with text_fp.open('a+') as text_file:
        text_file.write(text)

if __name__ == '__main__':
    '''
    This will split the dataset into **n_files** files so that the allennlp 
    multiprocess dataset reader can work more affectively. As the 
    multiprocess reader, reads in a concurrent way based on files.
    '''
    n_files_help = 'Number of files to split the training dataset into, '\
                   'of which the files will be saved in the directory given'
    parser = argparse.ArgumentParser()
    parser.add_argument("train_fp", type=parse_path, 
                        help='File Path to your training data')
    parser.add_argument("train_data_dir", type=parse_path, 
                        help='Directory to save the new training data split files.')
    parser.add_argument("n_files", type=int, 
                        help='Number of files to split the training dataset into, of which the files will be saved in the directory given')
    args = parser.parse_args()

    train_fp = args.train_fp
    train_data_dir = args.train_data_dir
    n_files = args.n_files
    if train_data_dir.is_dir():
        print(f'WARNING THE NEW TRAINING DATA DIRECTORY {train_data_dir} '
              'already exists, you should ensure that is empty!!')
    else:
        train_data_dir.mkdir(parents=True, exist_ok=True)

    num_lines = 0
    with train_fp.open('r') as train_file:
        for line in train_file:
            num_lines += 1
    
    lines_per_file = int(num_lines / n_files)
    print(f'Roughly {lines_per_file} lines per file')
    all_new_fps = [Path(train_data_dir, f'train_file_{file_number}') 
                   for file_number in range(n_files)]

    with train_fp.open('r') as train_file:
        line_count = 0
        fp_count = 0
        for line in train_file:
            line = line.strip()
            if line:
                if line_count != 0:
                    line = f'\n{line}'
                write_to_file(line, all_new_fps[fp_count])
                line_count += 1
            if line_count == lines_per_file:
                if (fp_count + 1) == n_files:
                    pass
                else:
                    line_count = 0
                    fp_count += 1