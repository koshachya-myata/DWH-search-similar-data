"""Startpoint."""
from data_generation.generate_simulation_data import generate_data
from src.inference import raise_server
from data_generation.disintersect import disintersect_folders
import sys
from config import DATA_DIR
from src.embeddings.FastTextOneElement import FastTextOneElement

if __name__ == '__main__':
    args = sys.argv
    print(f'Started with process arg "{args[1]}"')
    process_arg = args[1].lower()
    if process_arg == 'generate':
        print('Generate fake data.')
        generate_data(save_dir_path='test_data',
                      fake_data_count=15,
                      random_queris=3,
                      table_per_query=15
                      )
    if process_arg == 'train_ft_one_elem':
        print('Train FastText on elements of data.')
        model = FastTextOneElement(DATA_DIR)
        model.train(epochs=5000)

    if process_arg == 'disitersect_test':
        print('Delete intesect folders in test_data.')
        disintersect_folders(check_intersectoin_path='data',
                             source_path='test_data')
    if process_arg == 'raise_server':
        print('Raise server')
        raise_server()
