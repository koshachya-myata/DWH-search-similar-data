"""Startpoint."""
from data_generation.generate_simulation_data import generate_data
import sys
from config import DATA_DIR
from src.embeddings.FastTextOneElement import FastTextOneElement

if __name__ == '__main__':
    args = sys.argv
    process_arg = args[1].lower()
    if process_arg == 'generate':
        generate_data()
    if process_arg == 'train_ft_one_elem':
        print('Train FastText on elements of data')
        model = FastTextOneElement(DATA_DIR)
        model.train(epochs=3)
