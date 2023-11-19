"""Startpoint."""
from data_generation.generate_simulation_data import generate_data
import sys

if __name__ == '__main__':
    args = sys.argv
    process_arg = args[1].lower()
    if process_arg == 'generate':
        generate_data()
