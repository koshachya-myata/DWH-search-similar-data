"""Class for download datasets from kaggle."""
import os
import json
import subprocess
from typing import Optional
import zipfile
import kaggle
from faker import Faker
KAGGLE_CONFIG_DIR = os.path.join(os.path.expandvars('$HOME'), '.kaggle')


class RandomKaggleDataset:
    def __init__(self, random_queris=10,
                 table_per_query=10, data_path='data',
                 min_bytes=1024, max_bytes=3*1024**2):
        """Also init self.search_words."""
        self.search_words = ['security', 'users', 'traffic', 'packet',
                             'economics', 'client']
        self._used_data = set()
        self.random_queris = random_queris
        self.table_per_query = table_per_query
        self.data_path = data_path
        self.min_bytes = min_bytes
        self.max_bytes = max_bytes
        self.api = kaggle.api
        self.fake = Faker(locale='en-US')

    def download_dataset(self, dataset: str):
        """
        Download dataset to self.data_path.

        Args:
            dataset (str): dataset in format {user}/{data-set-name}
        """
        command = f'kaggle datasets download -d {dataset} -p {self.data_path}'
        subprocess.run(command, shell=True)

    def process_dataset_zip(self, dataset: str):
        """
        Unpack archive to self.data_fath dir and delete .zip.

        Args:
            dataset (str): dataset in format {user}/{data-set-name}
        """
        dataset_name = dataset.split('/')[1]
        path_to_zip_file = os.path.join(self.data_path, dataset_name) + '.zip'
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            print(path_to_zip_file)
            print(dataset_name)
            zip_ref.extractall(
                os.path.join(self.data_path, dataset_name)
                )
        os.remove(path_to_zip_file)

    def download_from_search_query(self, search_query: str) -> int:
        """
        Search and download dataset by query.

        Download no more than self.table_per_query datasets.
        Args:
            search_query (str): query.

        Returns:
            int: how mane datasets was downloaded.
        """
        #  read from the first page
        dataset_list = self.api.dataset_list(search=search_query,
                                             min_size=self.min_bytes,
                                             max_size=self.max_bytes,
                                             file_type='csv')[:100]
        downloaded = data_ind = 0
        while downloaded < self.table_per_query and \
                data_ind < len(dataset_list):
            if str(dataset_list[data_ind]) in self._used_data:
                data_ind += 1
                continue
            try:
                self.download_dataset(dataset_list[data_ind])
                downloaded += 1
                self._used_data.add(str(dataset_list[data_ind]))
                self.process_dataset_zip(str(dataset_list[data_ind]))
            except Exception:
                #  if dataset from competition and we dont join it.
                print(f'Bad with {dataset_list[data_ind]}')
                continue
            finally:
                data_ind += 1
        return downloaded

    def download_from_search_words(self):
        """Download datasets using self.search_words."""
        for query in self.search_words:
            self.download_from_search_query(query)

    def download_from_random_words(self):
        """Download datasets using fake.word self.random_queris times."""
        for _ in range(self.random_queris):
            q = self.fake.word()
            print('Random query:', q)
            self.download_from_search_query(q)

    def pipeline(self):
        """Download from self.search_words and from random queris."""
        self.download_from_search_words()
        self.download_from_random_words()

    @staticmethod
    def init_kaggle_config(username: Optional[str] = None,
                           api_key: Optional[str] = None):
        """
        Create kaggle config file in $HOME/.kaggle/kaggle.json

        If username/api_key is None than its takes from std-in.
        Args:
            username (str): kaggle username. Deafults to None
            api_key (str): kaggle api key. Deafults to None
        """
        if username is None:
            username = input('Enter your kaggle username:')
        if api_key is None:
            api_key = input('Enter your kaggle api key:')
        os.makedirs(KAGGLE_CONFIG_DIR, exist_ok=True)
        api_dict = {'username': username, 'key': api_key}
        with open(f"{KAGGLE_CONFIG_DIR}/kaggle.json", "w",
                  encoding='utf-8') as f:
            json.dump(api_dict, f)
        cmd = f"chmod 600 {KAGGLE_CONFIG_DIR}/kaggle.json"
        output = subprocess.check_output(cmd.split(" "))
        output = output.decode(encoding='UTF-8')
        print(output)


if __name__ == '__main__':
    if os.path.exists(KAGGLE_CONFIG_DIR):
        RandomKaggleDataset.init_kaggle_config()
