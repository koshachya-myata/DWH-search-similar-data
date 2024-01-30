"""Data representation classes."""
import os
import csv
from torch.utils.data import Dataset
from collections import namedtuple
from typing import List
import numpy as np


def get_csv_len(csv_path: str):
    """Get number of data rows in csv file."""
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)  # delete header for counting
        return sum(1 for _ in reader), header


def get_csv_row_on_ind(csv_path: str, ind: int):
    """Get ind-th row in csv data."""
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # delete header
        for _ in range(ind - 1):
            next(reader)
        return next(reader)


class DataColumnSentences(object):
    """Clasas for manage data column as sentences."""

    def __init__(self, datafile: str, column: str,
                 delimiter: str = ",", quotechar: str | None = '"',
                 lineterminator: str = "\r\n"
                 ):
        """Init."""
        self.column = column
        self.datafile = datafile
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.lineterminator = lineterminator
        # self.data_reader = csv.DictReader(self.datafile,
        #                                   delimiter=self.delimiter,
        #                                   quotechar=self.quotechar,
        #                                   lineterminator=self.lineterminator)

    def __iter__(self):
        """Iter over data."""
        data_reader = csv.DictReader(self.datafile, delimiter=self.delimiter,
                                     quotechar=self.quotechar,
                                     lineterminator=self.lineterminator)
        for row in data_reader:
            yield row[self.column]


class DataSentences(object):
    """Convert data column as sentences."""
    def __init__(self, data_dir: str):
        """Init."""
        self.data_dir = data_dir

    def __iter__(self):
        """iter over data."""
        for address, dirs, files in os.walk(self.data_dir):
            for name in files:
                file_pth = os.path.join(address, name)
                if file_pth[-4:] == '.csv':
                    with open(file_pth, newline='') as f:
                        reader = csv.reader(f)
                        cols = next(reader).split(',')
                        for col in cols:
                            dcs = DataColumnSentences(file_pth, col)
                            for sentence in dcs:
                                yield sentence


class AllCsvDataSet(Dataset):
    """
    Item is one of element in some row of col of some dataset.

    В отличие от DirectIterationDataset честно считывает данные,
    перемешивая их. Но __getitem__ работает за O(N).
    """
    def __init__(self, data_dir, return_row=False):
        self.data_dir = data_dir

        self._index_map = []
        # can we do it faster (with map)
        # but we need memory control.

        self._all_rows_len = 0
        self.return_row = return_row
        self._all_len = 0
        for address, dirs, files in os.walk(self.data_dir):
            for name in files:
                file_pth = os.path.join(address, name)
                if file_pth[-4:] == '.csv':
                    file_len, header = get_csv_len(file_pth)
                    self._all_rows_len += file_len
                    self._all_len += file_len * len(header)
                    self._index_map.append((
                        self._all_len,
                        file_pth,
                        file_len
                    ))

    def __len__(self):
        if self.return_row:
            return self._all_rows_len
        return self._all_len

    def __getitem__(self, idx):
        # it must be sorted for el[0]
        for i in range(len(self._index_map)):
            el = self._index_map[i]
            if idx < el[0]:
                file_path = el[1]
                file_len = el[2]
                index = idx
                if i > 0:
                    index = idx - self._index_map[i-1][0]
                len_now = 0
                col = 0
                while len_now < index:
                    len_now += file_len
                    col += 1
                if col:
                    col -= 1
                index -= col * file_len
                res = get_csv_row_on_ind(file_path, index)
                if self.return_row:
                    return res
                return res[col]
        raise Exception('getitem returned None')


class DirectIterationDataset(Dataset):
    """
    Датасет, который читает данные в колонках подряд.

    Для каждой таблицы создает свой ридер. И построчно, сверху вниз, считывает
    данные. В таком случае getitem работает за O(кол-во таблиц) (с учетом,
    векторых операций, может и за O(1), но данные для обучения
    не перемешиваются.
    """
    FileCsvReader = namedtuple('FileCsvReader', 'file, file_pth, reader')
    # IterMonitor = namedtuple('IterMonitor', 'iteration, len')

    def __init__(self, data_dir, return_row=False):
        self.data_dir = data_dir

        self._readers: List[self.FileCsvReader] = []
        self._last_rows = []
        self.accum_cols_count = [0]
        for address, dirs, files in os.walk(self.data_dir):
            for name in files:
                file_pth = os.path.join(address, name)
                if file_pth[-4:] == '.csv':
                    file = open(file_pth)
                    try:
                        reader = csv.reader(file)
                        header = next(reader)
                    except UnicodeDecodeError:
                        print(f'Bad symbol in {file_pth}.')
                        continue
                    self.accum_cols_count.append(self.accum_cols_count[-1] +
                                                 len(header))
                    self._last_rows.append(header)

                    new_elem = self.FileCsvReader(file=file, reader=reader,
                                                  file_pth=file_pth)
                    self._readers.append(new_elem)
        self._readed = [
            np.array(
                [True] * (
                    self.accum_cols_count[i] - self.accum_cols_count[i-1]
                    )
            ) for i in range(1, len(self.accum_cols_count))
            ]
        self._iter_table_map = {}
        now_iter = 0
        for i in range(1, len(self.accum_cols_count)):
            for j in range(self.accum_cols_count[i] -
                           self.accum_cols_count[i-1]):
                self._iter_table_map[now_iter+j] = i - 1
            now_iter += self.accum_cols_count[i] - self.accum_cols_count[i-1]
        self.accum_cols_count.pop(0)
        self._len = sum([len(el) for el in self._readed])

    def __len__(self):
        # if one iterate over dataset = iterate over all columns once
        return self._len

    def __getitem__(self, idx):
        table_id = self._iter_table_map[idx]
        col_ind = idx - self.accum_cols_count[table_id]
        file, file_pth, reader = self._readers[table_id]
        if self._readed[table_id][col_ind] is True:
            if np.all(self._readed[table_id]):
                try:
                    self._last_rows[table_id] = next(reader)
                except StopIteration:
                    file.close()
                    file = open(file_pth)
                    reader = csv.reader(file)
                    next(reader)  # skip header
                    self._last_rows[table_id] = next(reader)
                finally:
                    self._readed[table_id].fill(False)
        rt_elem = self._last_rows[table_id][col_ind]
        self._readed[table_id][col_ind] = True
        return rt_elem.split()


class CellSentence(object):
    """Iterate in format: one sentence = one cell;"""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def __iter__(self):
        """Iterate through the lines in the source."""
        for address, dirs, files in os.walk(self.data_dir):
            for name in files:
                file_pth = os.path.join(address, name)
                if file_pth[-4:] == '.csv':
                    with open(file_pth) as f:
                        reader = csv.reader(f)
                        try:
                            line = next(reader, None)  # delete header for counting
                            line = next(reader, None)
                        except UnicodeDecodeError:
                            print(f'Bad symbol in {file_pth}.')
                            continue
                        while line is not None:
                            for cell in line:
                                yield cell.split()
                            try:
                                line = next(reader, None)
                            except UnicodeDecodeError:
                                print(f'Bad symbol in {file_pth}.')
