"""Data representation classes."""
import os
import csv
from torch.utils.data import Dataset


def get_csv_len(csv_path: str):
    """Number of rows in csv data."""
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)  # delete header for counting
        return sum(1 for _ in reader), header


def get_csv_row_on_ind(csv_path: str, ind: int):
    """Get ind-th row in csv data"""
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # delete header
        for _ in range(ind - 1):
            next(reader)
        return next(reader)


class DataColumnSentences(object):
    """Conver data column as sentences."""
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
    """Conver data column as sentences."""
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
    """Item is one of element in some row of cosme col of some dataset."""
    def __init__(self, data_dir, return_row=False):
        self.data_dir = data_dir

        self._index_map = []
        # can we do it faster (with map)?
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
                # print('pth, len, ind:', file_path, file_len, index)
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
                # print('col, index:', col, index)
                res = get_csv_row_on_ind(file_path, index)
                if self.return_row:
                    return res
                # print('res:', res[col])
                return res[col]
        raise Exception('getitem returned None')
