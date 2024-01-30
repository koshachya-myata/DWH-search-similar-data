import numpy as np
from collections import namedtuple
import os
import csv

ColEmbedding = namedtuple('ColEmbedding', 'file_pth, col_name, embedding')


def get_columns_embeddings(model, data_dir) -> list['ColEmbedding']:
    res = []
    for address, dirs, files in os.walk(data_dir):
        for name in files:
            file_pth = os.path.join(address, name)
            if file_pth[-4:] == '.csv':
                with open(file_pth) as f:
                    # TODO: выделить в отдельную функцию
                    reader = csv.reader(f)
                    header = next(reader, None)
                    embeddings = {col_name: 0 for col_name in header}
                    rows_count = 0
                    line = next(reader, None)
                    n_cols = len(header)
                    while line is not None:
                        if len(line) != n_cols:
                            print('Count of cells != count of cols in',
                                  file_pth)
                            line = next(reader, None)
                            continue
                        rows_count += 1
                        for i in range(n_cols):
                            vec = 0
                            for el in line[i].split():
                                vec += model.wv.get_vector(el)
                            if len(line[i].split()) == 0 or np.all(vec == 0):
                                continue
                            vec = vec / len(line[i].split())
                            embeddings[header[i]] += vec / len(line[i].split())
                        line = next(reader, None)
                    for col_name in header:
                        if rows_count == 0:
                            continue
                        embeddings[col_name] /= rows_count
                        col_emb_tuple = ColEmbedding(file_pth=file_pth,
                                                     col_name=col_name,
                                                     embedding=embeddings[
                                                         col_name])
                        if np.all(embeddings[col_name] == 0):
                            continue
                        res.append(col_emb_tuple)
    return res
