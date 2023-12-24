import numpy as np
from collections import namedtuple

ColEmbedding = namedtuple('ColEmbedding', 'file_pth, col_name, embedding')


def get_embeddings_cell_base(model, tables, file_pths):
    """
    Takes a model and a list of tables, computes embeddings for each column
    of each table.
    """
    res = []
    for table in range(len(tables)):
        for col_name in table.columns:
            embedding = compute_column_embedding(model,
                                                 tables[table][col_name])
            result_tuple = ColEmbedding(file_pth=file_pths[table],
                                        col_name=col_name,
                                        embedding=embedding)
            res.append(result_tuple)
    return res


def compute_column_embedding(model, column):
    """
    Computes the embedding for a given column using the specified model.
    """
    column_values = column.astype(str).tolist()
    embeddings = [model.encode(value) for value in column_values]
    avg_embedding = np.mean(embeddings,
                            axis=0)

    return avg_embedding
