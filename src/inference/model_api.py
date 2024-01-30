"""Rise http-server with RL-agent."""
from flask import Flask, jsonify, request
import os
from scipy import spatial
import pickle
import numpy as np
import csv
from src.embeddings.get_embeddins import ColEmbedding
from gensim.models import FastText

model_pth = 'models/fasttext_one_element_240130-043242.model'

app = Flask(__name__)

pwd = os.getcwd()
with open(os.path.join(pwd, 'embeddings', 'embeddings.pkl'), 'rb') as f:
    embeddings_info = pickle.load(f)

embeddings_vectors = np.array([emb.embedding for emb in embeddings_info])
emb_tree = spatial.KDTree(embeddings_vectors)

model = FastText.load(model_pth)


@app.route('/predict_on_vector', methods=['POST'])
def predict_on_vector_post_request():
    data = request.json
    vec = data['embedding']
    # get the indices of the nearest neighbors
    neighbors = emb_tree.query(vec, k=10)[1]
    res = [embeddings_info[i] for i in neighbors]

    return jsonify({
        'neighbor_columns': res
        })


@app.route('/predict_on_file', methods=['POST'])
def predict_on_file_post_request():
    data = request.json
    file_pth = data['file_pth']
    cols_emb = []
    with open(file_pth) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        embeddings = {col_name: 0 for col_name in header}
        rows_count = 0
        line = next(reader, None)
        n_cols = len(header)
        while line is not None:
            if len(line) != n_cols:
                print('Count of cells in some rows != count of cols in',
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
                                         embedding=embeddings[col_name])
            cols_emb.append(col_emb_tuple)
    res = []
    for emb in cols_emb:
        neighbors_ind = emb_tree.query(emb.embedding, k=10)[1]
        neighbors_info = [embeddings_info[i] for i in neighbors_ind]
        res.append({'column': emb.col_name, 'neighbours':
                    {'names': neighbors_info}})

    return jsonify({
        'col_neighbours_map': res
        })


def raise_server(host: str = '0.0.0.0', port: int = 6113):
    """
    Raise server on host:port.

    Args:
        host (str, optional): host. Defaults to '0.0.0.0'.
        port (int, optional): port. Defaults to 6113.
    """
    app.run(host=host, port=port)


# def get_model_prediction(file_pth: str, host: str = '127.0.0.1',
#                          port: int = 6113) -> Union[list[float], None]:
#     data = {'file_pth': file_pth}
#     address = 'http://' + host + ':' + str(port) + \
#         '/predict_on_file_post_request'
#     response = requests.post(address, json=data)
#     if response.status_code == 200:
#         result = response.json()
#         col_neighbours_map = result['col_neighbours_map']
#         return col_neighbours_map
#     else:
#         print(f'Response code: {response.status_code}')
#         return None


if __name__ == "__main__":
    raise_server()
