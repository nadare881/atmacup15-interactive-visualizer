import os

import pandas as pd
import numpy as np

from gensim.models import word2vec

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM_DIMENTION = 32

if __name__ == "__main__":
    anime_df = pd.read_csv(os.path.join(ROOT_DIR, "inputs", "anime.csv"))
    train_df = pd.read_csv(os.path.join(ROOT_DIR, "inputs", "train.csv"))
    test_df = pd.read_csv(os.path.join(ROOT_DIR, "inputs", "test.csv"))

    all_df = pd.concat([train_df, test_df],axis=0)
    text = [df["anime_id"].values.tolist() for _, df in all_df.groupby("user_id")]

    vector_size = 64
    w2v_params = {
        "vector_size": vector_size,  ## <= 変更点
        "seed": 42,
        "window": 2000,
        "min_count": 1,
        "workers": 24,
        "epochs": 20
    }
    model = word2vec.Word2Vec(text, **w2v_params)

    res = np.zeros([anime_df.shape[0], vector_size])
    mean_ = model.wv.vectors.mean(axis=0)

    for i, v in enumerate(anime_df["anime_id"]):
        if model.wv.has_index_for(v):
            res[i] = model.wv[v]
        else:
            res[i] = mean_
    np.save(os.path.join(ROOT_DIR, "vectors", "anime2vec_baseline.npy"), res)
