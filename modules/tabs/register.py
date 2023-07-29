import os
import json
from typing import List

import gradio as gr
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from modules.ui import Tab, ROOT_DIR

# 検索用データの構築
# TODO: いずれユーザーの実装した検索手法を読み込めるようにする
anime_df = pd.read_csv(os.path.join(ROOT_DIR, "inputs", "anime.csv"))
anime_df["anime_ix"] = anime_df.index
anime_ix_map = {k: v for k, v in anime_df[["anime_id", "anime_ix"]].values}
japanese_name_tfidf_vectorizer = TfidfVectorizer(analyzer="char", binary=True, ngram_range=(1, 3))
japanese_name_tfidf_vec = japanese_name_tfidf_vectorizer.fit_transform(anime_df["japanese_name"])

# load anime vector
anime_vec = np.load(os.path.join(ROOT_DIR, "vectors", "anime2vec_baseline.npy"))
anime_vec_normed = anime_vec / np.maximum(1e-7, np.linalg.norm(anime_vec, ord=2, axis=1, keepdims=True))

def get_saved_review_list():
    res = []
    for file_name in os.listdir(os.path.join(ROOT_DIR, "reviews")):
        if file_name.endswith(".json"):
            res.append(file_name.rstrip(".json"))
    return res

class Register(Tab):
    def title(self):
        return "register"

    def sort(self):
        return 1

    def ui(self, outlet):
        evaluated_animes = {}
        def search_by_title(text: str, num_search: int):
            global japanese_name_tfidf_vec, japanese_name_tfidf_vectorizer
            score = japanese_name_tfidf_vec.dot(japanese_name_tfidf_vectorizer.transform([text]).T).toarray().T[0]
            choices = []
            res = []
            for i, row in anime_df.iloc[np.argsort(-score)].head(num_search).iterrows():
                choices.append("{:04} {}".format(row["anime_ix"], row["japanese_name"]))
                res.append(
                    [
                        row["anime_ix"],
                        row["anime_id"],
                        row["japanese_name"],
                        row["type"],
                        list(row["genres"].split(", "))

                    ]
                )
            return gr.Dropdown.update(choices=choices), gr.DataFrame.update(value=res)

        def recommend_by_reviewed(num_search: int):
            global anime_vec
            nonlocal evaluated_animes
            if len(evaluated_animes) == 0:
                return gr.Dropdown.update(choices=[]), gr.DataFrame.update(value=[])

            query = anime_vec[list(evaluated_animes.keys())].mean(axis=0)
            score = np.einsum("d,nd->n", query, anime_vec_normed)
            for k in evaluated_animes.keys():
                score[k] = -1

            choices = []
            res = []
            for i, row in anime_df.iloc[np.argsort(-score)].head(num_search).iterrows():
                choices.append("{:04} {}".format(row["anime_ix"], row["japanese_name"]))
                res.append(
                    [
                        row["anime_ix"],
                        row["anime_id"],
                        row["japanese_name"],
                        row["type"],
                        list(row["genres"].split(", "))
                    ]
                )
            return gr.Dropdown.update(choices=choices), gr.DataFrame.update(value=res)

        def register_review(key: str, review: int):
            nonlocal evaluated_animes
            k = int(key.split()[0])
            if review == 0:
                if k in evaluated_animes:
                    del evaluated_animes[k]
            else:
                evaluated_animes[k] = review

            reviews = []
            for k, v in evaluated_animes.items():
                reviews.append(
                    [
                        anime_df["anime_id"].values[k],
                        anime_df["japanese_name"].values[k],
                        v
                    ]
                )
            return gr.DataFrame.update(value=reviews)

        def load_reviews(file_name):
            global anime_ix_map
            nonlocal evaluated_animes
            evaluated_animes = {}
            file_path = os.path.join(ROOT_DIR, "reviews", file_name + ".json")
            if not os.path.exists(file_path):
                return gr.Textbox.update(value=""), gr.DataFrame.update(value=[])
            with open(file_path, encoding="utf-8") as f:
                d = json.load(f)
            reviews = []
            for l in d:
                ix = anime_ix_map[l["anime_id"]]
                evaluated_animes[ix] = int(l["score"])
                reviews.append(
                    [
                        int(anime_df["anime_ix"].values[ix]),
                        anime_df["anime_id"].values[ix],
                        anime_df["japanese_name"].values[ix],
                        anime_df["type"].values[ix],
                        int(l["score"])
                    ]
                )
            return gr.Textbox.update(value=file_name), gr.DataFrame.update(value=reviews)

        def save_reviews(file_name):
            global anime_ix_map
            nonlocal evaluated_animes
            reviews = []
            for k, v in evaluated_animes.items():
                reviews.append(
                    {
                        "anime_id": anime_df["anime_id"].values[k],
                        "score": int(v)
                    }
                )
            file_path = os.path.join(ROOT_DIR, "reviews", file_name + ".json")
            with open(file_path, encoding="utf-8", mode="w") as f:
                json.dump(reviews, f)
            return gr.Dropdown.update(choices=get_saved_review_list())

        def reset_reviews():
            nonlocal evaluated_animes
            evaluated_animes = {}
            return gr.DataFrame.update(value=[])

        with gr.Group():
            gr.Markdown(value="評価するアイテムの検索")

            with gr.Row().style(equal_height=False):
                num_search = gr.Slider(5, 30, value=10, step=1, label="検索する数")
                search_text = gr.Textbox(value="ここに検索する文字列を入力してください", label="検索する文字列")

            with gr.Row():
                search_button = gr.Button("文字列で検索する", variant="primary")
                recommend_button = gr.Button("レビュー済みからレコメンドする", variant="primary")

            with gr.Row():
                searched_dataframe = gr.DataFrame(headers=["anime_ix", "anime_id", "japanese_name", "type", "genres"], label="検索結果")

        with gr.Group():
            gr.Markdown(value="アイテムの評価の登録")

            with gr.Row():
                selected_anime = gr.Dropdown([], label="レビューするアニメ")
                review = gr.Slider(0, 10, value=5, step=1, label="レビュー(0で評価を削除、1~10で評価)")

            with gr.Row():
                review_button = gr.Button("レビューを登録する", variant="primary")

        with gr.Group():
            gr.Markdown(value="評価済みのアニメ")
            with gr.Row():
                load_name = gr.Dropdown(choices=get_saved_review_list(), label="保存するファイル名(拡張子無し)")
                save_name = gr.Textbox(value="reviewed", label="保存するファイル名(拡張子無し)")
            with gr.Row():
                load_button = gr.Button("ロードする", variant="primary")
                save_button = gr.Button("セーブする", variant="primary")
                reset_button = gr.Button("リセットする", variant="secondary")
            with gr.Row():
                registered_dataframe = gr.DataFrame(headers=["anime_id", "japanese_name", "score"], label="レビューするアニメ")

        search_button.click(
            search_by_title,
            inputs=[
                search_text,
                num_search,
            ],
            outputs=[
                selected_anime,
                searched_dataframe
            ],
            queue=True
        )
        recommend_button.click(
            recommend_by_reviewed,
            inputs=[num_search],
            outputs=[
                selected_anime,
                searched_dataframe
            ],
            queue=True
        )
        review_button.click(
            register_review,
            inputs=[
                selected_anime,
                review
            ],
            outputs=[registered_dataframe],
            queue=True
        )
        load_button.click(
            load_reviews,
            inputs=[load_name],
            outputs=[save_name, registered_dataframe],
            queue=True
        )
        save_button.click(
            save_reviews,
            inputs=[save_name],
            outputs=[load_name],
            queue=True
        )
        reset_button.click(
            reset_reviews,
            outputs=[registered_dataframe],
            queue=True
        )
