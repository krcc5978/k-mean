import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


def iris_data_load():
    iris = datasets.load_iris()
    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )
    df["label"] = iris.target
    return df


def kmeans(k, X, max_iter=300):
    X_size, n_features = X.shape

    # ランダムに重心の初期値を初期化
    centroids = X[np.random.choice(X_size, k)]

    # 前の重心と比較するために、仮に新しい重心を入れておく配列を用意
    new_centroids = np.zeros((k, n_features))

    # 各データ所属クラスタ情報を保存する配列を用意
    cluster = np.zeros(X_size)

    # ループ上限回数まで繰り返し
    for epoch in range(max_iter):

        # 入力データ全てに対して繰り返し
        for i in range(X_size):
            # データから各重心までの距離を計算（ルートを取らなくても大小関係は変わらないので省略）
            distances = np.sum((centroids - X[i]) ** 2, axis=1)

            # データの所属クラスタを距離の一番近い重心を持つものに更新
            cluster[i] = np.argsort(distances)[0]

        # すべてのクラスタに対して重心を再計算
        for j in range(k):
            new_centroids[j] = X[cluster == j].mean(axis=0)

        # もしも重心が変わっていなかったら終了
        if np.sum(new_centroids == centroids) == k:
            print("break")
            break
        centroids = new_centroids
    return cluster


if __name__ == '__main__':
    df = iris_data_load()

    input_data = df.iloc[:, :-1].values

    cluster = kmeans(3, input_data)

    df["cluster"] = cluster
    df.plot(kind="scatter", x=0, y=1, c="label", cmap="winter")  # cmapで散布図の色を変えられます。
    plt.title("true label")
    plt.show()

    df.plot(kind="scatter", x=0, y=1, c="cluster", cmap="winter")
    plt.title("clustering relust")

    plt.show()