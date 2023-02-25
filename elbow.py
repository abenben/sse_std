from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# エルボー法
def elbow(X):
    sse = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=0)
        km.fit(X)
        sse.append(km.inertia_)
    plt.plot(range(1, 11), sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()

# 期待値を計算する
def calc_expectation(X, km, k):
    # クラスタリング
    km.fit(X)
    # SSEを計算
    sse = km.inertia_
    # サンプル数
    n = len(X)
    # 次元数
    d = X.shape[1]
    # 期待値を計算
    e = sse + (d*k*(np.log(n))) / (2*n)
    return e

# データの読み込み
X = np.loadtxt('data.csv',delimiter=',',dtype='float32',usecols=[0,1])

elbow(X)

# エルボー点を求める
max_k = 10
mse = []
for k in range(1, max_k+1):
    km = KMeans(n_clusters=k, random_state=0)
    mse.append(calc_expectation(X, km, k))
elbow_point = np.argmin(np.diff(mse)) + 1

# エルボー点を表示
plt.plot(range(1, max_k+1), mse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Expectation')
plt.axvline(elbow_point, color='gray', linestyle='--')
plt.show()
print('Elbow point:', elbow_point)
