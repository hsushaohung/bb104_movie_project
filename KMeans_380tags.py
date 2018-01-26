from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
import json
import pickle
import pandas as pd
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

with open('C:\\Users\\Java\\分群\\詞庫\\movietags_v1', 'rb') as f:
    comments = pickle.load(f)

documents_old = []
for i in range(len(comments)):
    single_movie = ""
    for j in range(len(comments[i]['tags'])):
        single_movie += comments[i]['tags'][j] + " "
    documents_old.append(single_movie)

vectorizer = TfidfVectorizer(stop_words = None)
X = vectorizer.fit_transform(documents_old)

df = pd.DataFrame(X.toarray())

distortions = []

for i in range(1, 300):
    km = MiniBatchKMeans(
        n_clusters=i,
        init='k-means++',
        n_init=1,  # 1, 2, 3, 5
        #         max_iter = 125,   #50000, 40000, 30000, 20000, 10000, 5000, 2500, 1000, 500, 250 重點不在這裡!
        init_size=1500,  # 550, 1000角度變小, 2000差不多, 300越往尾端會有越大波動
        batch_size=500,  # 1000,1500
        random_state=0)
    km.fit(df)
    distortions.append(km.inertia_)

plt.plot(range(1, 300), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()