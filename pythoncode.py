### k-means clustering

from sklearn.cluster import KMeans

data_points = df.values
kmeans = KMeans(n_clusters=8).fit(data_points) # 클러스터 개수 8
kmeans.fit(df)

kmeans.labels_ # 각각의 data(row)들이 어떤 클러스터로 분류되었는지 list로 출력

### spectral clustering (Laplacian)

from sklearn.cluster import SpectralClustering

# 5-nn 방식을 이용하여 5개의 클러스터로 구분
spectralclustering = SpectralClustering(n_clusters=5, n_neighbors=5, affinity='nearest_neighbors')

spectralclustering.fit_predict(df) # 각각의 data(row)들이 어떤 클러스터로 분류되었는지 list로 출력

### PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=3) # 3개의 클러스터로 구분

components = pca.fit_transform(df) # principal components values
pcadf = pd.DataFrame(data = components, columns = ['pc1', 'pc2','pc3']) # principal component dataframe

pca.explained_variance_ratio_      # principal component들의 분산 (설명력)
sum(pca.explained_variance_ratio_) # PCA component들을 이용하여 전체 분산(데이터)을 설명하는 정도

pca.components_                    # principal component별로 pca에 사용된 변수들의 coefficient를 보여줌

### t-sne dimension reduction visualization

from sklearn.manifold import TSNE

n_components = 2 # 2차원으로 시각화

model = TSNE(n_components = n_components, perplexity = 30, learning_rate = 200)
# perplexity는 5~50 사이의 값으로 그려보며 최적값을 찾음
# learning_rate 10~1000 사이의 값으로 그려보며 최적값을 찾음

x, y = [],[]
k = model.fit_transform(df)  # x, y values of dimension reducted data
for i in range(len(df)) :
    x.append(k[i][0])
    y.append(k[i][1])

plt.scatter(x,y)

# t-sne 시각화는 실행할 때마다 다르게 나오기 때문에 x,y값을 저장해서 사용했음
model_df = pd.DataFrame(list(zip(x,y)), columns = ['x','y'])
model_df.to_csv('tsne.csv', header=True, index=False)

### umap dimension reduction visualization

import umap.umap_ as umap  #pip install 할 때 모듈 이름이 umap이 아니었음.

reducer = umap.UMAP()

embedding = reducer.fit_transform(df)

plt.scatter(embedding[:,0], embedding[:,1],
            #c=[sns.color_palette()[x] for x in df_cluster.cluster_id] # cluster별로 다른 색깔을 표시
           )
plt.gca().set_aspect('equal','datalim')