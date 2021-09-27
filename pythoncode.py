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

### NeuralNetwork
from sklearn.model_selection import train_test_split
from tensorflow import keras
# train_test_split
train, test = train_test_split(df, test_size = 0.2)
# df는 마지막 열에 Y값을 가지는 dataframe

## make NN model
NN = keras.Sequential()
NN.add(keras.layers.Input(shape = (train.shape[1]-1)))  # 종속변수 Y를 제외한 변수 개수를 input size로 설정
NN.add(keras.layers.Dense(11, activation = "relu"))     # hidden layer의 node 개수는 input변수의 개수와 output 변수의 개수의 중간 값으로 설정
NN.add(keras.layers.Dense(1, activation = None))        # output변수. 연속형이므로 activation function을 설정하지 않습니다.

Adam = keras.optimizers.Adam(learning_rate = 0.001)     # optimizer는 Adam, learning_rate는 0.001

NN.compile(optimizer = Adam, loss = 'mse', metrics = ['mse'])           # mse를 기준으로 모델을 평가/학습합니다.

# model fitting & evaluation
NN.fit(train.iloc[:,:-1],train.iloc[:,-1], epochs = 30, batch_size=10)  # batch_size = 10으로 설정하여 한번에 10개 row씩 학습됩니다.
                                                                        # epochs = 30으로 설정하여 train set을 30번 학습시킵니다.

NN.evaluate(test.iloc[:,:-1], test.iloc[:,-1])                          # test set에서의 성능을 확인입니다.

### GESD anomaly detection
import scipy.stats as stats
import matplotlib.pyplot as plt

# GESD는 통계량의 평균으로부터 가장 멀리 떨어진 점까지의 거리를 표준편차로 나눈 R(i)값을
# t-distribution에서 유도되는 t값에 관한 식(lambda(i))과 비교하여 이상치로 분류하는 기법입니다.
# (밑에 작성된 함수들에 포함된 '#'을 제거하면 print문을 통해 GESD 진행과정을 볼 수 있습니다.)
 
# test_stat = R(i)를 계산하는 함수입니다.
def test_stat(y, iteration):
    std_dev = np.std(y)
    avg_y = np.mean(y)
    abs_val_minus_avg = abs(y - avg_y)
    max_of_deviations = max(abs_val_minus_avg)
    max_ind = np.argmax(abs_val_minus_avg)
    cal = max_of_deviations/ std_dev
    #print('Test {}'.format(iteration))
    #print("Test Statistics Value(R{}) : {}".format(iteration,cal))
    return cal, max_ind

# critical_value = lambda(i)를 계산하는 함수입니다.
def calculate_critical_value(size, alpha, iteration):
    t_dist = stats.t.ppf(1 - alpha / (2 * size), size - 2)
    numerator = (size - 1) * np.sqrt(np.square(t_dist))
    denominator = np.sqrt(size) * np.sqrt(size - 2 + np.square(t_dist))
    critical_value = numerator / denominator
    #print("Critical Value(λ{}): {}".format(iteration, critical_value))
    return critical_value

# R(i)와 lambda(i)를 비교하는 과정을 print해주는 함수입니다.
def check_values(R, C, inp, max_index, iteration):
    if R > C:
        print('{} is an outlier. R{} > λ{}: {:.4f} > {:.4f} \n'.format(inp[max_index],iteration, iteration, R, C))
    else:
        print('{} is not an outlier. R{}> λ{}: {:.4f} > {:.4f} \n'.format(inp[max_index],iteration, iteration, R, C))

# ESD 테스트 함수입니다.
# 입력으로 Anomaly detection을 진행할 input_series와 유의확률 alpha, max_outliers 개수가 들어갑니다.
# 출력으로 out_lier 개수가 나옵니다.
def ESD_Test(input_series, alpha, max_outliers):
    stats = []
    critical_vals = []
    k = 0
    for iterations in range(1, max_outliers + 1):
        stat, max_index = test_stat(input_series, iterations)
        critical = calculate_critical_value(len(input_series), alpha, iterations)
        #check_values(stat, critical, input_series, max_index, iterations)
        input_series = np.delete(input_series, max_index)
        critical_vals.append(critical)
        stats.append(stat)
        if stat > critical:
            max_i = iterations
            k += 1
    #print('H0:  there are no outliers in the data')
    #print('Ha:  there are up to 10 outliers in the data')
    #print('')
    #print('Significance level:  α = {}'.format(alpha))
    #print('Critical region:  Reject H0 if Ri > critical value')
    #print('Ri: Test statistic')
    #print('λi: Critical Value')
    #print(' ')
    df = pd.DataFrame({'i' :range(1, max_outliers + 1), 'Ri': stats, 'λi': critical_vals})
    
    def highlight_max(x):
        if x.i == max_i:
            return ['background-color: yellow']*3
        else:
            return ['background-color: white']*3
    df.index = df.index + 1
    #print('Number of outliers {}'.format(max_i))
    
    return  k

ESD_Test(np.array(y),0.05,5)
# 유의확률 0.05 하에서 최대 5개까지 detecting