import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import scipy.stats as stats

"""
### 과정 설명
1. PCA를 적용할 시퀀스를 변수로 가지는 데이터프레임을 준비합니다.
2. 데이터프레임과 변수명을 입력하면 해당 변수들로 PCA를 하여 2개의 컴포넌트를 출력하는 함수를 생성압니다.
3. 시퀀스, 유의확률, 최대이상치 개수를 입력하면 GESD anomaly detection을 통해 이상치 개수를 출력하는 함수를 생성합니다.
3. 데이터프레임을 함수에 입력하여 원하는 변수들의 컴포넌트를 출력합니다.
4. 해당 컴포넌트를 이용하여 pc1, pc2를 축으로 하는 시각화를 작성합니다.
5. GESD 함수를 이용하여 outlier를 찾아내어 해당하는 점에 빨간색으로 표시합니다.
"""

## 2개 변수 PCA로 2개 컴포넌트 만드는 함수 (과정 2)
# df : 데이터프레임 , col1,2 : column이름 1,2(string)
def PCAdf(df, col1, col2) :
    pca = PCA(n_components=2)                                                # 2개의 컴포넌트를 생성합니다.
    components = pca.fit_transform(df.loc[:,[col1,col2]])                    # 입력에 사용된 col1, col2 변수에 대해 PCA를 진행합니다.
    returndf = pd.DataFrame(data = components, columns = ['pc1', 'pc2'])     # PCA 결과로 나온 principal component 1,2를 pc1,2로하는 데이터프레임 생성
    return returndf

## define function of GESD Anomaly detection (과정 3)
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

## 데이터프레임과 시퀀스 변수명을 입력하면 GESD anomaly detection으로 outlier 데이터프레임을 출력하는 함수를 정의합니다.
def outlier_df(df,var) :
    # 1. ESD_Test함수로 outlier 개수(n)를 파악한 뒤,
    # 2. var(시퀀스 변수명)을 기준으로 평균과 가장 멀리 떨어진 n개의 index를 파악
    # 3. 해당 index의 시퀀스 위치와 value를 dataframe으로 생성
    # 4. 시퀀스 위치와 value는 그래프에 이상치를 색칠할 때 x,y값으로 사용됩니다.
    n = ESD_Test(np.array(df[var]), 0.05, len(df))
    df0 = df.copy(deep=True)
    df0['seq_loc'] = range(len(df0))
    list(pd.to_numeric(df0['seq_loc']))
    df0['dist'] = abs(np.mean(df0[var]) - df0[var])
    ind = list(df0.sort_values(by='dist', ascending = False).iloc[:n,:].index)
    df0 = df0.loc[ind,['seq_loc',var]]
    
    return df0

df = pd.read.csv('<파일명.csv>')     # 데이터프레임
df_pca = PCAdf(df,'<col1>','<col2>') # <col1,2>에 컬럼명이 들어갑니다.

# joint outlier index
#(joint control region에는 pc1, pc2에서 검출된 outlier가 겹치기 때문에 unique한 값만 추출하기 위해 이상치들의 index에서 겹치는 값을 하나로 줄여줍니다.)
a = list(outlier_df(df_pca,'pc1').index)
b = list(outlier_df(df_pca,'pc2').index)
a.extend(b)
joint_ind = list(set(a))

del a,b
 
## 시각화
# joint control region
figure(figsize=(5, 5), dpi=80) # 시각화의 사이즈를 키웁니다. (생략가능)
plt.scatter(df_pca['pc1'], df_pca['pc2'], s=10) # pc1,2를 x,y축으로 scatter plot을 그립니다.
plt.plot([min(df_pca.iloc[:,0]), max(df_pca.iloc[:,0])], [np.mean(df_pca.iloc[:,0]),np.mean(df_pca.iloc[:,0])], 'k-', lw=2) # pc1의 평균으로 center line을 그립니다.
plt.plot([np.mean(df_pca.iloc[:,1]),np.mean(df_pca.iloc[:,1])], [min(df_pca.iloc[:,1]), max(df_pca.iloc[:,1])], 'k-', lw=2) # pc2의 평균으로 center line을 그립니다.
plt.scatter(list(df_pca.loc[joint_ind,].iloc[:,0]),list(df_pca.loc[joint_ind,].iloc[:,1]),s=30, color ='r') # GESD로 outlier에 빨간색으로 표시합니다.

# pc1의 시각화
figure(figsize=(10, 5), dpi=80) # 시각화의 사이즈를 키웁니다. (생략가능)
plt.plot(range(df_pca.shape[0]), df_pca['pc1'], marker = 'o', markersize = 3) # 시퀀스 순서와 pc1값을 x,y축으로 line graph를 그립니다.
plt.plot([0, df_pca.shape[0]], [np.mean(df_pca.iloc[:,1]),np.mean(df_pca.iloc[:,1])], 'k-', lw=2) # center line of pc1
plt.scatter(list(outlier_df(df_pca,'pc1').iloc[:,0]),list(outlier_df(df_pca,'pc1').iloc[:,1]),s=30, color ='r') # GESD로 outlier에 빨간색으로 표시합니다.

# pc2의 시각화
figure(figsize=(10, 5), dpi=80) # 시각화의 사이즈를 키웁니다. (생략가능)
plt.plot(range(df_pca.shape[0]), df_pca['pc2'], marker = 'o', markersize = 3) # 시퀀스 순서와 pc2값을 x,y축으로 line graph를 그립니다.
plt.plot([0, df_pca.shape[0]], [np.mean(df_pca.iloc[:,1]),np.mean(df_pca.iloc[:,1])], 'k-', lw=2) # center line of pc2
plt.scatter(list(outlier_df(df_pca,'pc2').iloc[:,0]),list(outlier_df(df_pca,'pc2').iloc[:,1]),s=30, color ='r') # GESD로 outlier에 빨간색으로 표시합니다.
