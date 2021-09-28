import os
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf

os.chdir("C:/Users/Kyeongjun/Desktop/Temperature")
trainData = pd.read_csv("train.csv")
trainData.columns = ["id", "temp_1", "press_1", "windvel_1", "windvel_2", "rain_1", "seapress_1",
                     "press_2", "temp_2", "seapress_2", "seapress_3", "rain_2", "sun_1", "humidity_1",
                     "winddirect_1", "sun_2", "winddirect_2", "sun_3", "winddirect_3", "windvel_3", "sun_4",
                     "humidity_2", "rain_3", "press_3", "seapress_4", "windvel_4", "winddirect_4", "windvel_5",
                     "press_4", "temp_3", "press_5", "humidity_3", "temp_4", "temp_5", "seapress_5", "sun_5",
                     "winddirect_5", "rain_4", "humidity_4", "humidity_5", "rain_5", "target_00", "target_01",
                     "target_02", "target_03", "target_04", "target_05", "target_06", "target_07", "target_08",
                     "target_09", "target_10", "target_11", "target_12", "target_13", "target_14", "target_15",
                     "target_16", "target_17", "target_18"]

Data0_17 = trainData[:4752-432]  # Target_18이 결측값인 데이터셋
Data18 = trainData[4752-432:]    # Target_18이 결측값이 아닌 데이터셋

del(trainData)

### scoring function
def daconMSE(y_true, y_pred):   
    diff = tf.math.abs(y_true - y_pred)    
    less_than_one = tf.where(diff < 1, 0.0, diff)    
    score = tf.math.reduce_mean(tf.math.square(less_than_one))    
    return score

### Feature Subsetting
# 위에서 나눈 데이터를 바탕으로 입력 변수와 출력 변수를 찢었습니다.
feature0_17 = Data0_17.iloc[:, 1:-19]
target0_17 = Data0_17.iloc[:, -19:-1]
feature18 = Data18.iloc[:, 1:-19]
target18 = Data18["target_18"]

del(Data0_17, Data18)

### time series split
# 시계열 데이터의 특성을 살려 train과 test를 쪼겠습니다.
# 처음 30일 데이터는 train : 20일 | test : 10일
# 마지막 3일 데이터는 train : 2일 | test : 1일

trainFeature0_17 = feature0_17.iloc[:3024, :]
testFeature0_17 = feature0_17.iloc[3024:, :]

trainFeature18 = feature18.iloc[:288, :]
testFeature18 = feature18.iloc[288:, :]

trainTarget0_17 = target0_17.iloc[:3024, :]
testTarget0_17 = target0_17.iloc[3024:, :]

trainTarget18 = target18.iloc[:288]
testTarget18 = target18.iloc[288:]

del(feature0_17, target0_17, feature18, target18)

### Multi-Layer Perceptron
# 일반 신경망 모델입니다. 모델을 구축하고 마지막 summary() 메소드로 모델의 모양을 확인하실 수 있습니다.
# keras는 신경망을 훈련시킬 때 사용할 데이터의 수를 정할 수 있습니다. 이는 아래 fit() 메소드에서 batch_size로 전달합니다.
# 따라서 모델 구축 간 신경망에 알려줘야 할 사항은 노드의 개수, 층의 개수, 활성화 함수의 종류, 최적화 기법 등 입니다.
# 아래 모델들은 전이학습을 위한 모델들입니다. 전이학습은 pre-training과 fine-tuning 단계로 나눌 수 있습니다.
# target18을 예측하기 위해 target00~target17의 데이터로 신경망을 사전훈련 시킵니다.

### pre-training
preTrainMLP = keras.Sequential()                                                                       # Keras는 Sequential()을 이용해 최초 망을 생성할 객체를 만들고, 이를 add() 메소드로 순차적으로 쌓을 수 있습니다.
preTrainMLP.add(keras.layers.Input(shape = (trainFeature0_17.shape[1])))                               # 입력층을 정의합니다. 현재 파일에서는 40개의 입력변수 전부를 사용하기 때문에 입력층의 노드 수도 40개입니다.
preTrainMLP.add(keras.layers.Dense(512, activation = "relu"))
preTrainMLP.add(keras.layers.Dense(128, activation = "relu"))
preTrainMLP.add(keras.layers.Dense(32, activation = "relu"))
preTrainMLP.add(keras.layers.Dense(trainTarget0_17.shape[1], activation = None))                       # 출력층을 정의합니다. 해당 신경망은 target_00~target_17을 예측하는 모델입니다. 따라서 출력층의 노드 수는 18개입니다.
                                                                                                       # 연속형 목표변수를 예측하기 때문에 활성화 함수는 없습니다.
                                                                                                       # 전이학습에 사용할 신경망이기 때문에 아래 fine-tuning 단계에서 출력층 제거할 예정입니다.
preTrainMLP.summary()                                                                                  # 위 과정으로 생성한 신경망의 개괄적인 전체 모양을 확인합니다.
Adam = keras.optimizers.Adam(learning_rate = 0.001)                                                    # 사용할 gradient descent 기법을 정의합니다. keras.optimizers 내에 수많은 기법이 있습니다. 해당 기법에서의 learning rate 0.001입니다.

preTrainMLP.compile(optimizer = Adam, loss = 'mse', metrics = ['mse'])                                 # 위의 Sequential 모델에 gradient descent 기법과 손실함수, 평가 지표를 정의합니다. 저희 문제에서는 mse를 사용하여 손실함수와 평가지표는 mse를 사용했습니다.

preTrainMLP.fit(trainFeature0_17, trainTarget0_17, epochs = 10, batch_size = 1)                     # 데이터를 바탕으로 신경망을 훈련시킵니다. 해당 신경망은 전체 4320개의 데이터 중 144개씩 데이터를 가져와 훈련합니다.
                                                                                                       # 4320개의 데이터를 바탕으로 훈련하는 횟수를 epochs로 정할 수 있습니다.
preTrainMLP.evaluate(testFeature0_17, testTarget0_17)                                                  # target00~target17을 예측하도록 훈련된 신경망에 대해 새로운 데이터를 넣어 예측하게 만들고, 이의 오차를 확인합니다.


preTrainMLP_json = preTrainMLP.to_json()                                                               # 만일 사전 학습이 잘 된 망이라 판단되신다면 파일로 저장 가능합니다. working directory에 저장됩니다.
with open("preTrainMLP.json", 'w') as json_file :
    json_file.write(preTrainMLP_json)


# Load Model : 위의 json 파일로 저장한 모델을 불러옵니다.
# json_file = open("preTrainMLP.json", 'r')
# loaded_model = json_file.read()
# json_file.close()
# MLP_Loaded = keras.models.model_from_json(loaded_model)


### fine-tuning
fineTuning = keras.models.Model(inputs = preTrainMLP.input, outputs = preTrainMLP.layers[-2].output)   # pre-trainig된 신경망에서 출력층을 떼어내 가져옵니다.
fineTuning.summary()
#for layer in fineTuning.layers :                                                                       # fine-tuning을 위해 pre-training된 신경망의 입력층과 은닉층의 가중치들을 학습하지 않게 합니다.
#    layer.trainable = False
fineTuningMLP = keras.Sequential()                                                                     # target_18을 예측하기 위한 모델을 만듭니다.
fineTuningMLP.add(fineTuning)                                                                          # 해당 모델은 pre-training 모델에서 출력층을 제외한 망 + 새로운 은닉층 1개 + target_18을 예측하기 위한 새로운 출력층 1개로 구성되어있습니다.
fineTuningMLP.add(keras.layers.Dense(1, activation = None))
fineTuningMLP.summary()

Adam = keras.optimizers.Adam(learning_rate = 0.001)

fineTuningMLP.compile(optimizer = Adam, loss = daconMSE)

fineTuningMLP.fit(trainFeature18, trainTarget18, epochs = 60, batch_size = 1)
fineTuningMLP.evaluate(testFeature18, testTarget18)                                                    # 전이학습으로 생성한 모델의 테스트 데이터에 대한 최종 성능을 평가합니다.

fineTuningMLP_json = fineTuningMLP.to_json()
with open("fineTuningMLP.json", 'w') as json_file :
    json_file.write(fineTuningMLP_json)

### final testing
# dacon에 제출할 파일을 만듭니다. 해당 파일은 working directory에 저장됩니다. 
test = pd.read_csv("test.csv")
result = pd.DataFrame({"id" : list(range(4752, 16272))})
result["id"] = test["id"]
test = test.iloc[:, 1:]

output = fineTuningMLP.predict(test)

result["Y18"] = output

result.to_csv("C:/Users/Kyeongjun/Desktop/Temperature/three.csv", header = True, index = False)

out = pd.DataFrame({"id" : list(range(4608, 4752))})
out["Y18"] = fineTuningMLP.predict(testFeature18)

out.to_csv("C:/Users/Kyeongjun/Desktop/Temperature/out.csv", header=True, index = False)

### Long Short Term Memory (LSTM)
# Modify Data to use LSTM

# 위의 MLP를 이용하지 않고, LSTM만으로 신경망을 생성하였습니다.
# LSTM은 RNN의 종류 중 하나로, CNN이 이미지 처리에 특화된 신경망인 것 처럼, LSTM은 시계열 데이터 처리에 능숙하다고 알려져있습니다.
# keras의 LSTM은 데이터를 [batch size, time step, 변수의 개수]로 받아 계산합니다. 따라서 LSTM에 넘겨줄 데이터의 형태는 3차원입니다.
# batch size : 신경망을 한 번 학습할 때 사용할 데이터의 개수입니다.
# time step : 간단하게 요약하면 LSTM에서 한 번에 처리 또는 기억할 데이터의 개수입니다. 저는 144로 설정해 하루 단위로 돌아가도록 했습니다.
# time step은 원래 데이터의 개수와 설정할 time step을 나누었을 때 나누어 떨어지도록 설정해야 합니다.

# 이론적으로 LSTM이 MLP보다 해당 데이터에서 성능이 더 좋아야 할 텐데, 현재 작성된 코드에서는 MLP가 성능이 더 좋은 편입니다.

time_step = 144
trainFeature0_17 = np.array(trainFeature0_17).reshape(-1, time_step, trainFeature0_17.shape[1])        # 40개의 변수를 가진 144개의 데이터가 21개 존재 : 21일간의 데이
trainFeature18 = np.array(trainFeature18).reshape(-1, time_step, trainFeature18.shape[1])
testFeature0_17 = np.array(testFeature0_17).reshape(-1, time_step, testFeature0_17.shape[1])
testFeature18 = np.array(testFeature18).reshape(-1, time_step, testFeature18.shape[1])

trainTarget0_17 = np.array(trainTarget0_17).reshape(-1, time_step, trainTarget0_17.shape[1])
trainTarget18 = np.array(trainTarget18).reshape(-1, time_step, 1)
testTarget0_17 = np.array(testTarget0_17).reshape(-1, time_step, testTarget0_17.shape[1])
testTarget18 = np.array(testTarget18).reshape(-1, time_step, 1)

                                                                                                       # keras에서 신경망을 만드는 방법은 Sequential()로 만드는 법과, layer들을 미리 쌓아두고 이를 신경망 모델로 정의하는 방법이 있습니다.
### pre-training                                                                                       # 해당 방법은 layer를 쌓아 서로 연결하고, 이를 신경망 모델로 정의합니다.
inputs = keras.layers.Input(shape = (time_step, trainFeature0_17.shape[2]))                            # LSTM의 입력층은 [time step, 변수의 개수]로 정의됩니다.
LSTM_layer_1 = keras.layers.LSTM(64, return_sequences = True)(inputs)                                  # return_sequences 인자는 LSTM 층을 여러개 쌓을 때 필요합니다.
LSTM_layer_2 = keras.layers.LSTM(128, return_sequences = True)(LSTM_layer_1)
LSTM_layer_3 = keras.layers.LSTM(64, return_sequences = True)(LSTM_layer_2)
outputs = keras.layers.Dense(trainTarget0_17.shape[2], activation = None)(LSTM_layer_3)

preTrainLSTM = keras.Model(inputs = inputs, outputs = outputs)                                         # keras.Model()로 연결한 층들을 모델로 만듭니다.
preTrainLSTM.summary()

Adam = keras.optimizers.Adam(learning_rate = 0.001)

preTrainLSTM.compile(optimizer = Adam, loss = "mse")

preTrainLSTM.fit(trainFeature0_17, trainTarget0_17, epochs = 100, batch_size = 1)
preTrainLSTM.evaluate(testFeature0_17, testTarget0_17)

preTrainLSTM_json = preTrainLSTM.to_json()
with open("preTrainLSTM.json", 'w') as json_file :
    json_file.write(preTrainLSTM_json)


### fine-tuning
fineTuning = keras.models.Model(inputs = preTrainLSTM.input, outputs = preTrainLSTM.layers[-2].output)
fineTuning.summary()

for layer in fineTuning.layers :
    layer.trainable = False
fineTuningLSTM = keras.Sequential()
fineTuningLSTM.add(fineTuning)
fineTuningLSTM.add(keras.layers.LSTM(32, return_sequences = True))
fineTuningLSTM.add(keras.layers.Dense(1, activation = None))
fineTuningLSTM.summary()

Adam = keras.optimizers.Adam(learning_rate = 0.001)

fineTuningLSTM.compile(optimizer = Adam, loss = "mse")

fineTuningLSTM.fit(trainFeature18, trainTarget18, epochs = 500, batch_size = 1)
fineTuningLSTM.evaluate(testFeature18, testTarget18)

fineTuningLSTM_json = fineTuningLSTM.to_json()
with open("fineTuningLSTM.json", 'w') as json_file :
    json_file.write(fineTuningLSTM_json)

model = keras.Sequential()
for i in range(2):
    model.add(keras.layers.LSTM(32, batch_input_shape=(1, 144, 1), stateful=True, return_sequences=True))
    model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.LSTM(32, batch_input_shape=(1, 144, 1), stateful=True))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(1))

model.summary()
