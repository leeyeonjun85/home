---
title: "Penguins 인공신경망"
excerpt: "Penguins 데이터로 인공신경망 구성하기"

categories:
  - Dev Log

tags:
  - 개발일지
  - 코딩
  - keras
  - Neural Network

use_math: true

header:
  teaser: /assets/images/aib/artificial-intelligence-3685928_1920.png

last_modified_at: 2023-02-21
---


<br><br>


![image]({{ site.url }}{{ site.baseurl }}/assets/images/etc/penguin_png.png){: .align-center width="70%"}  


<br><br>


<br><br>


# Penguins, 인공신경망 구성  


<br><br>


## 도입  
>Penguins 데이터로 인공신경망 만들기  
인공신경망을 공부하는MNIST를 자주 사용하곤 하지만, MNIST의 index는 60,000개로 사이즈가 크기 때문에 학습에 시간이 오래 걸린다.  
그래서 데이터 사이즈가 작고 귀여운 Penguins 데이터로 인공신경망을 연습하자


<br><br>


## 데이터 전처리  

### 데이터 읽어오기  

```python
# Data Load
penguins = sns.load_dataset('penguins')
print(f"🎬 Rows, Columns : {penguins.shape}")
# 🎬 Rows, Columns : (344, 7)
```

- 데이터는 Seaborn에서 불러왔다.
- 데이터에 대한 자세한 설명은 [Seaborn 링크 참조](https://github.com/allisonhorst/palmerpenguins) 해주세요

<br>

### 특성공학

- 특별한 특성공학은 하지 않음
- sex가 결측인 경우 3으로 대체
- 다른 특성은 평균으로 대체

```python
penguins_species = {
    'Adelie' : 0,
    'Gentoo' : 1,
    'Chinstrap' : 2,
}
penguins['species'] = penguins['species'].map(penguins_species)

penguins_island = {
    'Biscoe' : 0,
    'Dream' : 1,
    'Torgersen' : 2,
}
penguins['island'] = penguins['island'].map(penguins_island)

penguins_sex = {
    'Male' : 0,
    'Female' : 1,
    np.nan : 3,
}
penguins['sex'] = penguins['sex'].map(penguins_sex)

# NaN = Fill with Mean
for idx, col in enumerate(penguins.columns):
  penguins[col].fillna(penguins[col].mean(), inplace=True)
```

<br>

### 데이터 분할

```python
test_size = 0.2
x_train, x_test, y_train, y_test = train_test_split(penguins[features], penguins[target], test_size=test_size, random_state=rand_seed)
print(f'✨Train Shape : {x_train.shape} / {y_train.shape}\n🎉 Test Shape : {x_test.shape} / {y_test.shape}')
#✨Train Shape : (275, 6) / (275, 1)
#🎉 Test Shape : (69, 6) / (69, 1)
```

<br>

### 정규화(Normalization)

```python
x_train = (x_train - np.min(x_train, axis='index')) / (np.max(x_train, axis='index') - np.min(x_train, axis='index'))
x_test  = ( x_test - np.min( x_test, axis='index')) / (np.max( x_test, axis='index') - np.min( x_test, axis='index'))
```


<br><br>


## 탐색적 데이터 분석

### Seaborn 의 데이터 소개로 대체

![image](https://github.com/allisonhorst/palmerpenguins/raw/main/man/figures/culmen_depth.png){: .align-center width="60%"}  


![image](https://seaborn.pydata.org/_images/introduction_29_0.png){: .align-center width="80%"}  


<br><br>


## 모델링

### 기준모델 : Logistic Regression

- 로지스틱 회귀분석모델을 기준모델로 하자!

```python
log_model = LogisticRegression()
log_model.fit(x_train, y_train)
log_score = print_accuracy(log_model)
log_CM, fig = draw_CM(log_model)
# Train Accuracy : 0.978, Test Accuracy : 0.971
```

- 기본모델 정확도가 무려 97.1% 😳 (큰일이다...)
- 보통은 기준모델 이상을 목표로 하지만 오늘은 기준모델 만큼이라도 달성하는 것을 목표로 하자

![image]({{ site.url }}{{ site.baseurl }}/assets/images/coding/penguins/logit_cm.png){: .align-center width="60%"}  

<br>

### 기본 인공신경망 : 은닉층 0개

- 가장 기본적인 인공신경망을 만들어보자

```python
model1 = tf.keras.Sequential([
    Dense(3, activation='softmax'),
    ])

model1.compile(metrics=['accuracy']
               , optimizer='adam'
               , loss='sparse_categorical_crossentropy'
               )

results1 = model1.fit(x_train, y_train
                      , epochs=5
                      , verbose=0
                      )

model1_score = print_accuracy(model1)
log_CM, fig = draw_CM(model1)
model1.summary()
```

- 은닉층 없이 출력증에만 펭귄종류 3가지만 지정하고, 다중분류 활성화함수인 softmax 지정
- 최소한의 학습을 위하여 epochs를 5로 지정
- 입력 특성 7개, 출력 타겟클래스 3개, Total 파라미터 21개
- 평가 정확도 53.6%...😭😢😞 (딥러닝... 버려야 하나?)

```
# Train Accuracy : 0.436, Test Accuracy : 0.536
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_37 (Dense)            (None, 3)                 21        
                                                                 
=================================================================
Total params: 21
Trainable params: 21
Non-trainable params: 0
_________________________________________________________________
```

![image]({{ site.url }}{{ site.baseurl }}/assets/images/coding/penguins/nn_base1.png){: .align-center width="60%"} 

<br>

### 기본 인공신경망 : 은닉층 1개

- 은닉층을 하나 만들어 보자

```python
model1 = tf.keras.Sequential([
    Dense(100, activation='relu', kernel_initializer='he_uniform'),
    Dense(3, activation='softmax'),
    ])

model1.compile(metrics=['accuracy']
               , optimizer='adam'
               , loss='sparse_categorical_crossentropy'
               )

results1 = model1.fit(x_train, y_train
                      , epochs=5
                      , verbose=0
                      )

model1_score = print_accuracy(model1)
log_CM, fig = draw_CM(model1)
model1.summary()
```

- 하나의 은닉층에 노드 100개, 활성화함수는 ReLU, 가중치 초기화는 'he_uniform'을 지정
- 은닉층이 생기면서 추정해야 할 파라미터가 1,003개로 늘었다.
- 평가 정확도 76.8%!!

```
# Train Accuracy : 0.807, Test Accuracy : 0.768
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_38 (Dense)            (None, 100)               700       
                                                                 
 dense_39 (Dense)            (None, 3)                 303       
                                                                 
=================================================================
Total params: 1,003
Trainable params: 1,003
Non-trainable params: 0
```

![image]({{ site.url }}{{ site.baseurl }}/assets/images/coding/penguins/nn_base2.png){: .align-center width="60%"} 

<br>

### 기본 인공신경망 : 은닉층 2개

- 일반적으로 딥러닝이라 하면 은닉층이 2개 이상을 의미한다.
- 은닉층을 2개 추가해보자

```python
model1 = tf.keras.Sequential([
    Dense(100, activation='relu', kernel_initializer='he_uniform'),
    Dense(100, activation='relu', kernel_initializer='he_uniform'),
    Dense(3, activation='softmax'),
    ])

model1.compile(metrics=['accuracy']
               , optimizer='adam'
               , loss='sparse_categorical_crossentropy'
               )

results1 = model1.fit(x_train, y_train
                      , epochs=5
                      , verbose=0
                      )

model1_score = print_accuracy(model1)
log_CM, fig = draw_CM(model1)
model1.summary()
```

- 똑같은 은닉층을 2개로 늘렸다
- 평가 정확도 95.7%🎉

```
# Train Accuracy : 0.971, Test Accuracy : 0.957
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_40 (Dense)            (None, 100)               700       
                                                                 
 dense_41 (Dense)            (None, 100)               10100     
                                                                 
 dense_42 (Dense)            (None, 3)                 303       
                                                                 
=================================================================
Total params: 11,103
Trainable params: 11,103
Non-trainable params: 0
```

![image]({{ site.url }}{{ site.baseurl }}/assets/images/coding/penguins/nn_base3.png){: .align-center width="60%"} 


- 인공신경망의 은닉층이 2개 이상으로 늘어나지깐 정확도가 급 상승했다


<br><br>


## 인공신경망 실험

### 실험1 : 은닉층의 수를 늘리면 성능이 오늘까?

- 은닉층 1개 ~ 99개 까지 모델을 학습하고 정확도를 비교해보자

![image]({{ site.url }}{{ site.baseurl }}/assets/images/coding/penguins/test_hiddens.png){: .align-center width="60%"} 

- 2개 이후로는 성능의 향상이 크게 없는 것 같다
- 앞으로는 효율성을 위하여 은닉층은 2개로 고정할 것이다

<br>

### 실험2 : 은닉층의 노드를 늘리면 성능이 오를까?

- 은닉층의 노드는 가중치와 편향의 가중합 연산이 일어나는 부분이다.

![image]({{ site.url }}{{ site.baseurl }}/assets/images/coding/penguins/test_node.png){: .align-center width="60%"} 

- 은닉층이 오르면서 성능도 오르고, 성능의 안정감도 오르는 것 같다.
- 노드 80 이후로는 큰 차이가 없어보인다.
- 앞으로는 노드를 100정도로 고정할 것이다

<br>

### 실험3 : 은닉층의 모양은 성능과 어떤 상관이 있을까?

- 2개 은닉층의 노드 비율을 1:99 , 2:98 , 3:97 ...... 98:2 , 99:1 로 변화시켜 보자

![image]({{ site.url }}{{ site.baseurl }}/assets/images/coding/penguins/test_shape.png){: .align-center width="60%"} 

- 2개 은닉층의 비율이 크게 차이나는 1:99, 2:98 ... 98:2, 99:1 은 성능 점수도 낮고, 안정감도 떨어지는 것 같다.
- 은닉층의 비율은 비슷한 것이 좋겠다.

<br>

### 최적 Layer 모델

- 지금까지의 실험을 토대로 딥러닝 모델을 만들어보자

```python
def get_model(nodes=100, lr=0.001):
    random.seed(rand_seed)
    model = tf.keras.Sequential([
                                Dense(nodes, activation='relu', kernel_initializer='he_uniform'),
                                Dense(nodes, activation='relu', kernel_initializer='he_uniform'),
                                Dense(3, activation='softmax'),
                                ])

    model.compile(metrics=['accuracy']
                  , optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
                  , loss='sparse_categorical_crossentropy'
                  )
    
    return model
```

- 앞으로의 실험은 아래의 모델을 중심으로 실행할 것이다

<br>

### 실험4 : epochs

- epoch는 인공신경망이 순전파와 역전파를 통하여 1회 학습이 일어나는 과정이다.
- 인공신경망에서는 역전파를 통하여 모델의 업데이트가 일어나기 때문에 epoch가 많아지면 모델이 정교해질 가능성이 있다.
- epoch를 1~30 까지 조정하며 정확도를 평가해보자

![image]({{ site.url }}{{ site.baseurl }}/assets/images/coding/penguins/test_epoch.png){: .align-center width="60%"}  

<br>

### 실험5 : batch_size

- 경사하강법을 통하여 손실함수를 계산할 때 전통적으로는 모든 입력데이터에 대하여 계산을 수행하였다.
- 데이터가 커지면서 시간이 오래걸리고 비효율성이 높아져 확률적 경사하강법, 곧 일부데이터를 뽑아서 손실함수를 계산하는 방법이 생겨났다.
- batch_size를 지정한면 그만큼 입력데이터를 뽑아 미니배치(Mini-batch) 경사하강법을 수행하게 된다.
- 이때 입력데이터 수, batch_size, iteration은 다음과 같은 관계가 성립된다.
- $Nobs. Data = batch\_size \times Iteration(by \; epoch)$
- 실험 결과 80 이하가 적당한 것 같다.

![image]({{ site.url }}{{ site.baseurl }}/assets/images/coding/penguins/test_batch_size.png){: .align-center width="60%"}  


<br><br>


## 학습률 계획

- 가장 중요한 하이퍼파라미터 가운데 하나가 바로 학습률(Learning Rate)이다.
- 인공신경망에서 가중치 업데이트는 다음과 같은 계산으로 수행된다.
- $\theta_{ i,j+1 } = \theta_{ i,j } - \eta { { \delta }\over{ \delta \theta_{ i,j } } }J(\theta)$
  - $\theta_{ i,j+1 }$ : 새롭게 생신된 가중치
  - $\theta_{ i,j }$ : 갱신 전 가중치
  - $\eta$ : 학습률
  - ${ { \delta }\over{ \delta \theta_{ i,j } } }J(\theta)$ : 해당 지점에서의 기울기 
- 그래서 학습률은 가중치 업데이트과정 가운데 가중치의 변화폭이라고 이해할 수 있다.
- 학습률은 직업 지정해줄 수도 있지만 주로 Step Decay, Cosine Decay의 두가지 방식이 자주 사용된다.

### 학습률 Step Decay

- 학습률 Step Decay는 사용자 함수를 만들어 LearningRateScheduler() 메써드에 담아 콜백으로 학습과정으로 넘겨주면 된다.

```python
# 학습률 Step Decay를 위한 함수
def step_decay(epoch):
    start = 0.2
    drop = 0.5
    epochs_drop = 3
    lr = start * (drop ** np.floor((epoch)/epochs_drop))
    return lr

lr_scheduler = LearningRateScheduler(step_decay, verbose=0)

model = get_model()

model_results = model.fit(x_train, y_train
                          , batch_size              = 16 # None
                          , epochs                  = 10 # 1
                          , verbose                 = 0 # 'auto'
                          , callbacks               = [lr_scheduler] # None
                          )

model_accuracy = print_accuracy(model, verbose=1)

# 그래프 그리기
fig, ax = plt.subplots( figsize=(5, 4) )
fig.suptitle('Step Decay', fontsize = 22, fontweight = 'bold', y = 1)

sns.lineplot(x=range(1,11), y=model_results.history['lr'], color='r', label='Learning Rate')

plt.xlabel('epoch', fontsize = 14)
plt.ylabel('Learning Rate', fontsize = 14)
plt.show()
```

- 학습률이 epoch가 진행됨에 따라 만든 함수에 따라서 감소하는 것을 볼 수 있다.
- Train Accuracy : 0.996, Test Accuracy : 0.986

![image]({{ site.url }}{{ site.baseurl }}/assets/images/coding/penguins/step_decay.png){: .align-center width="60%"} 

<br>

### ### 학습률 Cosine Decay





<br><br>


## 과적합 방지


<br><br>


## 하이퍼파라미터 튜닝


```python
# 데이터를 불러온다
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
print(f"{train_images.shape},{train_labels.shape},{test_images.shape},{test_labels.shape}")
```

```
(60000, 28, 28),(60000,),(10000, 28, 28),(10000,)
```

- Fashion MNIST 데이터는 MNIST와 함께 딥러닝의 'Hellow, World!'와 같은 데이터이다.
- keras.datasets.fashion_mnist.load_data() 메써드는 넘파이 튜플을 반환한다.
- 학습데이터는 60,000개, 평가데이터는 10,000개이다.
- 28픽셀의 흑/백 특성을 담고 있다.

<br>

### 데이터 정규화
- 데이터를 255로 나누어 0~1사이의 수로 정규화(Normalization)한다.
- 정규화 하는 이유?
  - 0~1 사이로 맞추어 계산 값이 너무 커지는 것을 방지
  - Local Minimun에 빠지는 것을 방지(학습 속도 향상)

- 훈련데이터 인덱스20,000번 데이터의 정규화 하기 이전 값 살펴보기

```python
train_images[attention_train][10]
```

```
array([  0,   0,   0,   0,   0,   0,   0,   0,   2,   0,   0,   0,   0,
       160, 255, 217, 255,  94,   0,   0,   0,   1,   4,   0,   0,   0,
        65,  38], dtype=uint8)
```

- 정규화 실행 $x - x_{min} \over x_{max} - x_{min}$  
  - 최소값이 0이기 때문에 최대값(255)로 나누기만 하였다

```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

```
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.00784314, 0.        ,
       0.        , 0.        , 0.        , 0.62745098, 1.        ,
       0.85098039, 1.        , 0.36862745, 0.        , 0.        ,
       0.        , 0.00392157, 0.01568627, 0.        , 0.        ,
       0.        , 0.25490196, 0.14901961])
```

<br>

### 데이터 살펴보기

- 타겟 라벨 확인하기

```python
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(pd.DataFrame({'Label':class_names}).to_markdown())
```

|    | Label       |
|---:|:------------|
|  0 | T-shirt/top |
|  1 | Trouser     |
|  2 | Pullover    |
|  3 | Dress       |
|  4 | Coat        |
|  5 | Sandal      |
|  6 | Shirt       |
|  7 | Sneaker     |
|  8 | Bag         |
|  9 | Ankle boot  |

<br>

- 훈련데이터 인덱스20,000번 데이터를 시각화 해보자

```python
plt.imshow(train_images[attention_train], cmap='gray_r')
plt.colorbar(shrink = .4)
plt.grid(False)
plt.show()
```

![image]({{ site.url }}{{ site.baseurl }}/assets/images/coding/f_mnist/f_mnist_train20000.png){: .align-center width="60%"}  

<br>

- 훈련데이터 25개 데이터를 시각화 해보자

```python
for i, val in enumerate(range(attention_train-12,attention_train+13)):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[val], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[val]])
plt.show()
```

![image]({{ site.url }}{{ site.baseurl }}/assets/images/coding/f_mnist/f_mnist_review.png){: .align-center width="80%"}  


<br><br>


## 모델링

### 모델 구성하기

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

- 28픽셀 흑백 데이터이기 때문에 입력층에는 (28, 28)로 입력한다.
- 은닉층의 노드는 128개, 활성화함수는 relu를 사용했다.
- 출력층은 10개 노드와 softmax 활성화함수를 지정하였다.

<br>

### 모델 학습하고 평가하기

```python
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
```

```
Epoch 1/10
1875/1875 [==============================] - 7s 3ms/step - loss: 0.4966 - accuracy: 0.8267
Epoch 2/10
1875/1875 [==============================] - 5s 3ms/step - loss: 0.3765 - accuracy: 0.8638
Epoch 3/10
1875/1875 [==============================] - 6s 3ms/step - loss: 0.3392 - accuracy: 0.8759
Epoch 4/10
1875/1875 [==============================] - 5s 3ms/step - loss: 0.3156 - accuracy: 0.8855
Epoch 5/10
1875/1875 [==============================] - 5s 3ms/step - loss: 0.2979 - accuracy: 0.8905
Epoch 6/10
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2817 - accuracy: 0.8958
Epoch 7/10
1875/1875 [==============================] - 5s 3ms/step - loss: 0.2686 - accuracy: 0.9003
Epoch 8/10
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2586 - accuracy: 0.9033
Epoch 9/10
1875/1875 [==============================] - 5s 3ms/step - loss: 0.2477 - accuracy: 0.9072
Epoch 10/10
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2401 - accuracy: 0.9104
313/313 - 0s - loss: 0.3328 - accuracy: 0.8863 - 488ms/epoch - 2ms/step

Test accuracy: 0.8863000273704529
```

- 순전파와 역전파를 10번 반복하여 신경망을 학습한다.
- 평가데이터에 대하여 정확도를 산출
- 평가정확도 88.6%

<br>

### 모델 확인하기

- 모델 추정  

```python
probability_model = tf.keras.Sequential([model, keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
```

#### 평가 인덱스 2,000번, 모델 추정  

- 모델은 8번(Bag)이라 예측
- 추정이 맞는지 확인해 보자

```python
i = test_1
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.xticks(range(10), class_names, rotation=90)
plt.show()
```

![image]({{ site.url }}{{ site.baseurl }}/assets/images/coding/f_mnist/f_mnist_test1.png){: .align-center width="70%"}  

- 맞았다~!😄

#### 평가 인덱스 800번, 모델 추정  

- 모델은 9번(Ankle boot)이라 예측
- 추정이 맞는지 확인해 보자

```python
i = test_2
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.xticks(range(10), class_names, rotation=90)
plt.show()
```

![image]({{ site.url }}{{ site.baseurl }}/assets/images/coding/f_mnist/f_mnist_test2.png){: .align-center width="70%"}  

- 틀렸다...😥

#### 평가데이터 15개 추정  

```python
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols

fig, ax = plt.subplots( figsize=(2*2*num_cols, 2*num_rows) )

for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
```

![image]({{ site.url }}{{ site.baseurl }}/assets/images/coding/f_mnist/f_mnist_test15s.png){: .align-center width="70%"} 

- 모델이 대부분 맞혔는데(파란색), 틀린것(빨간색)도 있다


#### 혼돈행렬(Confusion Matrix)  

![image]({{ site.url }}{{ site.baseurl }}/assets/images/coding/f_mnist/f_mnist_cm.png){: .align-center width="70%"}  




<br>
<br>
<br>
<br>


<center>
<h1>끝까지 읽어주셔서 감사합니다😉</h1>
</center>


<br>
<br>
<br>
<br>


<!-- 


{: .notice}
{: .notice--primary}
{: .notice--info}
{: .notice--warning}
{: .notice--danger}
{: .notice--success}

 -->





