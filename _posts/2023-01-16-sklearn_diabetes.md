---
title: "Toydata - Scikit-Learn - Diabetes"
excerpt: "토이데이터 살펴보기 Scikit-Learn : Diabetes"
categories:
  - Coding
  - Toydata
  - Linear_Regression
tags:
  - 개발일지
  - 코딩
  - Toydata
  - EDA
  - Linear_Regression

use_math: true

header:
  teaser: https://lh3.googleusercontent.com/fife/AAbDypBUQ29Ak5npZ7hFSqRztU1Rj8nd6ceL3K26T26uqAFYdHc8N4G1hd2fsTLe7Ls0eIjTl2vTusrWvf8GOkmEx9c5rG8mLfoVApYdOBSqHtNdbrIgQYK-DUldZGKkoXArWKbPlxGSNNGmliYm6iLiHSEeCEHQkNW9iMgQlPb9ZR8c4MAXLrFzcKIfUs-et9c46lgSWAZpN7xxqjImQv1-J2MoRWlmi0ToWsSF5fWThCoKP0a9ine4xju7r_ytD0vFLgssGUMkN1W-3HX_nx3wKHIg1Nte3lctYONEKQn8c9Xsny0k4r6_RivG1-gCXBGj0vCmYHVLmv87ympXkQYKdKybgBqyiHV1RDpBNHJKrYInp7v8oAqNlfz4QnqoW6szwFu9ujwEhROqvb9JQzOHZEsKhp4WXZDHyqdnHzGa-vWr7VBLAcpx5Rj6BHMAPLcJ-A_CTIxh7PHKHIq60JMSghn0UKi1_kAq7rIohu2ukQxonPxJ-W6GRY-PzWQeIVnWgDgQ3kZj4sHaoqUc4QsCMFis7MYp4iNPKFWMTThsliKSSycosoj40Hh23wqrnBPhYewfnoUp3Jj02YwJKnzNuovgAopq294iqZjtOOptTaVqWcHhp-08Q6DeVcI2EWJf7zBtT27YRi7NSe5Khs0XwfzG6_wUUi3_77RETXUfpnmME5N2_JKIF_hu5SjS279-nlPSGp3YkD2lgsKzPAGKqgPHsBQ6bZlmx7UoXkYUiQuWvVremyymljd_bahiju_wPYLD1xOwoHwOeB0vqn8sLX_iC5ruz48yQc0TP2Am10K9N32MhjHDw9nt_qVI7Fm4MVGX7U3a_6calivWtFDjQ4aQehecq1_9zO6v6tJrzyRw19va4RQ8rcM6q_BOR41DKAZotTyBM5H0F0NtC7oHySMp7in_76_4JZ_KX288nCP8lgqROvFA0TR62d8wzdrm4wwg0Yz8QrkVAav2sdKDqo8eh_0hc0cDoZ4hdyAFLLKZSmLOrw8YmLRWh09gg21pklTOYg19BuRbYNwJZ87tyr_gyHbxgoVSQY8ZfPZP5fuTFCFUSUq5NJqgGmtmaVjbFVrNBwwQaS25MZbHHPbYf3t9cbZQa1Y9VmkCYakWz2XkcD0968f67Jb9j4K9umJwnODdAljztCl51HAdLnFfJImJv0mATeZT5sfmg9hE66MEAgPJuIwXWX5fjm_8WamlJsxPX9hxAXMGrf-Q97jko01TuB41cT9PuLkLmJCX-fqi0k3oLRSMP7F4jKD1I4r3j094860mzAF-WRGSSK07bG4MtWBNMvIGJ7wdV7PbyCGs7FLfU2ovSBTH8WxQ-fPLvjtK0vjogMW25Kx86XUXuQ51Xj9GUFzTQjxZy6aSj4HlEIZmUYbQwSwYelb1NjAQZCTBhgDHBUOb7BwC-v2st3aMey5ub_eRv4aAso_jqaMWmzTpJ98-yQKAN75_adAo6vIXtry1GCga6ts_ZuATBAA62zVBC0-thq5-M8nI4KzjG5FrJNm1FmkvILt9=w1200-h780

last_modified_at: 2023-01-16
---


<div style="text-align : center;">
  <img alt="image" width="80%"
  src="https://lh3.googleusercontent.com/fife/AAbDypCXUiJMZ6fUWSd_BhOMgic50T5Ilhb0D0_kqkkd49Y5f1_RFYbt5Qh8QcX0XaZxFwsQ1BSYcEFVGGggHeu2yNGz4H5TJv9Z7UwY6O1fDriMmpdYRLSR_66-l9izq0bHBvtykdqxV8v1-D7NBCgsFrCERnIOwU8MLSUSQkFBza0BymUM86CZvEzr_7kcsFyyBF3-eOVGe1Xy7xNZF51g7q8I9rDZI9lyQGCWB5wPkd-d_du-_rQ87WFZkHAzPJrYHA6jlJTIfGNGypisAtmxfoeX3v1LIrbls-kJ6ggt-NafMkKpaHaRijCoy-wpSLilhoFxgSsdoJkLx8tbWejy-Yll-25WwR6hAbT5bSm3GF-y1NwbQ1Aj94LMkCxLyHaXSPmMQTrho0rfjYgOp1Mn3qPMERnatb5nAu7qInYV1YtAZEhvJ3Ng2brLHdvfgLaRiMuumtqIU3LF_TE6VrIIHml2cIkUcSirAih-NqwAeNvP2AL7ivSfnyD5JDVVHhLDFCEL5-qIZZBS1onfqXEWODdFH5EV3lEwoiN5wdyIMKwzuxyi0CACJeQSSnOzVUxIV6EOIJhryrBlsa7O4xFf0afg2tV5gNL1_pYm3zMyJMfwW6ptMk8_cfkhc_tZTyyl0qQOdpBQmcSe_4pUyYzLNd0vLe6_7WCQBXA4T-afTLJb3ucw9FZ29bhWohm_VWCYVbrofUk5sIukxbbKOG9i6uwIn4qjOvCuTHqudnckqXL4HeRaLNbXNaFfTXN5ioH41xIP4ZUQvev-duezO4OZLI8eO-H-64doamXb9bm31fyq_ymwzJGo331IPbR9mTxh8hNKPmPCjmce4ewaCuI_zEjrc9NQA2iCLqhTm8Rp26wA_lJszK64GRZkj_DlinBh83cr1Ft4QILF7_23YyZfQ5YpE3Ts952OHZ58fCPwreJifPIIS0Rcppt8tod1yHtYMnxdv-XAEY5QCCpzU_cLSiTcxxUzwfLrWs0tyuIL8gvaVMcn8IgAoLxGRqThbLMQJBWNy8pmaED3BlYBZlTTlgefhM4r15GMK3tKc_tkP6UCrhB5huSib_wXZ99-Ng32ru6bVLZdSc1-0mqPy2M39X4pYhnkRJQHWK43G5XWukqIxPJo65ftC6mBEG4IqFHU9u6l-6TQEgxb3_7Jr0rEdoPDnNgvXrC4xTWoqlq7H99VgjPJkn4GE4SG3Dt2wjGsIJ34g6bldgstiKMNeq8gNeHe8uca_Ej0F-d2XJud61l5BF2HSeiXOtmOWneZhWwysU_D_ZVJHkOOhBbFY9VA1yaovj91jDcA-ofW5cUVVR3QVWOn9V__u2thbSkPfBpT6Y8nLf3AUqXRnYxHZHklptIhwatGkmdm_YlX3MPCjJOegW3ZiVfbw2fFiWtWMhzij3-4_avpAwO_crjlKLDDptV-HFflZ_IuCy4PWXn-qRaoXbyHPlRdbrsQrxsrQOV8agIojta_PjI62PCyQOvKLGAKxh-NQf0wVkkO958V09aqk7rM2k2B_AGYog5v=w1920-h780">  
</div>  
  


<br><br><hr><br><br>
<div style="text-align : center; font-size : 2rem">

토이데이터 살펴보기
Scikit-Learn : Diabetes

</div>  
<br><br><hr><br><br>

# Scikit-Learn 의 Diabetes 데이터셋

- 사이킷런에서 제공하는 여러 토이데이터셋 가운데 diabetes (당뇨병) 살펴보기


>- schikit learn 의 diabetes 데이터
>- [사이킷런 제공 연습데이터셋 리스트](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets)  
>- [diabetes 공식문서](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes)

<br><br>

## 데이터 확인
### 불러오기
- 기본적인 방법으로 데이터 불러오기   

```python
# 라이브러리 임포트
from sklearn import datasets

# 당뇨병데이터 불러오기
diabetes = datasets.load_diabetes()
```  

<br><br>

### 데이터 확인
- sklearn 에서 제공하는 토이데이터는 딕셔너리형태로 제공
  - 딕셔너리 형태는 {key : value}
  - 예  
  {'이름' : '홍길동',   
  '사는곳' : '아산시 탕정면',  
  'age' : 38}  
- 데이터 딕셔너리가 가지고 있는 key 가 무엇이 있는지 확인
  ```python
  diabetes.keys()
  ```
  >dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
  - key를 하나하나 확인하여 key에 있는 value를 확인해보자  

- data  

  ```python
  diabetes.data
  ```  

  >array([[ 0.03807591,  0.05068012,  0.06169621, ..., -0.00259226,
         0.01990842, -0.01764613],
       [-0.00188202, -0.04464164, -0.05147406, ..., -0.03949338,
        -0.06832974, -0.09220405],
       [ 0.08529891,  0.05068012,  0.04445121, ..., -0.00259226,
         0.00286377, -0.02593034],
       ...,
       [ 0.04170844,  0.05068012, -0.01590626, ..., -0.01107952,
        -0.04687948,  0.01549073],
       [-0.04547248, -0.04464164,  0.03906215, ...,  0.02655962,
         0.04452837, -0.02593034],
       [-0.04547248, -0.04464164, -0.0730303 , ..., -0.03949338,
        -0.00421986,  0.00306441]])
  - 흔히 독립변수, 피쳐라고 부르는 데이터들이 나온다
  - 행과 열이 있는 2차원 배열이다

- target  

  ```python
  diabetes.target
  ```  

  >array([151.,  75., 141., 206., 135.,  97., 138.,  63., 110., 310., 101.,
        69., 179., 185., 118., 171., 166., 144.,  97., 168.,  68.,  49.,
        68., 245., 184., 202., 137.,  85., 131., 283., 129.,  59., 341.,
        87.,  65., 102., 265., 276., 252.,  90., 100.,  55.,  61.,  92.,
       259.,  53., 190., 142.,  75., 142., 155., 225.,  59., 104., 182.,
       128.,  52.,  37., 170., 170.,  61., 144.,  52., 128.,  71., 163.,
       150.,  97., 160., 178.,  48., 270., 202., 111.,  85.,  42., 170.,
       200., 252., 113., 143.,  51.,  52., 210.,  65., 141.,  55., 134.,
        42., 111.,  98., 164.,  48.,  96.,  90., 162., 150., 279.,  92.,
        83., 128., 102., 302., 198.,  95.,  53., 134., 144., 232.,  81., ...
        ...
  - 종속변수, 타겟이라고 부르는 데이터
  - 1차원 배열  

- DESCR  
  ```python
  print(diabetes.DESCR)
  ```  

  ```
  Diabetes dataset
  ----------------
  Ten baseline variables, age, sex, body mass index, average blood
  pressure, and six blood serum measurements were obtained for each of n =
  442 diabetes patients, as well as the response of interest, a
  quantitative measure of disease progression one year after baseline.

  **Data Set Characteristics:**

    :Number of Instances: 442

    :Number of Attributes: First 10 columns are numeric predictive values

    :Target: Column 11 is a quantitative measure of disease progression one year after baseline

    :Attribute Information:
        - age     age in years
        - sex
        - bmi     body mass index
        - bp      average blood pressure
        - s1      tc, total serum cholesterol
        - s2      ldl, low-density lipoproteins
        - s3      hdl, high-density lipoproteins
        - s4      tch, total cholesterol / HDL
        - s5      ltg, possibly log of serum triglycerides level
        - s6      glu, blood sugar level

  Note: Each of these 10 feature variables have been mean centered and scaled by the standard deviation times `n_samples` (i.e. the sum of squares of each column totals 1).

  Source URL:
  https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html

  For more information see:
  Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004) "Least Angle Regression," Annals of Statistics (with discussion), 407-499.
  (https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf)
  ```  

  - 데이터에 대한 소개를 볼 수 있는 key
  - 혈액정보가 포함되어있는 10개의 당뇨병과 관련된 변수
  - 각 변수는 442개 측정값
  - 종속변수는 1년 후의 병의 경과에 대한 양적인 측정값
  - 각 변수는 평균중심화하고, 표준편차와 샘플 수의 곱으로 나누었음  
    - 평균은 0, 합은 1로 변형

- feature_names  

  ```python
  diabetes.feature_names
  ```  

  - 각 독립변수(=피쳐)의 이름  

- data_filename, target_filename  

  ```python
  print(diabetes.data_filename)
  print(diabetes.target_filename)
  ```  

  - 데이터파일 이름과, 타겟파일 이름

<br><br>

### 데이터 준비

- pandas 데이터프레임
  ```python
  import pandas as pd

  df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
  df['target'] = diabetes.target
  df
  ```
  - diabetes.data 을 데이터프레임 df로 지정
  - df에 target 추가

- df 정보
  ```python
  df.info()
  ```  

  ```
  <class 'pandas.core.frame.DataFrame'>
  RangeIndex: 442 entries, 0 to 441
  Data columns (total 11 columns):
  #   Column  Non-Null Count  Dtype  
  ---  ------  --------------  -----  
  0   age     442 non-null    float64
  1   sex     442 non-null    float64
  2   bmi     442 non-null    float64
  3   bp      442 non-null    float64
  4   s1      442 non-null    float64
  5   s2      442 non-null    float64
  6   s3      442 non-null    float64
  7   s4      442 non-null    float64
  8   s5      442 non-null    float64
  9   s6      442 non-null    float64
  10  target  442 non-null    float64
  dtypes: float64(11)
  memory usage: 38.1 KB
  ```  

  - 테이터타입은 판다스 데이터프레임
  - 442개 인덱스
  - 11개 컬럼
  - 11개 컬럼 모두 float(소수형) 타입

- 기술통계
  ```python
  df.describe()
  ```

  ||age |sex |bmi |bp |s1 |s2 |s3 |s4 |s5 |s6 |target |
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  | count |442.000 |442.000 |442.000 |442.000 |442.000 |442.000 |442.000 |442.000 |442.000 |442.000 |442.000 |
  | mean |-0.000 |0.000 |-0.000 |0.000 |-0.000 |0.000 |-0.000 |0.000 |-0.000 |-0.000 |152.133 |
  | std |0.048 |0.048 |0.048 |0.048 |0.048 |0.048 |0.048 |0.048 |0.048 |0.048 |77.093 |
  | min |-0.107 |-0.045 |-0.090 |-0.112 |-0.127 |-0.116 |-0.102 |-0.076 |-0.126 |-0.138 |25.000 |
  | 25% |-0.037 |-0.045 |-0.034 |-0.037 |-0.034 |-0.030 |-0.035 |-0.039 |-0.033 |-0.033 |87.000 |
  | 50% |0.005 |-0.045 |-0.007 |-0.006 |-0.004 |-0.004 |-0.007 |-0.003 |-0.002 |-0.001 |140.500 |
  | 75% |0.038 |0.051 |0.031 |0.036 |0.028 |0.030 |0.029 |0.034 |0.032 |0.028 |211.500 |
  | max |0.111 |0.051 |0.171 |0.132 |0.154 |0.199 |0.181 |0.185 |0.134 |0.136 |346.000 |
  
  - 10개 피쳐들의 평균이 0이다.


  ```python
  df.describe()
  ```  

  ||sum |
  |:---:|:---:|
  | age |-0.000 |
  | sex |0.000 |
  | bmi |-0.000 |
  | bp |0.000 |
  | s1 |-0.000 |
  | s2 |0.000 |
  | s3 |-0.000 |
  | s4 |0.000 |
  | s5 |-0.000 |
  | s6 |-0.000 |
  | target |67243.000 |
  - 10개 변수들의 합계는 0이다.


  ```python
  df.isna().sum()
  ```  

  ||0 |
  |:---:|:---:|
  | age |0.000 |
  | sex |0.000 |
  | bmi |0.000 |
  | bp |0.000 |
  | s1 |0.000 |
  | s2 |0.000 |
  | s3 |0.000 |
  | s4 |0.000 |
  | s5 |0.000 |
  | s6 |0.000 |
  | target |0.000 |
  - 혹시나 해서 결측값을 찾아보았지만, 결측값은 없다.



  ```python
  df.duplicated().sum()
  ```  

  - 중복값도 없다.

- 상관관계

  ```python
  df.corr()
  ```

  ||age |sex |bmi |bp |s1 |s2 |s3 |s4 |s5 |s6 |target |
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  | age |1.000 |0.174 |0.185 |0.335 |0.260 |0.219 |-0.075 |0.204 |0.271 |0.302 |0.188 |
  | sex |0.174 |1.000 |0.088 |0.241 |0.035 |0.143 |-0.379 |0.332 |0.150 |0.208 |0.043 |
  | bmi |0.185 |0.088 |1.000 |0.395 |0.250 |0.261 |-0.367 |0.414 |0.446 |0.389 |0.586 |
  | bp |0.335 |0.241 |0.395 |1.000 |0.242 |0.186 |-0.179 |0.258 |0.393 |0.390 |0.441 |
  | s1 |0.260 |0.035 |0.250 |0.242 |1.000 |0.897 |0.052 |0.542 |0.516 |0.326 |0.212 |
  | s2 |0.219 |0.143 |0.261 |0.186 |0.897 |1.000 |-0.196 |0.660 |0.318 |0.291 |0.174 |
  | s3 |-0.075 |-0.379 |-0.367 |-0.179 |0.052 |-0.196 |1.000 |-0.738 |-0.399 |-0.274 |-0.395 |
  | s4 |0.204 |0.332 |0.414 |0.258 |0.542 |0.660 |-0.738 |1.000 |0.618 |0.417 |0.430 |
  | s5 |0.271 |0.150 |0.446 |0.393 |0.516 |0.318 |-0.399 |0.618 |1.000 |0.465 |0.566 |
  | s6 |0.302 |0.208 |0.389 |0.390 |0.326 |0.291 |-0.274 |0.417 |0.465 |1.000 |0.382 |
  | target |0.188 |0.043 |0.586 |0.441 |0.212 |0.174 |-0.395 |0.430 |0.566 |0.382 |1.000 |

- 가장 마지막 행이 타겟이기 때문에, 타겟행만 보면...
  - 체질량지수(bmi)와 혈청 트리글리세리드 수치(s5)가 상관관계가 높다.


- 상관관계 표 보기좋게 그리기
  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns

  # 상관관계 히트맵 그리기
  coff_df = df.corr()

  # 그림 사이즈 지정
  fig, ax = plt.subplots( figsize=(12,12) )
  fig.suptitle('Correlation Heat Map', fontsize = 24, fontweight = 'bold', y = 0.95)

  # 삼각형 마스크를 만든다(위 쪽 삼각형에 True, 아래 삼각형에 False)
  mask = np.zeros_like(coff_df, dtype=np.bool)
  mask[np.triu_indices_from(mask)] = True

  # 히트맵을 그린다
  sns.heatmap(coff_df, 
              cmap = 'RdYlBu_r', 
              annot = True,             # 실제 값을 표시한다
              mask=mask,                # 표시하지 않을 마스크 부분을 지정한다
              linewidths=.5,            # 경계면 실선으로 구분하기
              cbar_kws={"shrink": .5},  # 컬러바 크기 절반으로 줄이기
              vmin = -1,vmax = 1        # 컬러바 범위 -1 ~ 1
            )  
  plt.show()
  ```

  <div style="text-align : center;">
    <img alt="image" width="80%"
    src="https://lh3.googleusercontent.com/fife/AAbDypCUozLz2f-qxKzSwFpWAQahGSKTM0SWOK4H8Th7HDlaCQZK1-6aa5LEDHV6gE6mqAAPTSkO6nGUGMvUPmx5w7DtuQEgxTeW-V1YbL-ziLY5cCidjCdb5aCUg95U4sCfT2ZPL9mCtpXjT3_APUA6vOZSzqY4OaxVYa-8CmbfQXq06pFdvY0jfE12bVXigqLx3-8WP8t6v1SxS0vCLcIgprnfwyrU1UbpTT46_jSakjA78KINyQx5xBBJl6Z_QXILxARZgkdM-pQhWTPm9yjU09BuGt7n7pJn0ZMo9CMDjDF1DyJp_Cymt0sVV0j9PsiIVQQS6m6rI95bTaq2NjjTzqAcOLl8hbc4-uT4N-HGkgpdRJpjsBCAyheJl1CmTz8fX6emc6V42pxf9LAETDthV-lpery-RjA_R22uhKxRM1xmqe5C4eUWEUYpmvHgXoydwaVmMA5JNzbYwfepE1vIeeJy6yFqALGm7Ou1u-4o838WTi2Kq6VW653z8yzVmsxdI-WnWuBy8g2VZRNRVex97jlw5a4lOmo69o_xQ3A86pstV6OxkCMpv-OU9DjUR1SnBXQC2qek7dinFsujqsD4SBKI3hCoIq_prWoQie6enWXacGPIRr9H-f92KwFz9TzIW04sIHaJptDtBKKbDKBxwaxCCTlkZ-lRyYUiYCzHcWjAfD7OoqrkpIQ5sybTyfvycgl87UuzktKYHaDh5L9Dc3FHBeuZIvDdv7LlPqzDMfIOPv-Pm0gKgKRJn7W7Nhg6gCNq4LVaxWS7j5_VyC0wmU4EE2wjNNzHhF6-jMYSz1fWC5j4MhDThu0I7bUBoUJZ7gliDXtaQG83f5wL5p7Iyfe1bRazrOuzVHWOoB83grRfFqmNaWPS5zwzw8mn736F80WBpSHfik_gP-2wvaL4Tz6aDg0EP676HmGlkE9jdxqSL_tsQuefqhl2ou0HEG2-8pqnoFZRBG1dekaV8ONy_XkElwtl9qteNTL3bIgRWB9b5rLI0faUO6N7vxKqKK7a0ko8zxUXCy62IAxo_-ZDsYfSLvf1DZsBdL7xltDs8ghyV3dgJ23pWMDG3FqbO6HVDuZdw9ZblLUBBHlgUSsFJwM09pLdfoAkCd5Or_eRyNNNxiPiAP3kUpgXL1mhAqvcWHsRXPcfYBUKEbRgB6y4YmeWYInCukqiq71-hXyqvlGJtzog5iZOTaoiswua7PgPyJeYwWWaE2rtMUAMF13KC7ik-XnZ6je8ooWFO0uyiRy6WcvjBbR-l5R9bA2JWV3b4t-j83ONAmrT3xNXP-5wDDxziWnGda21CCqlCwWzJFytuuTDmruylm2h0FzrVzdh_YI3J5JebDZdnfdVruC5AGMPYEabAg-aouCs4tIfYXI87zU2wn0KgDbTmZy6mN1OHP-asxI88Kut52dr3Vp2u2XkSDXBIQZ9czfjAXZAoUZT5ZvSW6yREEKvmsLqj-ho4ha0AG4u8UiRPSXeImwQYzsGNp3N0ZmyGzaQfS0L76mag8UFmnBUoiyBe-Up=w1920-h865">  
  </div>  
  

- 데이터 분포 시각화
  ```python
  fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(22,8))
  fig.suptitle('Scatter Plots', fontsize = 32, fontweight = 'bold', y = 0.95)

  cols = diabetes.feature_names

  for i in range(0, 2):
    for j in range(0, 5):
      sns.scatterplot(x = df[cols[i*5 + j]], y = df['target'], ax = ax[i][j])
      ax[i][j].axhline(df['target'].mean(), c='r', label = 'Base of Mean')

  plt.show()

  # 분석결과 저장
  file_name = 'sklearn_diabetes_scatterPlots'
  fig.savefig(path_results + file_name, dpi=150, facecolor='#eeeeee')
  ```  

  <div style="text-align : center;">
    <img alt="image" width="95%"
    src="https://lh3.googleusercontent.com/fife/AAbDypAssKrnCnSbF4uSrisVveDsEJ4RH4iG_20tHnJWzTyLOKTHqfI0xFQzc-omFyzhi2_RAnxvs5ur6WsEoO7mzFVI9MH_3XMb5VYzbadYGZ4aQB1_lNFP2x0ssnN9FK5Fa3RCxZn9SSaTISUL1XnDXmk9BXJPAjij-h3L9Q3mWDbl0k-B6rzggxV_l47NIoREppHCEJMNzMuGVxpFgXHnH8PaZFZS5HnfeEOGvGj12C8OrW8o2kRiSsrXbLvVeLgkXb8zDwn5rqmyt6s8XWZ2ZU1TW4Wj8FvFpWr7iISi3TKhKSPNfwdise1Bun18JIIoxgC1goCsj34wydqjofsuWW2NDL4M35mYt8MBgntgzXWwvY2_yX9N2EvY9nQkhj0pzV2D2uiM_-BpmJHWq3pkFIXIMqFlPZdjmsuGm8o7OGw8fCYD7kKYe6Sbe7O1v9TT_Qw3KrmaZATdLoMYRO9wKNPvo36ZL5_r2OPM1gvKCpcPaAMbooYKGtgItIgIq45ywaEpCesExXpoiGuqavZZIucUwa-gQndAcUgB3HOmS8nftSqMXJCUJVY7JeOObNlOJpv8e7YZwp_CoDEFIg8yahSW7F3zrSIzJ6IRlDo51WlizmKav-fcRIfkei37MdSS_2yUveuZ7sV469aqFiW7NhZVrLPsNP5N1SxsG8tynnGe_1y02zDGdsiPufWQDbGleTMtMbzS9w4z_qmUtRLvAPJ1eD0ehEOa_EEyB57RDDhA5snd85VQpfUXlGI_BKPg9B3-TiKhTUa7Bweoh322eAINjvqPSd0A-Bx6XZOY6Gf86esOAyic5tlc63cUT_LfKOd8E9EBU4tEEn9IU1iN_IPhKxBLYxiE_wdRPjCP_4HiOAXQn3iRVlFforfB7eddMMR_61NFWgmGTNT89II_B932CxI-suDvEc1Lm4WybuKiKGiwBQqIJvX_V3FvahUW9yvDHE9XqnAhS-lr4wcOrlA05jJFS4GjHa63TODBlNDW8fBr376p2Cb46gZreWswjvRja_j7QBRof1quo0s_VnYv-KT6c8DYSQ4_KCAs8IcGbBFNTizm3RpYqwZqJ3TW7EdZb4R9OqYdosv4ZHcrL7kjC26UX3IuYghahSfnQKcrGwkLqDNy_AOKq9MImCfANMIPoTW20Gr3iGNtzEnu5gw_mNmqVvKOGrbEb_2nVAX93k0ghqEPXdnUqRBVv_iEqbeQAbtwwiyUAjw9Kj3yrPvvziMCHVZIGgXbFKzFIuKH9YTlt4g3oboUEBUJ5qgtI0hAgg-UbMxSD9sH6RTH67Sh8UvEFNP3-ll8DK1dqaCJBkZoVo8xHt6RGoTiBip0svwycZ1ddz9ObkCKDVorXNTAkKnhBTLcfAQ35q11qMMFfk7II4CBPEB62_11CNL-ZYhz67ok1_n1YQH_OHKSV1lUuWqTxZbMMJ5tMtB-Tb2dTLNKT7j-z7OKajlGp9YRgXpj0QYow-QZsvRCduPRqRz4ToJdDFB5yWG1r6OgOZfqnNirPd8wLy0iG4hB=w1200-h865">  
  </div>  







