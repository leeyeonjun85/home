---
title: "파이썬 데이터분석"
excerpt: "파이썬으로 데이터분석 연습하기"

categories:
  - Dev Log

tags:
  - 개발일지
  - 코딩
  - 파이썬
  - 데이터분석

header:
  teaser: https://upload.wikimedia.org/wikipedia/commons/f/f8/Python_logo_and_wordmark.svg

last_modified_at: 2022-12-09
---


![python](https://upload.wikimedia.org/wikipedia/commons/f/f8/Python_logo_and_wordmark.svg)


# 마이크로데이터를 활용한 데이터분석 실습

## 분석환경
- [구글코렙(colaboratory)](https://colab.research.google.com/)
- 무료로 인터넷 브라우저 창에서 파이썬을 실습할 수 있는 장점이 있음
- 구글드라이브와 연동 쉬움, 깃허브 연동 가능

## pandas 라이브러리 불러오기
```python
import pandas as pd
```

## 데이터 기초사용

- .read_csv()  

CSV는 .read_csv()메서드를 활용하여 쉽게 파일을 읽어올 수 있다. colab에 csv를 불러오는 매서드를 사용하니 다음과 같은 에러를 만났다.  
![error1](../assets/images/post/python/20221210_000554.png)  
"UnicodeDecodeError" 구글링으로 찾아보니 "encoding="cp949" 옵션을 추가하여 쉽게 해결 하능하다.
```python
df = pd.read_csv(path, encoding="cp949")
```
![image1](../assets/images/post/python/20221210_002237.png)  

rows가 19958, columns가 272나 되는 큰 데이터이다.

- 특정 열 조회  
  - 열을 하나만 조회할 때는 df['소득']으로 가능하지만, 2개이상의 열을 조회 할 때는 `[]`로 한번 더 감싸줘야 한다.

![image2](../assets/images/post/python/20221210_002524.png)  

- 기술통계 보기
```python
df[['소득','가계지출금액']].describe()
```

- 결측치 보기
df


