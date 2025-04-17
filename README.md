# SK 네트웍스 Family AI 캠프 12기 2차 프로젝트

## 팀소개

### 팀 이름 : 블라블라

<table>
  <thead>
    <td align="center">
      <a href="https://github.com/sungho-kwon">
        <img src="./readme_img/sungho.png" height="150" width="150" alt="sungho"/><br /><hr/>
        권성호
      </a><br />
    </td>
    <td align="center">
      <a href="https://github.com/sungil-cho">
        <img src="./readme_img/sungil.png" height="150" width="150" alt="sungil"/><br /><hr/>
        조성지
      </a><br />
    </td>
    <td align="center">
      <a href="https://github.com/ikyung-kim">
        <img src="./readme_img/ikyung.png" height="150" width="150" alt="ikyung"/><br /><hr/>
        김이경
      </a><br />
    </td>
    <td align="center">
      <a href="https://github.com/seokwon-lee">
        <img src="./readme_img/seokwon.png" height="150" width="150" alt="seokwon"/><br /><hr/>
        이석원
      </a><br />
    </td>
    <td align="center">
      <a href="https://github.com/yonggyu-lee">
        <img src="./readme_img/yonggyu.png" height="150" width="150" alt="yonggyu"/><br /><hr/>
        이용규
      </a><br />
    </td>
  </thead>
</table>

## 프로젝트 소개

### 🏦 프로젝트 명: 은행 고객 이탈 예측

![이미지](./readme_img/4.png)

#### 📆 개발 기간

> 2025.04.16 ~ 2025.04.17 (총 2일)

#### 📂 데이터 셋

> https://www.kaggle.com/datasets/andieminogue/newspaper-churn

- 은행 고객 이탈

#### 💵 프로젝트 배경

#### 💴 프로젝트 필요성

## 기술스택

| 목록        | 기술                                                                                                                                                                                                                   |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 언어        | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)                                                                                                                    |
| 데이터 분석 | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white), ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)                  |
| 시각화      | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-1f77b4?style=for-the-badge&logo=seaborn&logoColor=white) |
| 협업        | ![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white) ![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)                         |

## WBS

### 📆 프로젝트 일정정

| 내용               | 기간                |
| ------------------ | ------------------- |
| 프로젝트 주제 설정 | 25.03.31 - 25.03.31 |
| 데이터 수집        | 25.03.31 - 25.03.31 |
| 데이터 EDA         | 25.03.31 - 25.03.31 |
| 데이터 전처리      | 25.03.31 - 25.03.31 |
| 모델 선정          | 25.03.31 - 25.03.31 |
| 모델 학습 및 평가  | 25.03.31 - 25.04.01 |
| README 작성        | 25.03.31 - 25.04.01 |
| 발표 준비          | 25.04.01 - 25.04.01 |

## 주요 모델

| **번호** | **모델명**                              |
| -------------- | --------------------------------------------- |
| 1              | **로지스틱 회귀** (Logistic Regression) |
| 2              | **K-최근접 이웃** (KNN)                 |
| 3              | **의사결정나무** (Decision Tree)        |
| 4              | **랜덤 포레스트** (Random Forest)       |
| 5              | **XGBoost**                             |
| 6              | **서포트 벡터 머신** (SVM)              |
| 7              | **다층 퍼셉트론** (MLPClassifier)       |
| 8              | **보팅 분류기** (Voting Classifier)     |

## EDA

### 📊 데이터 로드

그림

### 📊 데이터 드랍

그림

### 📊 데이터 결측치 확인

그림

### 📊 데이터 이상치 확인

그림

### 📊 이탈 여부 라벨 인코딩

그림

### 📊 인코딩전 히트맵 시각화

그림

### 📊 인코딩_ 라벨 인코딩, 원 핫 인코딩

그림

## 데이터 구성(임시)

| **컬럼명**                   | **의미**                                                                             |
| ---------------------------------- | ------------------------------------------------------------------------------------------ |
| **Attrition_Flag**           | 고객 이탈 여부 (**타겟 변수**)▪ Existing Customer (잔류)▪ Attrited Customer (이탈) |
| **Customer_Age**             | 고객 나이                                                                                  |
| **Gender**                   | 성별 (M/F)                                                                                 |
| **Dependent_count**          | 부양가족 수                                                                                |
| **Education_Level**          | 교육 수준 (예: Graduate, High School 등)                                                   |
| **Marital_Status**           | 혼인 상태 (Married, Single 등)                                                             |
| **Income_Category**          | 소득 구간 (예: $60K - $80K 등)                                                             |
| **Card_Category**            | 보유한 카드 종류 (Blue, Silver, Gold 등)                                                   |
| **Months_on_book**           | 해당 고객이 보유한 지 몇 개월 되었는지 (은행에 남아있는 기간)                              |
| **Months_Inactive_12_mon**   | 최근 12개월 내 비활성 월 수                                                                |
| **Avg_Utilization_Ratio**    | 평균 신용카드 사용률 (리볼빙 잔고 / 한도)                                                  |
| **Avg_Open_To_Buy**          | 평균 사용 가능한 신용금액 (Credit_Limit - Total_Revolving_Bal)                             |
| **Contacts_Count_12_mon**    | 최근 12개월 내 고객센터에 연락한 횟수                                                      |
| **Credit_Limit**             | 고객의 신용한도                                                                            |
| **Total_Revolving_Bal**      | 리볼빙 잔고 (돌려막기 용도의 카드 미지급액 등)                                             |
| **Total_Amt_Chng_Q4_Q1**     | 최근 1분기 대비 4분기 거래금액 변화율                                                      |
| **Total_Trans_Amt**          | 총 거래 금액 (최근 기간)                                                                   |
| **Total_Trans_Ct**           | 총 거래 횟수 (최근 기간)                                                                   |
| **Total_Ct_Chng_Q4_Q1**      | 거래 횟수의 분기 변화율                                                                    |
| **Total_Relationship_Count** | 총 금융상품 보유 개수 (예: 신용카드, 예금, 대출 등 포함 추정)                              |

## 데이터 시각화

그림, 그림

## 모델별 분석

### 🤖 모델별 분석

| 모델                           | Precision (Class `0`) | Recall (Class `0`) | F1 Score (Class `0`) | Precision (Class `1`) | Recall (Class `1`) | F1 Score (Class `1`) |
| ------------------------------ | ----------------------- | -------------------- | ---------------------- | ----------------------- | -------------------- | ---------------------- |
| 로지스틱 회귀 (Cost-Sensitive) | 0.30                    | 0.56                 | 0.39                   | 0.88                    | 0.71                 | 0.78                   |
| KNN                            | 0.31                    | 0.63                 | 0.42                   | 0.89                    | 0.69                 | 0.78                   |
| 랜덤 포레스트 (Cost-Sensitive) | 0.49                    | 0.45                 | 0.47                   | 0.88                    | 0.90                 | 0.89                   |
| XGBoost (Cost-Sensitive)       | 0.79                    | 0.20                 | 0.32                   | 0.85                    | 0.99                 | 0.91                   |
| MLP (Cost-Sensitive)           | 0.35                    | 0.55                 | 0.42                   | 0.88                    | 0.77                 | 0.82                   |
| SVM (Cost-Sensitive)           | 0.32                    | 0.60                 | 0.42                   | 0.89                    | 0.72                 | 0.79                   |

---

### 🤖 적용후(먼가 임시시)

| Model                        | Accuracy (Before → After) | Precision (Before → After) | Recall (Before → After) | F1 Score (Before → After) | ROC AUC (Before → After) |
| ---------------------------- | -------------------------- | --------------------------- | ------------------------ | -------------------------- | ------------------------- |
| **LogisticRegression** | 0.8319 → 0.8320           | 0.6814 → 0.6815            | 0.6140 → 0.6142         | 0.6459 → 0.6461           | 0.8739 → 0.8743          |
| **RandomForest**       | 0.8601 → 0.9618           | 0.7606 → 0.9811            | 0.6416 → 0.8638         | 0.6960 → 0.9187           | 0.9190 → 0.9896          |
| **XGBoost**            | 0.8623 → 0.8907           | 0.7565 → 0.8243            | 0.6613 → 0.7145         | 0.7057 → 0.7655           | 0.9225 → 0.9460          |
| **LightGBM**           | 0.8636 → 0.8808           | 0.7794 → 0.8180            | 0.6329 → 0.6721         | 0.6985 → 0.7379           | 0.9237 → 0.9388          |

### 모델별 성능비교 그래프

그림 그림

## 최종학습 모델 선정

```
이러이러한 상황으로 해당 모델을 선정했다.
```

## 은행행 이탈 예측 대시보드

그림, 그림

## 인사이트 및 결론

```
블라브라
```

## 회고

| 팀원     | 한 줄 회고 |
| -------- | ---------- |
| 권성호   |            |
| 조성지   |            |
| 김이경   |            |
| 이석원   |            |
| 이용규   |            |
