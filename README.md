# SK 네트웍스 Family AI 캠프 12기 2차 프로젝트

## 팀소개

### 팀 이름 : 마리오없는 마리오팀

<table>
  <thead>
    <td align="center">
      <a href="https://github.com/seokwon-lee">
        <img src="./readme_img/seokwon.png" height="150" width="150" alt="seokwon"/><br /><hr/>
        이석원
      </a><br />
    </td>
    <td align="center">
      <a href="https://github.com/sungho-kwon">
        <img src="./readme_img/sungho.png" height="150" width="150" alt="sungho"/><br /><hr/>
        권성호
      </a><br />
    </td>
    <td align="center">
      <a href="https://github.com/yonggyu-lee">
        <img src="./readme_img/yonggyu.jpg" height="150" width="150" alt="yonggyu"/><br /><hr/>
        이용규
      </a><br />
    </td>
    <td align="center">
      <a href="https://github.com/sungil-cho">
        <img src="./readme_img/sungil.jpg" height="150" width="150" alt="sungil"/><br /><hr/>
        조성지
      </a><br />
    </td>
    <td align="center">
      <a href="https://github.com/ikyung-kim">
        <img src="./readme_img/ikyung.png" height="150" width="150" alt="ikyung"/><br /><hr/>
        김이경
      </a><br />
    </td>
  </thead>
</table>

## 프로젝트 소개

### 🏦 프로젝트 명: 은행 고객 이탈 예측

#### 📆 개발 기간

> 2025.04.16 ~ 2025.04.17 (총 2일)

#### 📂 데이터 셋

> https://github.com/adin786/bank_churn/tree/main

- 은행 고객 이탈

#### 💵 프로젝트 배경
![이탈 비율](https://i.ibb.co/HTJ6ZqvT/2025-04-17-141405.png)
![이탈 비율](https://i.ibb.co/Ps3rDN84/2025-04-17-141434.png)
- 주거래은행 이탈 비율이 높아지고 있음
    - 시중은행 이탈률: **10.1%**
    - 인터넷 전문은행 이탈률: **30.7%**
    - 연령이 낮을수록 이탈 비율이 높고 장기적으로 이탈 비율이 증가할 것으로 예상됨

#### 💴 프로젝트 필요성
![이탈 의향](https://i.ibb.co/YBf4NFp5/2025-04-17-141456.png)
- 과반이 넘는 소비자가 **거래 이탈 의향**을 가지고 있음
- 따라서 고객 이탙을 막기 위한 **이탈 예측 모델**이 필요함

![비대면 대출 갈아타기 서비스](https://img.asiatoday.co.kr/file/2024y/02m/02d/2024020201000169500007791.jpg)
- 특히 최근 낮은 금리를 지원하는 **비대면 대출 갈아타기 서비스**로 인해, 비대면 플랫폼으로의 이탈 비율이 높아지고 있음
- 때문에 은행 데이터를 기반으로 고객의 이탈을 예측하는 모델은 시의성 측면에서도 적합한 주제임

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
| 프로젝트 주제 설정 | 25.04.17 - 25.04.18 |
| 데이터 수집        | 25.04.17 - 25.04.18 |
| 데이터 EDA         | 25.04.17 - 25.04.18 |
| 데이터 전처리      | 25.04.17 - 25.04.18 |
| 모델 선정          | 25.04.17 - 25.04.18 |
| 모델 학습 및 평가  | 25.04.17 - 25.04.18 |
| README 작성        | 25.04.17 - 25.04.18 |
| 발표 준비          | 25.04.17 - 25.04.18 |

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

## 데이터 구성

| **컬럼명**                | **의미**                                |
| ------------------------- | --------------------------------------- |
| **이탈여부**              | 고객이 이탈했는지 여부                  |
| **부양가족수**            | 고객이 부양하는 가족 수                 |
| **교육수준**              | 고객의 교육 수준                        |
| **총거래관계수**          | 고객과 은행 간 전체 거래 항목 수        |
| **12개월비활성개월수**    | 최근 12개월 동안 비활성 기간(월) 수     |
| **12개월고객접촉횟수**    | 최근 12개월 동안 고객 접촉 횟수         |
| **신용한도**              | 고객의 신용 한도                        |
| **회전잔액**              | 회전(남은) 잔액                         |
| **1~4분기총이용금액변화** | 1~4분기 동안의 총 이용금액 변화         |
| **총거래금액**            | 총 거래 금액                            |
| **1~4분기거래횟수변화**   | 분기별 거래 횟수의 변화                 |
| **나이그룹**              | 나이 범주형 그룹                        |
| **수입**                  | 고객의 수입 수준                        |
| **결혼여부**              | 고객의 결혼 여부                        |
| **카드등급**              | 고객의 카드 등급                        |
| **성별**                  | 고객의 성별                             |


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

### 🤖 적용후

| Model                        | Accuracy (Before → After) | Precision (Before → After) | Recall (Before → After) | F1 Score (Before → After) | ROC AUC (Before → After) |
| ---------------------------- | -------------------------- | --------------------------- | ------------------------ | -------------------------- | ------------------------- |
| **LogisticRegression** | 0.8319 → 0.8320           | 0.6814 → 0.6815            | 0.6140 → 0.6142         | 0.6459 → 0.6461           | 0.8739 → 0.8743          |
| **RandomForest**       | 0.8601 → 0.9618           | 0.7606 → 0.9811            | 0.6416 → 0.8638         | 0.6960 → 0.9187           | 0.9190 → 0.9896          |
| **XGBoost**            | 0.8623 → 0.8907           | 0.7565 → 0.8243            | 0.6613 → 0.7145         | 0.7057 → 0.7655           | 0.9225 → 0.9460          |
| **LightGBM**           | 0.8636 → 0.8808           | 0.7794 → 0.8180            | 0.6329 → 0.6721         | 0.6985 → 0.7379           | 0.9237 → 0.9388          |

### 모델별 성능비교 그래프
![4  기본튜닝비교](https://github.com/user-attachments/assets/6b7b124b-9f86-4b2e-b9fb-87391ab8a8f8)
![5  튜닝모델 AutoML 비교](https://github.com/user-attachments/assets/a6bfb180-59b8-4da4-86cd-8919c149ace7)


## 최종학습 모델 선정

```
F1 Score 비교 결과, MLP의 성능이 가장 좋았음
```

## 은행 이탈 예측 대시보드
![11111](https://github.com/user-attachments/assets/83b50709-b311-4285-87f3-419eb02175d6)
![22222](https://github.com/user-attachments/assets/49d983f8-fe83-4e05-a91d-afc16f9a7916)
![33333](https://github.com/user-attachments/assets/6017fe4a-67cc-485c-872b-0264f0877aa0)


## 인사이트 및 결론

### 예측 결과 인사이트
- 미혼일수록 이탈 비율이 높으므로 1인 예적금 상품 등 싱글을 대상으로 한 상품을 제공해야 함
- 여자일수록 이탈 비율이 높으므로 여성을 타겟한 마케팅이 필요함
- 회전잔액이 없을수록 이탈 비율이 높으므로 고객들의 은행거래금액을 높일 수 있는 혜택이나 인센티브를 제공해야 함

### 🏢 은행 입장에서의 이점

- 이탈 위험이 높은 고객을 미리 파악해 타겟 맞춤형 프로모션을 제공할 수 있음
- 맞춤형 금융상품 추천으로 고객 충성도를 높이고 이탈을 방지할 수 있음

### 🙇‍♀️ 소비자 입장에서의 이점

- 본인의 은행 이용 패턴에 맞는 혜택을 제공받을 수 있음
- 개선된 고객 경험을 얻을 수 있음


## 회고

| 팀원     | 한 줄 회고 |
| -------- | ---------- |
| 권성호   |            |
| 조성지   |            |
| 김이경   |            |
| 이석원   |            |
| 이용규   |            |
