# [2] Hinton, Geoffrey E., Simon Osindero, and Yee-Whye Teh. "A fast learning algorithm for deep belief nets." Neural computation 18.7 (2006): 1527-1554. [pdf](http://www.cs.toronto.edu/~hinton/absps/ncfast.pdf) (Deep Learning Eve)

- 그 유명한 제프리 힌튼의 인공신경망 심폐소생술 논문이다.
- 딥러닝 용어가 등장한 의미있는 문헌이다.
- 2006년, 인공신경망이 외면당하던 암흑기에도 꿋꿋이 연구를 이어왔던 제프리 힌튼의 논문이다.
- 인공신경망이라는 단어가 들어가기만해도 reject되던 시기라서, Deep을 붙인 DNN(Deep Neural Network)라는 용어를 사용했는데, 이 때부터 본격적으로 딥러닝이라는 용어가 사용되기 시작했다.
- 개인적으로는 딥러닝의 인식을 바꾼 가장 큰 사건 중 하나라고 느껴진다.
- 그 다음으로는 이전 리뷰에서도 언급한 2012년도 ImageNet 대회에서 (또) 제프리 힌튼 팀의 AlexNet으로 자신의 주장을 증명해버린 사건을 꼽고 싶다.
- 2022년의 가장 핫했던 단어, [중요한 것은 꺾이지 않는 마음] 이라는 글에 가장 잘 어울리는 연구 중 하나라고 생각된다.
- 계획보다 너무 오래걸린(이미지 삽입과 마크다운 수정까지 장장 6시간에 걸친) 첫 리뷰에 너덜너덜해져서 듬성듬성 리뷰할까 했는데, 바로 다음 문헌이 이거라서 마음을 다잡고 리뷰를 진행할 수 있었다.
- [원문 링크](http://www.cs.toronto.edu/~hinton/absps/ncfast.pdf)

# 0. Abstract
1. 숨겨진 레이어가 많고, 빽빽하게 연결된 신경망에서 추론을 어렵게 만드는 "explaining away effects를 없애기 위해 "complementary priors"를 사용하는 법을 소개한다.
2. Greedy 알고리즘을 쓰면 매우 많은 파라미터를 가진 매우 깊은 신경망에서도 파라미터를 빠르게 찾아낼 수 있다.(학습 과정을 초기화 하는데 사용)
3. 이 생성 모델은 MNIST 손글씨 데이터셋에 대해서 이전 분류 알고리즘들보다 훨씬 우수한 성능을 보인다.
4. 숫자들이 있는 저차원의 manifold들은 최상위의 연관 메모리의 자유 에너지 환경에 있는 "긴 협곡"으로 모델링 된다.
5. 연관 메모리가 고려하고 있는(what associative memory has in mind) 부분을 표시하기 위해서 직접 연결을 사용해 "긴 협곡"을 쉽게 탐색할 수 있다.

# 1. INTRO
- 이전의 연구
  - 데이터 벡터가 주어졌을 때의 숨겨진 활동들의 조건부 분포를 추론하기 어렵기 때문에, 빽빽하게 연결되고, 많은 숨겨진 층을 갖고 있는 DBN(directed belief net)을 학습시키는 것은 어렵다.
  - Variational methods는 실제 조건부 분포에 대해 간단한 근사치를 사용하지만, 독립성을 가정하는 깊은 층에서는 근사치가 몹시 안 좋다.
  - 또, 모든 파라미터들을 함께 학습시켜야하기 때문에, 파라미터 수가 증가하면 시간이 미친듯이 늘어난다.
- 이 연구에서의 제안

    <img src = ./imgs/1.png width="400" height="400">

  - 최상층에 있는 2개의 히든 레이어가 방향성이 없는 연관 메모리를 형성하고 [그림 1], 나머지 히든 레이어들이 연고나 메모리의 표현을 이미지의 픽셀과 같은 관찰 가능한 변수로 변환하는 방향성 비순환 그래프(하이브리드 모델)를 제안한다.
- 하이브리드 모델의 장점
  1. 그리디 알고리즘을 써서 빠르게 좋은 파라미터를 찾아낸다.
  2. 학습 알고리즘은 비지도이지만, 라벨과 데이터를 동시에 생성하는 모델을 학습해서 라벨도 있는 데이터에서도 쓸 수 있다.(지도학습에도 쓸 수 있음! 학계는 추후 비지도보다 지도학습에 더 집중하게 됨)
  3. 손글씨 MNIST 데이터셋에서의 기존 방법보다 훨씬 뛰어난 학습을 하는 fine-tuning 알고리즘이 있다.
  4. 생성 모델(generative model)을 쓰면, 깊은 히든 레이어의 분산 표현을 쉽게 해석할 수 있다.
  5. Percept를 형성함에 있어 필요한 추론이 빠르고 정확하다.
  6. 학습 알고리즘이 local이다. 시냅스 강도의 조정이 시냅스 전 뉴런과 시냅스 후 뉴런의 상태에만 의존한다.
  7. 의사소통 수단이 간단하다. 뉴런이 stochastic(확률적인) 이진 상태를 전달하기만 하면 된다.

# 2. Complementary Priors

- Explaining away?

   <img src = ./imgs/2.png width="400" height="400">

  - "Explaining away" 현상[그림 2]은 방향성 신경망에서 추론을 어렵게 만든다.
  - 체한 것 같아서 의사한테 가서 진단을 받는 상황을 예로 들어보자!
    - 소화불량 증상이 보이는 상황에선 과식했을 확률이 80%로 생각된다.
    - 그런데 직전에 갑작스런 운동을 했으면 과식했을 확률이 30%로 줄어든다.
    - 원인이 과식이 아니라 갑작스런 운동이라는 독립적인 변수에 의해 소화불량이 일어난다.
  - 이렇게 Explaining away 효과가 발생하면 서로 독립인 변수인 '과식(원인1)'과 '갑작스런 운동(원인2)'이 '소화불량 증상(결과)'에 영향을 미치기 때문에 추론이 어려워지게 된다!

- Logistic belief net(Neal, 1992)
  - 로지스틱 신경망은 확률론적 신경망으로 구성된다.
  - 신경망이 데이터를 생성하는데 사용되는 경우, unit i를 켤 확률은 부모 노드인 j가 켜졌는지의 여부와 가중치 w ij를 이용한 로지스틱 함수가 된다.
  - 로지스틱 신경망이 은닉층을 하나만 가지면 은닉 노드들이 서로 독립적으로 되어 prior distribution이 독립 분포가 되지만, posterior distribution은 비독립이 되는데, 이는 데이터에서 오는 likelihood 때문이다.
  - 이와 반대 상관 관계를 갖는 "Complementary" prior를 생성하기 위해 추가 히든 레이어를 사용하면, 첫 번째 히든 레이어에서 발생하는 "Explaining away" 현상을 없앨 수 있다!
  - 그런 다음 Likelihood와 prior가 곱해지면 독립 분포로 표현되는 posterior를 구할 수 있다.
## 2.1. Infinite Logistic Belief Net with Tied Weights

   <img src = ./imgs/3.png width="400" height="400">

- 무한히 깊은 히든 레이어 1에서 무작위 구성으로 시작한 다음, 레이어에서 각 변수의 이진 상태가 선택되는 하향식 "anscestral" pass를 수행해서 [그림 3]의 무한 방향성 네트워크에서 데이터를 생성할 수 있다.
- 베르누이 분포에 따라 위 레이어의 활성화된 부모노드에서 오는 하향식(top-down) 방식으로 데이터가 생성된다.
- 이 부분은 다른 방향성 비순환 신경망과 비슷하지만, 가시적 단위의 데이터 벡터로 시작한 다음 전치된 가중치 행렬을 사용하여 각 히든 레이어에 대한 팩토리얼 분포(factorial distributions)를 차례로 추론함으로써 모든 히든 레이어에 대한 실제 posterior distribution을 샘플링할 수 있다.
- 실제 posterior distribution으로부터 샘플을 만들어, 데이터와 얼마나 다른지 계산하면 데이터 v0 로그 확률의 도함수(미분값)를 계산할 수 있다.
- 단일 데이터 벡터 v0에 대한 최대 likelihood 학습 규칙은 다음과 같다.
  
   <img src = ./imgs/2-2.png width="600" height="100">

- 우변은 샘플링된 값들의 평균을 나타낸다.
- ^vi0는 가시 벡터 vi0가 샘플링된 히든 레이어로부터 재구성(재생성)된 경우 유닛 i가 켜져 있을 확률이다.
- ^vi0와 vi0가 가까울 수록 샘플링된 히든 레이어로부터 실제 가시 벡터를 잘 재생성할 수 있고, 모델은 이러한 방향으로 학습된다.
- 첫 번째 히든 레이어 H0의 샘플링된 이진 상태에서 두 번째 히든 레이어 V1에 대한 posterior distribution을 계산하는 것은 데이터를 재구성하는 것과 정확히 동일한 프로세스이기 때문에, v1i는 확률이 ^v0i인 베르누이 확률 변수의 샘플이다.
  
   <img src = ./imgs/2-3.png width="600" height="100">
   
- 때문에 수식은 [그림 2.3]으로 바뀔 수 있다.
- h0j에 대한 v1i의 종속성(비독립성)은 문제가 되지 않는다
- 가중치가 복제되기 때문에 생성 가중치에 대한 전체 도함수(미분값)는 모든 레이어 쌍(all pairs of layers) 사이에서 생성 가중치의 도함수(미분값)을 합산해 얻는다.
  
   <img src = ./imgs/2-4.png width="600" height="100">

- 식을 풀어쓰면 이렇게 되는데, 첫 항과 마지막 항을 빼곤 서로 상쇄돼서 모두 없어지고, [그림 3.1]의 볼츠만 머신 수식과 같아지게 된다.
   <img src = ./imgs/3-1.png width="600" height="100">


# 3. Restricted Boltzmann Machines and Contrastive Divergence Learning (RBM과 대조적 발산)
- RBM?
  - Infinite Logistic Belief Net은 위와 같이 RBM과 유사하다.
  - RBM에는 서로 연결되지 않은 숨겨진 단일 레이어가 있으며, 드러나있는 레이어에 방향이 지정되지 않은 대칭 연결이 있다.
  - RBM에서 데이터를 생성하기 위해 레이어 중 하나에서 임의의 상태로 시작한 다음 번갈아가며 Gibbs sampling을 수행할 수 있다.
  - 한 레이어에 있는 모든 유닛은 다른 레이어에 있는 유닛의 현재 상태가 주어지면 병렬로 업데이트되며 시스템이 equilibrium distribution에서 샘플링할 때까지 반복된다.
  - 이게 가중치가 묶인 Infinite Logistic Belief Net에서 데이터를 생성하는 과정과 정확히 똑같은 프로세스라는 점이 중요하다!!!
  - RBM에 대한 대략적인 설명으로는
    - 1) 입력값이 들어온다
    - 2) Forward로 생성된 출력값으로부터
    - 3) Backward로 생성된 입력값을 구하여
    - 4) 1)과 3)이 같아질 때까지 반복하면서 가중치(w)를 초기화시킨다.

    <img src = ./imgs/4.png width="600" height="400">


# 4. A Greedy Learning Algorithm for Transforming Representations (표현 변환에 있어서의 탐욕적 학습 알고리즘)
- 복잡한 모델을 학습하는 효율적인 방법은 순차적으로 학습되는 간단한 모델 세트를 결합하는 것이다.
- 시퀀스의 각 모델이 이전 모델과 다른 것을 학습하도록 하기 위해 각 모델이 학습된 후 데이터가 어떤 방식으로든 수정된다.
- Greedy Algorithm?
  - 시퀀스의 각 모델이 데이터의 다른 표현을 받아낼 수 있도록 하는 것
  - 모델은 입력 벡터에 대해 비선형 변환을 수행하고, 시퀀스의 다음 모델에 대한 입력으로 사용될 벡터를 출력으로 생성

    <img src = ./imgs/5.png width="400" height="400">

  - [그림 5]는 상위 2개 레이어가 무방향 연결(undirected connections)을 통해 상호 작용하고 다른 모든 연결은 방향이 있는 다층 생성 모델을 보여준다.
  - 상단의 무방향 연결은 묶인 가중치가 있는 무한히 많은 상위 레이어를 갖는 것과 같다.
  - 레이어 내 연결이 없으며 분석을 단순화하기 위해 모든 레이어에 동일한 수의 유닛이 있다.
  - 상위 레이어 사이의 매개 변수가 W0에 대한 complementary prior를 구성하는데 사용될 것이라고 가정하여 매개변서 W0에 대한 합리적인(최적까지는 아니지만..)값을 학습하는 것이 가능하다.
  - 이 가정 하에서 W0를 학습하는 작업은 RBM을 학습하는 작업으로 축소되며, 이는 여전히 어렵지만 Contrastive divergence를 최소화하여 그나마 좋은 근사값을 솔루션으로 얻을 수 있다.
  - W0가 학습되면 WT0를 통해 데이터를 매핑해 첫 번째 숨겨진 레이어에서 더 높은 수준의 "데이터"를 생성할 수 있다.

- RBM이 원본 데이터의 완벽한 모델인 경우 상위 수준의 "데이터"는 이미 상위 수준의 가중치 매트릭스에 의해 완벽하게 모델링된다.
- 하지만 일반적으로 RBM은 원본 데이터를 완벽하게 모델링할 수 없으며 다음과 같은 그리디 알고리즘을 사용하여 생성 모델을 개선할 수 있다.

  1. 모든 가중치 행렬이 같다고 가정하고 W0를 학습한다.
  2. W0를 고정하고 WT0를 사요하여 첫 번째 히든 레이어의 변수 상태에 대한 factorial approximate posterior distributions(계승 근사 사후 분포?)를 추론한다. 이후 더 높은 수준의 가중치가 변경되어 이 추론 방법이 더 이상 올바르지 않은 경우에도 마찬가지이다.
  3. 가중치가 더 높은 모든 행렬을 서로 연결하되 W0에서 분리한 상태에서 WT0를 사용하여 원본 데이터를 변환하여 생성된 상위 수준의 "데이터"의 RBM모델을 학습한다.
- 이 그리디 알고리즘이 상위 가중치 행렬을 변경하면 생성 모델은 확실히 개선된다.

# 5. Back-Fitting with the Up-Down Algorithm
- 가중치 행렬을 한번에 한 계층씩 학습시키는 것은 효율적이지만 최적은 아니다.
- 상위 레이어의 가중치가 학습되면 가중치나 간단한 추론 절차가 하위 레이어에 적합하지 않다.
- Greedy learning에 의해 생성된 suboptimality(준 최적성)는 부스팅과 같은 지도학습 방법들에 비해 상대적으로 innocuous하다.
  - 레이블이 부족한 경우가 대부분이고, 각 레이블은 매개변수에 대해 몇 비트의 제약만 제공할 수 있으므로 일반적으로 과적합(overfitting)이 과소적합(underfitting)보다 더 큰 문제가 된다.
  - 따라서 이전 모델로 돌아가서 다시 적용하면 득보다 실이 더 많아지게 된다.
- 하지만 비지도 학습은 레이블이 지정되지 않은 매우 큰 데이터셋을 사용할 수 있으며, 각 케이스는 매우 고차원적일 수 있으므로, 생성 모델에 많은 제약 조건을 제공한다.
- 과소적합은 심각한 문제로, 먼저 학습된 가중치가 나중에 학습된 가중치에 더 잘 맞도록 수정되는 후속 단계의 back-fitting으로 완화될 수 있다.

- "Down-pass"
  - 최상위의 연관 메모리 상태에서 시작해서 하향식 생성 연결을 사용하여 확률론적으로 각 하위 레이어를 차례대로 활성화한다.
  - 이 과정 동안 최상위 무방향 연결과 생성 방향 연결은 변경되지 않는다.
  - 상향식 인식 가중치만 수정된다.
    - 이것은 연관 메모리가 "down-pass"를 시작하기 전에 평형 분포에 정착하도록 허용되는 경우 "wake-sleep"알고리즘의 sleep 단계와 동일하다.
    - 하지만 연관 메모리가 업 패스에 의해 초기화된 다음 다운 패스를 시작하기 전에 번갈아가며 Gibbs 샘플링을 몇 번 반복하는 동안만 실행되도록 허용하면, 필요성을 제거하는 "대조적인(contrastive)" 형태의 "wake-sleep"알고리즘이 된다.
  - 대조적인 형태는 sleep 단계의 문제를 해결한다.
    - 실제 데이터에 사용한 것과 유사한 표현에 대해 "인식(recognition)" 가중치가 학습되고 있는지 확인하고 모드 평균화 문제를 제거하는 데도 도움이 된다.
  - 최상위 수준의 연관 메모리를 사용함으로써 wake 단계에서의 문제도 해결한다.
  - 독립적인 최상위 유닛은 ancester pass를 허용하는데 필요한 것처럼 보이지만 가중치의 최상위 레이어에 대한 variational approxiamation이 매우 나쁘다(poor)는 것을 의미한다.

# 6. Performance on the MNIST Database
## 6.1. Training the Network
- MNIST 손글씨 데이터
  - 공개적으로 사용 가능한 데이터베이스이다.
  - 6만개의 학습 데이터(training imgaes, 훈련 이미지)와 1만개의 테스트 데이터(이미지)가 있다.
  - 다양한 패턴 인식 기술에 대한 결과들이 공유되어 있어 새로운 인식 기술을 평가하는데 유용하다.
- 학습
  - 학습 초기 단계에서 4챕터에서 이야기한 그리디 알고리즘을 사용해, 가중치의 각 레이어를 개별적으로, 맨 아래부터 시작해서 학습했다.
  - 각 레이어는 트레이닝 셋("epoch"라고 함)을 통해 30회의 sweep동안 학습되었다.
  - 학습하는 동안 각 RBM의 "보이는(visible)" 레이어레 있는 단위는 0과 1 사이의 값의 activity을 가진다.
  - 이것은 가중치의 맨 아래 레이어를 학습할 때 정규화된 픽셀 세기(intensities)였다.
  - 더 높은 가중치 레이어를 학습하는 경우 RBM에서 보이는 유닛의 실제 값의 activity는 하위 수준의 RBM에서 히든 유닛의 활성화 확률이다.
  - 각 RBM의 히든 레이어는 해당 RBM이 학습될 때 확률론적인 이진값을 사용했다.
  - 레이블은 10개 단위의 "softmax"그룹에서 하나의 단위를 켜서 표시했다.
  - 이 그룹의 activity가 상위 레이어의 activity에서 재구성되었을 때 정확히 하나의 유닛이 활성화되도록 했다.
  - Greedy layer-by-layer training 이후, 네트워크는 5챕터에서 설명한 업-다운 알고리즘을 사용해 300에포크동안 다른 learinig rate(lr)와 weigth decay로 학습됐다.
  - 훈련한 후 나머지 1만개의 이미지로 성능을 검증했다.
    - 처음 100에포크 동안 업패스 다음에는 다운패스를 수행하기 전의 연관 메모리에서 번갈아가며 Gibbs 샘플링을 세 번 반복했다.
    - 두 번째 100에포크 동안 6회의 iteration이 수행되었다.
    - 마지막 100에포크 동안 10회의 iteration이 수행되었다.
    - Gibbs샘플링의 반복 횟수가 올라갈 때마다 validation셋의 오류가 눈에 띄게 줄었다.
    - Validation셋에서 가장 잘 수행된 네트워크 오류율은 1.39%였다.
    - 이후 최종 오류율이 낮아질때까지 훈련되었다.
    - 최종 오류율은 1.25%였다.

## 6.2. Testing the Network
- 네트워크를 테스트하는 한 방법은 이미지의 확률론적 업패스를 사용하여 연관 메모리의 하위 레이어에 있는 500개 단위의 binary states를 fix하는 것이다.
  - 이 상태가 되면 레이블 유닛에 0.1의 초기 실제 값 activity가 제공되고, 교대로 Gibbs샘플링을 몇 번 반복한 다음 올바른 레이블 유닛을 활성화 하는데 사용된다.
- 더 나은 방법은 연관(associative) 메모리의 하위 레이어에 있는 500유닛의 binary states를 고정한 다음 각 레이블 unit을 차례대로 켜고, 510-component binary 벡터의 정확한 자유 에너지 결과값을 계산하는 것이다.


# 7. Looking into the Mind of a Neural Network
- 모델에서 샘플을 생성하기 위해 Marcov 체인이 equilibrium distribution으로 수렴될 때까지 최상위 연관 메모리에서 번갈아가며 Gibbs 샘플링을 수행한다.
- 그 다음 이 분포의 샘플을 아래 레이어에 대한 입력으로 사용하고, generative connection을 통해 단일 다운 패스로 이미지를 생성한다.
- Gibbs 샘플링 중에 레이블 유닛을 특정 클래스로 고정하면 모델의 클래스 조건부 분포(class-conditional distributions)에서 이미지를 볼 수 있다.
- 임의의 binary 이미지를 입력으로 제공하여 상위 두 레이어의 상태를 초기화할 수도 있다.

    <img src = ./imgs/9.png width="400" height="400">

  - [그림 9]는 레이블이 고정된 상태에서 자유롭게 진행될 때 연관 메모리의 클래스 조건부 상태가 어떻게 진화하는지 보여준다.
  - 이 내부 상태는 연관 메모리가 고려하고 있는("has in mind") 것을 보기 위해 20회의 iteration마다 다운 패스를 수행하여 "관찰"된다.
    - 여기서 has in mind의 mind는 은유적인 표현이 아니라, 우리의 정신 상태와 같은 지각의 단계를 의미한다.


# 8. Conclusion
- 이 연구에서는 한 번에 한 레이어씩 깊고 조밀하게 연결된 신경망을 학습하는 것이 가능하다는 것을 보여줬다.
- 상위 레이어를 무시하는 대신, 상위 레이어가 존재하지만 실제 posterior를 정확히 factorial로 만드는 complementary prior를 구현하도록 제한적으로 묶인 가중치를 가지고 있다고 가정한다.
- 각 레이어가 학습된 후 해당 가중치는 상위 레이어의 가중치에서 풀려난다(untied).
  - 상위 가중치를 적용하면 전체 생성 모델이 개선된다는 것을 보여주기 위해 variational bound를 보여줄 수 있다.
- Fast, greedy learning algorithm의 힘을 입증하기 위해 숫자 이미지와 레이블의 우수한 생성 모델을 학습하는 훨씬 느린 fine-tuning 알고리즘의 가중치를 초기화했다.


