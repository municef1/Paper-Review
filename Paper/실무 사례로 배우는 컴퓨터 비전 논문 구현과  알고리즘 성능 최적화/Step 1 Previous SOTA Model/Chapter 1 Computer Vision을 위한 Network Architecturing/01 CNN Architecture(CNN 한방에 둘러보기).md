# 01. CNN Architecture(CNN 한방에 둘러보기)

# ImageNet Classification with Deep Convolutional Neural Networks

## AlexNet Architecture

- Convolution layer 5개 + fully connected layer 3개

  <img src = ./imgs/1.png width="600" height="600">

### 1) Convolution이란?

  <img src = ./imgs/2.png width="600" height="200">

  <img src = ./imgs/3.png width="600" height="200">

  <img src = ./imgs/4.png width="600" height="200">

  <img src = ./imgs/5.png width="600" height="200">


위 > 아래 순으로 그림을 보자.
위부터 한칸씩 읽어나간다! (Stride = 1)
커널의 크기를 보자. 3 * 3이므로 3 * 3 크기로 input을 읽어나간다!
딥러닝에선 언제나 출력의 shape가 중요하다.
5 * 5 input을 Kernel = 3 * 3, stride = 1로 읽으면 어떤게 만들어질까?
그림과 같이 3 * 3 사이즈의 shape가 출력으로 만들어진다!

- Convoluion 동작
- stride가 1이면 한칸씩, 2면 2칸씩 이동
- Input과 Kernel을 행렬연산해서 이동해나가는 방식임.
- Input에 커널 사이즈를 곱한 뒤 더해나가면서 통과시키고, Stride만큼 이동해나가는 방식.
- 때문에 Output의 크기는 Input의 크기와 Kernel의 크기, Stride의 크기에 의해 결정됨을 알 수 있다.
- Pytorch에서는 다음과 같이 코드를 쓸 수 있다.
    - torch.nn.Conv2d(in_channels, out_channels, kernel_size, ..)

### 2) Activation Function이란?

- 활성함수라고 부르며, 비선형 함수임.
- 입력을 받아서 활성화, 비활성화 여부를 결정함.
- Sigmoid?
    

  <img src = ./imgs/6.png width="600" height="300">
    
- 모든 값을 0과 1 사이로 만들어주기 때문에, 확률 문제에서 많이 애용된다.
    - sigmoid(x) = 1 / (1+e^(-x))
    - 모든 실수 값을 0에서 1 사이의 미분 가능한 수로 만들어주기 때문에 분류문제와 비용함수 문제에서 많이 사용함
    - 또, 리턴 값이 확률값이기 때문에 결과를 확률로 해석하기 용이함.

- ReLU(Rectified Linear Unit)
    
  <img src = ./imgs/7.png width="600" height="300">

- 0보다 작은 값에 대해 모두 0으로 처리하기 때문에 연산 속도가 빨라진다.
0보다 작은 입력에 대해 뉴런이 죽는 단점이 있지만, 구현이 쉬워 많이 사용된다.
    - f(x) = max(0,x)
    - Sigmoid가 갖는 gradient vanising문제를 해결하기 위해 제일 먼저 나온 함수
    - x가 0보다 크면 기울기가 1인 직선, 0보다 작으면 함수값이 0이 되어, 뉴런이 죽는 단점이 생길 수 있음
    - 하지만, 연산량이 적어 학습이 빠르고 구현이 매우 쉬워 많이 사용됨.

- Leaky ReLU
    

  <img src = ./imgs/8.png width="600" height="300">
    
- ReLU와는 다르게 0보다 작은 x에 대해 0으로 처리하지 않는다!
때문에 특정 상태에 대해 뉴런이 죽지 않는다!
    - fa(x) = max(ax, x)
    - ReLU가 0에서 뉴런이 죽는 문제를 해결하기 위해 나옴.
    - 0보다 작은 x에 대해 0으로 만들지 않고, 미분값이 0이 되지 않게 만듦.

- Softmax
    

  <img src = ./imgs/9.png width="600" height="300">
    
- 개 고양이 돼지가 타겟이면, output이 개 70 고양이 25 돼지 5로 나온다고 치자.
그러면 개일 확률이 70%라고 판단할 수 있게 된다!
이렇게 다중 분류에서 사용된다!
    - 모든 output을 더했을 때, 합이 1이 되는 특징.
    - Input값이 여러 개이기 때문에 활성화 함수이지만 정형화된 그래프를 갖지 않음.
    - 모두 0과 1 사이가 되며, 합은 1이 되도록 만듦.
    - 다중 분류에 쓰임.

### 3) Pooling이란?

- 앞의 과정을 거쳐서 나온 모든 데이터가 필요하지 않기 때문에 사용.
- 파라미터를 줄이기 때문에 해당 네트워크의 표현력이 줄어들어 과적합을 방지함.
- 계산량이 줄어들어 빨라지고, 처리할 자원 소모가 적음
- MaxPooling
    

  <img src = ./imgs/10.png width="600" height="300">
    
- 빨간 filter 안에선 5가 최대값이므로, 5만 넘긴다!
초록 filter 안에선 8이 최대값이므로, 8만 넘긴다!
또, stride가 2인 경우, 2 * 2의 필터를 2칸 이동해서 적용시킨다!
    - 2 * 2 필터에 Stride 2짜리면, 2 * 2만큼 읽고, 그 안에서 가장 큰 수만 뽑음. 2칸 이동해서 또 가장 큰 수만 뽑음

- AvgPooling
    
  <img src = ./imgs/11.png width="600" height="300">

- 빨간 filter 안에 있는 수의 평균을 구한다! 1 2 4 5의 합은 12이며, 4칸으로 나누면 3이므로, 결과는 3!
    - 최대값 대신 평균값을 취함.
    - 덜 중요한 요소를 포함할 수 있음.
    - 대신 분산을 사용할 수 있기 때문에, 물체의 위치를 보다 잘 파악 가능.
    - 객체 탐지에서 사용하기 용이함.
- GlobalAvgPooling(GAP)
    

  <img src = ./imgs/12.png width="600" height="300">
    
- 전체의 평균을 구한다! 전체를 보기 때문에 특정 부분에 대한 과적합을 피할 수 있다!
사람으로 치면 눈에 초점을 아예 풀고 쳐다본다고 생각하자!
    - 전체의 평균을 취함. 인풋을 하나로 만들어버림
    - CNN모델에서 과적합을 피해서 feature를 포착할 수 있음.

### 4) Dropout?


  <img src = ./imgs/13.png width="600" height="400">

- 과적합을 피하는 방법 중 하나.
    - 과적합은 데이터가 적기 때문에 생김.
- 일부 뉴런을 비활성화해서 사용함.
- 모델의 결합이 여러 개의 형태를 띄도록 만들어주는 것이라 생각할 수 있음.

### 5) Fully Connected Layer?

  <img src = ./imgs/14.png width="600" height="500">

- 왼쪽의 표준 신경망과 오른쪽 dropout이 적용된 신경망의 비교.
특정 뉴런들이 비활성화 되어있다.
- 완전히 연결됐다는 뜻.
- 한 Layer의 모든 뉴런이 다음 Layer의 모든 뉴런과 연결된 상태로 진행하는 형태임.