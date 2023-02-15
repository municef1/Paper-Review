# 01. VGG, ResNet, DenseNet

# VGG classification

- 모든 Convolution layer에서 3 * 3 kernel을 사용함
- 16 ~ 19에 달하는 깊은 신경망을 학습이 가능
- VGG-19
    - 16 Convolution Layers + 3 Fully-connected Layers
- 다른 이전의 모델들과는 다르게 3 * 3 kernel 만을 사용해서 성능을 비약적으로 향상시킴
- 3개의 kernel 을 3번의 비선형 함수를 넣음
- 때문에 layer가 증가함에 따라 비선형성이 증가하고, 이것이 모델의 식별성 증가로 이어짐
- 각 Conv layer에는 Batch Normalization을 사용함
    - 평균을 낼 때 데이터를 배치 단위로 나눠 평균을 낼 데이터의 양을 줄여줌
    - 이 때 배치로 나뉜 분산값이 쏠리지 않도록 변형된 분포가 나오지 않도록 조절해주는 것
    - 때문에 학습 속도를 빠르게 할 수 있음
    - 가중치 초기화에 대한 민감도를 감소시켜줌
    - 모델의 regularization 효과를 기대할 수 있음

## ResNet

- 지금도 각종 대회에서 많이 쓰이는게 보임!
- Visual Recognition에서 Depth는 성능 향상에 있어 매우 중요한 요소임
- 하지만, depth의 증가는 여러 문제를 일으켰음
    - Overfitting
    - **Vanishing gradient**
    - **Exploding gradient**
    - 연산량의 증가 등
- 때문에 모델이 깊어지면 성능이 확 올라갔다가 다시 확 떨어져버리는 일이 발생함
- 이는 과적합 때문이 아니었음
- Layer 수가 더 추가되어 test 에러 뿐만 아니라, training 에러도 함께 높아졌기 때문이었음.
- 이런 문제를 해결한게 **Residual function**
    - 이전의 모델은 하나의 Convolution network의 output이 다음 network의 input으로 들어가는 형태
    - ResNet 모델에서는 입력이 출력에도 들어가는 숏컷을 만들어, 깊게 쌓을 수 있게 만들었음.
- **Bottleneck defsign**
    - 3-layer block(bottleneck design)을 사용하여 101-layer 및 152-layer ResNet을 구성했음
    - 더 높은 정확도를 기록함
    - 1 * 1, 64 / 3 * 3, 64 / 1 * 1, 256 구조로 되어있음

## DenseNet

- Dense connectivity
- 한 layer의 input feature를 이후 layer의 input feature에 concat 하는 방식
- Feature map들을 feed forward형태로 concat해주며, 이를 통해 ResNet에서 추가해야 할 정보와, 보존해야 할 정보가 명시적으로 구분되어 ResNet보다 좋은 성능을 보임
- CNN 모델들은 down-sampling을 위해 pooling 연산을 수행하는데, 이 과정을 거치고 나면 feature map 사이즈가 달라지고, 그러면 concat으로 feature map을 연결할 수가 없게 됨
- 이를 막기 위해 네트워크를 여러 개의 Dense Block들로 구성함
- Dense Block들 사이에 있는 layer들을 Transition layer라고 부름
- 이 Layer가 convolution, pooling을 수행하고, batch normalization, ReLU, 1 * 1 convolution, 2 * 2 average pooling 순서의 구조를 가짐
- 논문에서는 transition layer의 1 * 1 convolution에 줄일 채널 수를 결정하는 세터를 하이퍼 파라미터로 사용하고, 실험시에 0.5값을 사용하여 feature map 수를 반으로 줄였음