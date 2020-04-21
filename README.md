# computerVision
현장에서 사용되는 핵심 비전 알고리즘 소개.
<br>컴퓨터비전 응용에 적용할 수 있는 딥러닝 모델을 중점적으로 다룸.
<br>OpenCV, Tensorflow 등의 공개 라이브러리를 이용한 컴퓨터비전 관련 프로젝트 수행


## 환경
- OS : Window10
- Python : 3.7.7
- tensorflow : 2.1.0
- numpy : 1.18.3
- matplotlib : 3.2.1

<br>

___
## Learning
* Loss 그래프의 수렴을 통해 학습을 더해야할지 말지 판단
* Iteration을 증가했음에도 불구하고, Loss가 안떨어진다면 **learning_rate**를 좀더 줄인다.

<br>

___
## MNIST
손으로 쓴 숫자영상을 구분한다.<br>
데이터 크기가 작고, 학습이 잘된다.<p>
2-layer : Softmax Regression<br>
3-layer : Multi-Layer Perceptron

### Data-Set
- \* Download : http://yann.lecun.com/exdb/mnist/<br>
- \* Data : 28x28 Pixels (0~255)
### 2 layer : Softmax Regression
- 입력 : 784개<br>
- 출력 : 10개(0~9)
### 3 layer : Multi-Layer Perceptron
- 입력 : 784개<br>
- 출력 : 10개<br>
- hidden Node는 hyper Parameter(조정)
### 참고
https://ml4a.github.io/ml4a/neural_networks/<br>
http://neuralnetworksanddeeplearning.com/


<br>

___
## Memo
* TensorFlow
  - tf.compat.v1 : Bring in all of the public TensorFlow interface into this module.
  - tf.Variable : https://www.tensorflow.org/api_docs/python/tf/Variable
  - tf.compat.v1.Session : A class for running TensorFlow operations.
  - tf.math.reduce_mean : Computes the mean of elements across dimensions of a tensor.
  - tf.math.square : Computes square of x element-wise.

* Numpy
  - numpy-indexing : https://kongdols-room.tistory.com/58
  - argmax : list에서 최대값의 index 값

* Matplotlib
  - plt.controuf : 예뻐진다..?
  - (empty)