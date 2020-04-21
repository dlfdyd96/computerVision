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

# MNIST
손으로 쓴 숫자영상을 구분한다.<br>
데이터 크기가 작고, 학습이 잘된다.<p>
## [2-layer : Softmax Regression](./MNIST_softmax.md)<br>
## [3-layer : Multi-Layer Perceptron](./MNIST_MLP.md)

### Data-Set
- Download : http://yann.lecun.com/exdb/mnist/<br>
```python
# 옛날 자료 보면 이렇게 되어있다. (Package 안에 포함 됨)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=true)
# 요즘은 파일 경로 error 가 발생 할 경우, 자동으로 MNIST data를 다운까지 해줌
mnist = tf.keras.datasets.mnist.load_data(path='./data/mnist.npz')
```
- Data(**샘플**) : 28x28 Pixels = 784개의 **특징벡터** (0~255)
### NPZ MNIST Data Read
```python
mnist = np.load('./data/mnist.npz')

x_train = mnist['x_train']
x_test = mnist['x_test']
y_train = mnist['y_train']
y_test = mnist['y_test']
```
#### OpenCV, PIL 이용해서 matplot에 이미지 출력하는 방법
1. OpenCV : `img[:,:,::-1]`  bgr 순인 opencv를 rgb 순인 matplot으로 바꿔서 출력
2. PIL : `np.array(PIL.Image.open('cat.jpg'))`
### 레이블을 원핫 벡터로 바꾸기
*함수*
```python
def one_hot(labels, targets=10):
    samples=len(labels)
    out=np.zeros((samples,targets))
    out[range(smaples), labels] = 1
    return out
y_train_onehot = one_hot(y_train, 10)

y_train_onehot[0]
>>> array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])
```
*tensorflow*
```python
tf.one_hot([0, 1, 2, 0], depth=3)
>>> array([1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
          [1, 0, 0])
```
### 참고
GitHub : https://ml4a.github.io/ml4a/neural_networks/<br>
Neural Network : http://neuralnetworksanddeeplearning.com/<br>
MNIST 파일 다운로드 : https://s3.amazonaws.com/img-datasets/mnist.npz<br>
NPY, NPZ 다루기 : https://blog.naver.com/nonezerok/221904297903

<br>

___
## Memo
* **Learning**
   - Loss 그래프의 수렴을 통해 학습을 더해야할지 말지 판단
   - Iteration을 증가했음에도 불구하고, Loss가 안떨어진다면 **learning_rate**를 좀더 줄인다.
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
  - matplot.lib.pyplot.imshow() : 이미지 보여줌
  - plt.rcParams['figure.figsize']=(12,9) : 이미지를 크게 보여준다. (인치 단위)
  - plt.xticks([]) : 가로 숫자를 없애준다.
  - plt.imshow()\plt.show() : 이미지 보여줌 