x = [0, 1, 2]
y = [1, 3, 2]
# 행렬 연산을 하자. x,y를 변경
X = [[0, 1], [1, 1], [2, 1]]
Y = [[1], [3], [2]]
# 계산을할때 inverse를 계산.(package 이용)
import numpy as np

X = np.array(X) # numpy X 가 된다.
Y = np.array(Y) # numpy Y 가 된다.

# X^T*X 를 구해주자.
xtx = np.matmul(X.T, X)
# 이걸 또 역행렬해주자.
xtx_inv = np.linalg.inv(xtx)

# W는
W = np.matmul(np.matmul(xtx_inv, X.T), Y)
print(W)