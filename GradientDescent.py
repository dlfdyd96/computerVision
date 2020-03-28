X = [0, 1, 2]
Y = [1, 3, 2]

# yhat = ax + b

a = 0.1
b = 0.0
#Learning Rate
LR = 0.1

Loss = []
for k in range(100): #반복횟수
    delta_a = 0
    delta_b = 0
    #lost(에러)가 얼마나 줄어드는지
    E = 0
    for i in range(3): #data가 3개
        yHat = a * X[i] + b
        delta_a += LR*(Y[i] - yHat) * X[i]
        delta_b += LR*(Y[i] - yHat) * 1
        E += (Y[i] - yHat)**2
        E /= 2
    Loss.append(E)
    a = a + delta_a
    b = b + delta_b
    print('{0} : a={1:0.4f} b={2:0.4f}, loss={3:0.8f}'.format(k, a, b, E))

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(Loss)
plt.show()