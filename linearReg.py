import numpy as np
import matplotlib.pyplot as plt
x = np.array([1,2,3,4,5,6])
y = np.array([1.5,4.5,5.8,9.1,11,11.5])
w0,w1 = np.random.rand(2)
print(w0,w1)
y_pred = []
epochs = 1000
lr = 0.01
error = []
for epoch in range(epochs):
    epoch_cost, cost_w0, cost_w1 = 0, 0, 0
    for i in range(len(x)):
        y_pred = w1*x[i]+w0
        epoch_cost += (y[i]-y_pred)**2

    for j in range(len(x)):
        partial_wrt_w0 = -2*(y[j]-(w1*x[j]+w0))
        partial_wrt_w1 = -2*(x[j])*(y[j]-(w1*x[j]+w0))

        cost_w0 += partial_wrt_w0
        cost_w1 += partial_wrt_w1
    w0 = w0-cost_w0*lr
    w1 = w1-cost_w1*lr

    print("w1 = ",w1)
    print("w0 = ",w0)

    error.append(epoch_cost)
    y_pred = []

y_pred = [w1*1+w0,w1*2+w0,w1*3+w0,w1*4+w0,w1*5+w0,w1*6+w0]

print(y_pred)

plt.scatter(x,y, label = "actual")
plt.plot(x,y_pred, c = "r", label = "predicted")
plt.legend()
plt.show()
