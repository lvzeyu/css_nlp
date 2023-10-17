#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
# 関数定義
def function_f(x):
    return x[0]**2 + x[1]**2
# 勾配
def numberical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x) # xと同じ形状の配列を生成
    for idx in range(x.size):
        # f(x+h)の計算
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val # 値を元に戻す
    return grad


# In[2]:




# Run gradient descent

# 勾配降下法の実装
def gradient_descent(f, init_x, lr=0.01, num_iterations=100):
    x = init_x
    x_history = [x.copy()]

    for i in range(num_iterations):
        grad = numberical_gradient(f, x)
        x -= lr * grad
        x_history.append(x.copy())

    return np.array(x_history)


# Parameters
init_x = np.array([-3.0, 4.0])
lr = 0.1
num_iterations = 20
x_history = gradient_descent(function_f, init_x, lr, num_iterations)

# 可視化
x = np.linspace(-4.5, 4.5, 200)
y = np.linspace(-4.5, 4.5, 200)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

plt.figure(figsize=(8, 6))
contour = plt.contour(X, Y, Z, levels=10, colors='black', linestyles='dashed')
plt.clabel(contour)

plt.plot(x_history[:, 0], x_history[:, 1], 'o-', color='red')
plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.title("Gradient Descent on $f(x_0, x_1) = x_0^2 + x_1^2$")
plt.show()


# In[3]:


# Parameters
init_x = np.array([-3.0, 4.0])
lr = 0.1
num_iterations = 20

# Run gradient descent
x_history = gradient_descent(function_f, init_x, lr, num_iterations)

# 可視化
x = np.linspace(-4.5, 4.5, 200)
y = np.linspace(-4.5, 4.5, 200)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

plt.figure(figsize=(8, 6))
contour = plt.contour(X, Y, Z, levels=10, colors='black', linestyles='dashed')
plt.clabel(contour)

# Adding arrows for each gradient
for i in range(1, len(x_history)):
    plt.quiver(x_history[i-1][0], x_history[i-1][1], 
               x_history[i][0]-x_history[i-1][0], 
               x_history[i][1]-x_history[i-1][1], 
               angles="xy", scale_units="xy", scale=1, color="blue")

plt.plot(x_history[:, 0], x_history[:, 1], 'o-', color='red')
plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("$X_0$")
plt.ylabel("$X_1$")
plt.title("Gradient Descent on $f(x_0, x_1) = x_0^2 + x_1^2$")
plt.show()


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# 既に定義済みのfunction_fとnumberical_gradientを利用します。

# 勾配降下法の実装
def gradient_descent(f, init_x, lr=0.01, num_iterations=100):
    x = init_x
    x_history = [x.copy()]

    for i in range(num_iterations):
        grad = numberical_gradient(f, x)
        x -= lr * grad
        x_history.append(x.copy())

    return np.array(x_history)

# Parameters
init_x = np.array([-3.0, 4.0])
lr = 0.1
num_iterations = 20

# Run gradient descent
x_history = gradient_descent(function_f, init_x, lr, num_iterations)

# 3D Visualization
x = np.linspace(-4.5, 4.5, 200)
y = np.linspace(-4.5, 4.5, 200)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
ax.contour(X, Y, Z, zdir='z', offset=0, cmap='viridis', linestyles='dashed')

# Plotting the path of gradient descent
Z_history = np.array([function_f(x) for x in x_history])
ax.plot(x_history[:, 0], x_history[:, 1], Z_history, 'o-', color='red')

# Adding arrows for each gradient in 3D
for i in range(1, len(x_history)):
    ax.quiver(x_history[i-1][0], x_history[i-1][1], Z_history[i-1], 
              x_history[i][0]-x_history[i-1][0], 
              x_history[i][1]-x_history[i-1][1], 
              Z_history[i]-Z_history[i-1], 
              color="blue", arrow_length_ratio=0.05)

ax.set_title("3D Visualization of Gradient Descent on $f(x_0, x_1) = x_0^2 + x_1^2$")
ax.set_xlabel("$X_0$")
ax.set_ylabel("$X_1$")
ax.set_zlabel("$f(X_0, X_1)$")
plt.show()


# In[ ]:




