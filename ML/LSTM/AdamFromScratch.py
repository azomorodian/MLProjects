import numpy as np
from numpy.random import permutation
class Line():
    def __init__(self):
        self.w0 = np.random.uniform(0,1,1)
        self.w1 = np.random.uniform(0,1,1)

    def evaluate(self,x):
        return self.w0+self.w1*x

    def dx_w0(self,x,y):
        yhat = self.evaluate(x)
        return 2*(yhat - y)

    def dx_w1(self,x,y):
        yhat = self.evaluate(x)
        return 2*x*(yhat-y)

    def __str__(self):
        return f"y= {self.w0[0]} + {self.w1[0]}*x"

def stochastic_sample(xs,ys):
    perm = permutation(len(xs))
    x = xs[perm[0]]
    y = ys[perm[0]]
    return x,y

def gradient(dx,xs,ys):
    N = len(ys)
    total = 0
    for x,y in zip(xs,ys):
        total = total + dx(x,y)
    gradient = total/N
    return gradient


def gd(model , xs, ys, lr=0.01, epochs=10000):
    for epoch in range(epochs):
        model.w0 = model.w0 - lr*gradient(model.dx_w0,xs,ys)
        model.w1 = model.w1 - lr*gradient(model.dx_w1,xs,ys)
        if epoch % 100 == 0:
            print(f"Iteration {epoch}")
            print(model)

def sgd(model , xs, ys, lr=0.01,epochs=10000):
    for epoch in range(epochs):
        x,y = stochastic_sample(xs,ys)
        model.w0 = model.w0 - lr*model.dx_w0(x,y)
        model.w1 = model.w1 - lr*model.dx_w1(x,y)
        if epoch % 100 == 0:
            print(f"Iteration {epoch}")
            print(model)

def sgd_momentum(model , xs, ys, lr,decay_factror,epochs):
    prev_g0 = 0
    prev_g1 = 0
    for epoch in range(epochs):
        x,y = stochastic_sample(xs,ys)
        g0 = decay_factror*prev_g0 - lr*model.dx_w0(x,y)
        g1 = decay_factror*prev_g1 - lr*model.dx_w1(x,y)
        model.w0 = model.w0 + g0
        model.w1 = model.w1 + g1
        prev_g0 , prev_g1 = g0 , g1
        if epoch % 100 == 0:
            print(f"Iteration {epoch}")
            print(model)
def adagrad(model , xs, ys,lr= 0.1 ,epochs=10000, eps = 0.0000001):
    G = [[0.0],
         [0,0]]
    for epoch in range(epochs):
        x,y = stochastic_sample(xs,ys)
        g0 = model.dx_w0(x,y)
        g1 = model.dx_w1(x,y)
        G[0][0] = G[0][0] - g0*g0
        G[1][1] = G[1][1] - g1*g1
        model.w0 = model.w0 - (lr/np.sqrt(G[0][0]+eps))*g0
        model.w1 = model.w1 - (lr/np.sqrt(G[1][1]+eps))*g1
        if epoch % 100 == 0:
            print(f"Iteration {epoch}")
            print(model)

def RMSprop(model , xs, ys, lr= 0.1 ,decay_factor = 0.9 , epochs=10000, eps = 0.0000001):
    E = [0,0]
    for epoch in range(epochs):
        x, y = stochastic_sample(xs,ys)
        g0 = model.dx_w0(x,y)
        g1 = model.dx_w1(x,y)
        E[0] = decay_factor*E[0] + (1 - decay_factor)*g0*g0
        E[1] = decay_factor*E[1] + (1 - decay_factor)*g1*g1
        model.w0 = model.w0 - (lr/np.sqrt(E[0]+eps))*g0
        model.w1 = model.w1 - (lr/np.sqrt(E[1]+eps))*g1
        if epoch % 100 == 0:
            print(f"Iteration {epoch}")
            print(model)
def adadelta(model, xs, ys,decay_factor = 0.9,epochs = 10000,eps = 0.0000001):
    E_g = [0.0]
    E_p = [0,0]
    delta_p = [0,0]
    for epoch in range(epochs):
        x, y = stochastic_sample(xs,ys)
        g0 = model.dx_w0(x,y)
        g1 = model.dx_w1(x,y)
        E_g[0] = decay_factor*E_g[0] + (1 - decay_factor)*g0*g0
        E_g[1] = decay_factor*E_g[1] + (1 - decay_factor)*g1*g1

        E_p[0] = decay_factor*E_p[0] + (1 - decay_factor)*delta_p[0]*delta_p[0]
        E_p[1] = decay_factor*E_p[1] + (1 - decay_factor)*delta_p[1]*delta_p[1]

        delta_p[0] = np.sqrt(E_p[0] + eps)/np.sqrt(E_g[0]+eps)*g0
        delta_p[1] = np.sqrt(E_p[1] + eps)/np.sqrt(E_g[1] + eps)*g1

        model.w0 = model.w0 + delta_p[0]
        model.w1 = model.w1 + delta_p[1]
        if epoch % 100 == 0:
            print(f"Iteration {epoch}")
            print(model)

def adam(model, xs, ys, learning_rate=0.1, b1 = 0.9, b2 = 0.999,epsilon = 0.00000001,max_iter = 1000):
    m = [0,0]
    v = [0,0]
    g = [0,0]
    t = 1
    for i in range(max_iter):
        x , y = stochastic_sample(xs,ys)
        g[0] = model.dx_w0(x,y)
        g[1] = model.dx_w1(x,y)

        m = [b1*m_i + (1-b1)*g_i for m_i,g_i in zip(m,g)]
        v = [b2*v_i + (1-b2)*(g_i**2) for v_i,g_i in zip(v,g)]

        m_cor = [m_i/(1-(b1**t)) for m_i in m]
        v_cor = [v_i/(1-(b2**t)) for v_i in v]

        model.w0 = model.w0 - (learning_rate/np.sqrt(v_cor[0]+epsilon))*m_cor[0]
        model.w1 = model.w1 - (learning_rate/np.sqrt(v_cor[1]+epsilon))*m_cor[1]
        t = t + 1
        if i % 100 == 0:
            print(f"Iteration {i}")
            print(model)

xs = np.linspace(1,10,10)
ys = 2*xs + 5
print(tuple(zip(xs,ys)))


model = Line()
print("Gradient Descent : ")
gd(model,xs,ys)
print("GD Optimizer")
print(f"{model}")

print("sgd")
model1 = Line()
sgd(model1,xs,ys)
print("SGD Optimizer")
print(f"{model1}")


print("adam")
model1 = Line()
adam(model1,xs,ys)
print("ADAM Optimizer")
print(f"{model1}")




