import math, random
import numpy as np 

def relu(x):
    return np.maximum(0.0, x)

def d_relu(x):
    return (x>0).astype(x.dtype)

def softmax(logits):
 logits = logits - logits.max(axis=1, keepdims=True) # stability
 e = np.exp(logits)
 return e / e.sum(axis=1, keepdims=True)

def one_hot(y, num_classes):
 Y = np.zeros((y.size, num_classes))
 Y[np.arange(y.size), y] = 1.0
 return Y

class MLP2:
    """
    2-layer MLP:
      hidden = ReLU(X @ W1 + b1)
      out    = hidden @ W2 + b2
    mode: 'regression' (MSE) or 'classification' (softmax CE)
    """
    def __init__(self, in_dim, hidden_dim, out_dim, mode='regression', seed=0):
        rng = np.random.default_rng(seed)
        k1 = math.sqrt(2.0 / in_dim)      # He init for ReLU
        k2 = math.sqrt(2.0 / hidden_dim)
        self.W1 = rng.normal(0, k1, (in_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.normal(0, k2, (hidden_dim, out_dim))
        self.b2 = np.zeros((1, out_dim))
        assert mode in ('regression', 'classification')
        self.mode = mode
 
    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.h1 = relu(self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2
        if self.mode == 'classification':
            self.probs = softmax(self.z2)
            return self.probs
        return self.z2  # regression raw output
 
    def loss(self, y):
        if self.mode == 'regression':
            # y: shape (N, out_dim)
            diff = self.z2 - y
            return 0.5 * np.mean(np.sum(diff * diff, axis=1))
        else:
            # y: integer labels shape (N,)
            N = y.shape[0]
            # cross-entropy
            logp = -np.log(self.probs[np.arange(N), y] + 1e-12)
            return np.mean(logp)
 
    def backward(self, y):
        N = self.X.shape[0]
        if self.mode == 'regression':
            # dL/dz2 = (z2 - y)
            dz2 = (self.z2 - y) / N
        else:
            # dL/dz2 = probs - onehot(y)
            Y = one_hot(y, self.z2.shape[1])
            dz2 = (self.probs - Y) / N
 
        # Gradients for layer 2
        dW2 = self.h1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
 
        # Backprop through ReLU
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * d_relu(self.z1)
 
        # Gradients for layer 1
        dW1 = self.X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)
 
        return dW1, db1, dW2, db2
 
    def step(self, grads, lr=1e-2, weight_decay=0.0):
        dW1, db1, dW2, db2 = grads
 
        # L2 regularization on weights (not biases)
        if weight_decay > 0:
            dW1 += weight_decay * self.W1
            dW2 += weight_decay * self.W2
 
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
 
 
def batch_iter(X, y, batch_size, shuffle=True, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        j = idx[i:i + batch_size]
        yield X[j], y[j]

def train(model, X, y, epochs=200, lr=1e-2, batch_size=64, weight_decay=0.0,
          verbose=True):
    losses = []
    for e in range(1, epochs + 1):
        for Xb, yb in batch_iter(X, y, batch_size, shuffle=True, seed=e):
            model.forward(Xb)
            grads = model.backward(yb)
            model.step(grads, lr=lr, weight_decay=weight_decay)
        # track loss full-batch
        model.forward(X)
        L = model.loss(y)
        losses.append(L)
        if verbose and (e % max(1, epochs // 10) == 0):
            print(f"epoch {e:4d}/{epochs}  loss={L:.4f}")
    return losses

