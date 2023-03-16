
### Basic Knowledge

- Derivatives with a Computation Graph
<img src="/images/Derivatives_with_a_Computation_Graph.png" width="500" /> 

- Derivatives of activation functions
<img src="/images/derivatives_sigmoid
.png" width="500" /> 
<img src="/images/derivatives_tanh
.png" width="500" /> 
<img src="/images/derivatives_relu
.png" width="500" /> 

#### Logistic Regression as a Neural Network

- Recap
    - For one example $x^{(i)}$:
    $z^{(i)} = w^T x^{(i)} + b$
    $\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})$ 
    $ \mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})$

    - The cost is then computed by summing over all training examples:
    $ J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})$

- Logistic regression derivatives
<img src="/images/Logistic_regression_derivatives.png" width="500" /> 

- Logistic Regression Gradient Descent
    - Forward Propagation
    $A = \sigma(w^T X + b) = (a^{(1)}, a^{(2)}, ..., a^{(m-1)}, a^{(m)})$
    $J = -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)}))$

    - Backward propagation
    $ \frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T$
    $ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})$

    - $ \theta = \theta - \alpha \text{ } d\theta$


#### Shallow Neural Network

- Model Structure
<img src="/images/2_layers_nn.jpg" width="500" /> 

- Forward Propagation
    $Z^{[1]} =  W^{[1]} X + b^{[1]}$
    $A^{[1]} = \tanh(Z^{[1]})$
    $Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$
    $\hat{Y} = A^{[2]} = \sigma(Z^{[2]})$
    $J = - \frac{1}{m} \sum\limits_{i = 1}^{m} \large{(} \small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right) \large{)} \small$

- Backward propagation
<img src="/images/nn_gradients.png" width="500" /> 
<img src="/images/2_layers_backward.jpg" width="500" /> 

- Vectorizing Justificaton
<img src="/images/Vectorizing_logistics.png" width="500" /> 
<img src="/images/Justification_vectorized_implementation.png" width="500" /> 

- Random initialization
    - np.random.randn(a,b) * 0.01 to have a small parameters, avoid simoid's first iteration with small slope



























Normalizing Inputs
- 加快训练速度

Vanishing / Exploding Gradients


Logistic, Binary Cross Entropy Loss 
    VS One layer network, Cross Entropy Loss
https://zhuanlan.zhihu.com/p/38853901












TF-IDF
https://zhuanlan.zhihu.com/p/41091116

当一个词在文档频率越高并且新鲜度高（即普遍度低），其TF-IDF值越高。

TF-IDF兼顾词频与新鲜度，过滤一些常见词，保留能提供更多信息的重要词