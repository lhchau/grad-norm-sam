# Accelerated Sharpness-Aware Minimization using Adaptive Learning Rate

## Motivation 

Consider an objective function of the form $L(w) = \frac{h}{2}(w-z)^2 + C$ where $w$ is a scalar parameter. Assuming w is the current value of the parameter, what is the optimal $\eta$ that takes us to the minimum in one step? It is easy to visualize that, as it has been known since Newton, the optimal $\eta$ is the inverse of the second derivative of $L$, i.e. $1/h$. Any smaller or slightly larger value will yield slower convergence. A value more then twice the optimal will cause divergence.

In multidimension, things are more complicated. If the objective function is quadratic, the surfaces of equal cost are ellipsoids. Intuitively, if the learning rate is set for optimal convergence along the
direction of largest second derivative, then it will be small enough to ensure (slow) convergence along all the other directions. This corresponds to setting the learning rate to the inverse of the second derivative in the direction in which it is the largest. The largest learning rate that ensures convergence is twice that value. The actual optimal $\eta$ is somewhere in between. Setting it to the inverse of the largest second derivative is both safe, and close enough to the optimal.

The second derivative information is contained in the Hessian matrix $\partial^2 L(w) / \partial w_i \partial w_j$. The Hessian can be decomposed (diagonalized) into a product of the form $H=R \Lambda R^T$ where $\Lambda$ is a diagonal matrix whose diagonal terms (the eigenvalues of $H$) are the second derivatives of $L(w)$ along the principal axes of the ellipsoids of equal cost, and $R$ is a rotation matrix which defines the directions of these principal axes. The direction of largest second derivative is the principal eigenvector of $H$, and the largest second derivative is the corresponding eigenvalue (the largest one). In short, it can be shown that the optimal learning rate is the inverse of the largest eigenvalue of $H$:
$$
\eta_{opt} = \frac{1}{\lambda_{max}}
$$

**Taylor expansion**: Although it is often unrealistic to compute the Hessian $H$, there is a simple way to approximate the product of $H$ by a vector of our choosing:
$$
H v = \frac{\nabla L(W + \rho v) - \nabla L(W)}{\rho} + O(\rho^2)
$$

with the assumption $L$ is locally quadratic (.i.e. ignoring the $O(\rho^2)$ term), the product of $H$ by any vector $v$ can be estimated by subtracting the gradient of $L$ at point $(W + \rho v)$ from the gradient at $W$. 

In conclusion, the procedure for approximating largest positive eigenvalue of the second derivative matrix of the average objective function is

1. pick a normalized, $N$ dimensional vector $v$ at random. Pick two small positive constants $\rho$ and $\beta$
2. pick a training example (input and label) $X$. Perform a regular prop and a backward prop. Store the resulting gradient vector $G_1 = \nabla L(W)$
3. add $\rho \frac{G_1}{||G_1||}$ to the current weight vector $W$,
4. perform a forward loop and backward prop on the same training example $X$ using the perturbed weight vector. Store the resulting gradient vector $G_2 = \nabla L(W + \rho \frac{G_1}{||G_1||})$
5. update vector $v$ with the running average formula, $v \leftarrow (1 - \beta) v + \frac{\beta}{\rho}(G_2 - G_1)$
6. restore the weight vector to its original value $W$.
7. loop to step 2 until $||v||$ stablizes.
8. set the learning rate $\eta$ to $||v||^{-1}$, and go on to a regular training session.

This procedure is quite similar to update rule of Sharpness-Aware Miminization (SAM) optimizer, its formula is below:

1. $g_t = \nabla L(w_t)$
2. $w_{t+0.5} = w_t + \rho \frac{g_t}{||g_t||}$
3. $g_{t+0.5} = \nabla L(w_{t+0.5})$
4. $w_{t+1} = w_t - \eta g_{t+0.5}$

We utilize the perturbed gradient computed at each step in SAM to approximate the largest eigenvalue, its procedure can be illustrated below:

1. $g_t = \nabla L(w_t)$
2. $w_{t+0.5} = w_t + \rho \frac{g_t}{||g_t||}$
3. $g_{t+0.5} = \nabla L(w_{t+0.5})$
4. $h_t = \beta h_{t-1} + (1 - \beta) \frac{1}{\rho} (g_{t+0.5} - g_t)$
5. $w_{t+1} = w_t - \frac{\eta}{||h_t||} g_{t+0.5}$