For a simple 1D mass spring system, using different numerical integrator would lead to different results.

The elastic restoring force of such system is $f(x) = -kx$, where $k$ is the stiffness coefficient.

### Explicit Euler

$$
\left\{\begin{matrix}
x_{i+1} = x_i + hv_i\\
v_{i+1} = v_i + h\frac{f(x_i)}{m}
\end{matrix}\right.
$$

we can easily get
$$
\begin{pmatrix}x_{i+1}\\v_{i+1}\end{pmatrix} = T\begin{pmatrix}x_{i}\\v_{i}\end{pmatrix} = \begin{pmatrix}1&h\\-\frac{hk}{m}&1\end{pmatrix}\begin{pmatrix}x_{i}\\v_{i}\end{pmatrix}
$$
The eigen values of matrix $T$​ are
$$
\lambda = 1\pm ih\sqrt{\frac{k}{m}}
$$
Therefore, after $n$ steps, the system would be
$$
\begin{align*}
& \begin{pmatrix}x_{n}\\v_{n}\end{pmatrix} = \begin{pmatrix}1&h\\-\frac{hk}{m}&1\end{pmatrix}^n\begin{pmatrix}x_0\\v_0\end{pmatrix}\\
& = \big(U\begin{pmatrix}\lambda_1&0\\0&\lambda_2\end{pmatrix}U^{-1}\big)^n\begin{pmatrix}x_0\\v_0\end{pmatrix}\\
& = U\begin{pmatrix}\lambda_1^n&0\\0&\lambda_2^n\end{pmatrix}U^{-1}\begin{pmatrix}x_0\\v_0\end{pmatrix}
\end{align*}
$$

The norm of eigen value is $1+\frac{hk}{m}$ , which is bigger than 1. Therefore, the result diverges.



### Semi-Implicit/Symplectic Euler

$$
\left\{\begin{matrix}
x_{i+1} = x_i + hv_{i+1}\\
v_{i+1} = v_i + h\frac{f(x_i)}{m}
\end{matrix}\right.
$$

similarly, we can get
$$
\begin{align*}
& \begin{pmatrix}x_{i+1}\\v_{i+1}\end{pmatrix} = \begin{pmatrix}x_i+h(v_j+h\frac{-kx_i}{m})\\v_j+h\frac{-kx_i}{m}\end{pmatrix}\\
& = T\begin{pmatrix}x_{i}\\v_{i}\end{pmatrix} = \begin{pmatrix}1-\frac{h^2k}{m}&h\\-\frac{hk}{m}&1\end{pmatrix}\begin{pmatrix}x_{i}\\v_{i}\end{pmatrix}
\end{align*}
$$
The eigen values of matrix $T$ are
$$
\lambda = \frac{2m-h^2k\pm\sqrt{h^4k^2-4mh^2k}}{2m}
$$
when $h^4k^2-4mh^2k > 0$, it is easily to get that $\lambda > 0$, therefore the result diverges

when $h^4k^2-4mh^2k < 0$, the norm of $\lambda$ is exactly 1, therefore the result would be stable.

In a summary, semi-Implicit Euler method is conditionally stable when $h < 2\sqrt\frac{m}{k}$

Though when $h < 2\sqrt\frac{m}{k}$​, the system is stable, the system does not follow the energy conservation. It can be checked by comparing the system energy before and after an iteration.

Before the iteration, the energy of the system is
$$
E_i = EK_i + EP_i = \frac{1}{2}mv_i^2 + \frac{1}{2}kx_i^2
$$
After the iteration, the energy of the system is
$$
E_{i+1} = \frac{1}{2}mv_{i+1}^2 + \frac{1}{2}kx_{i+1}^2 \\
=\frac{1}{2}[m(-\frac{hk}{m}x_i + v_i)^2 + k((1-\frac{h^2k}{m})x_i+hv_i)^2] \\
=\frac{1}{2}[(m+h^2k)v_i^2-\frac{2h^3k^2}{m}x_iv_i+(k-\frac{h^2k^2}{m}+\frac{h^4k^3}{m^2}x_i^2)] \\
= E_i + \frac{1}{2}h^2k[(v_i-\frac{hk}{m}x_i)^2-\frac{k}{m}x_i^2] \\
=E_i + \frac{1}{2}h^2k(v_{i+1}^2-\frac{k}{m}x_i^2)
$$
It is obvious that the new energy is different from the old energy, therefore, the system's energy is not conserved.

### Implicit Euler

$$
\left\{\begin{matrix}
x_{i+1} = x_i + hv_{i+1}\\
v_{i+1} = v_i + h\frac{f(x_{i+1})}{m}
\end{matrix}\right.
$$



for the first term
$$
x_{i+1} = x_{i} + h(v_i + h\frac{-kx_{i+1}}{m}) \\ 
x_{i+1} = \frac{m}{m+kh^2}x_i + \frac{mh}{m+kh^2}v_i
$$
for the second term
$$
v_{i+1} = v_j + h\frac{-kx_{i+1}}{m} \\ = v_i - \frac{kh}{m} (x_i+hv_{i+1})\\
v_{i+1} = -\frac{kh}{m+kh^2}x_i + \frac{m}{m+kh^2}v_i
$$
Therefore
$$
\begin{pmatrix}x_{i+1}\\v_{i+1}\end{pmatrix} = T\begin{pmatrix}x_{i}\\v_{i}\end{pmatrix} = \frac{1}{m+kh^2}\begin{pmatrix}m&mh\\-kh&m\end{pmatrix}\begin{pmatrix}x_{i}\\v_{i}\end{pmatrix}
$$

The eigen value of $T$ is
$$
\lambda = \frac{m\pm ih\sqrt{mk}}{m+kh^2}
$$
The norm of eigen values is $\|\lambda\| = \frac{m}{m+kh^2}<1$​. Therefore, the result would converges to 0.​

