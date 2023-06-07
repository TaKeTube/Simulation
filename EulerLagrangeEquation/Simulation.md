# Rod in Gravitational field

![res](C:res.gif)

### Introduction

In this mini project, we simulate the dynamics of a rod with uniform mass distribution in a gravitational field. We first derive the Euler-Lagrange in a general coordinate system, then use forward Euler method to solve the ODE. The simulation program is written using [taichi programming language](https://github.com/taichi-dev/taichi), and we use auto diff of taichi to compute the differential of the general potential energy because the analytical form is very complex.

### Modelling

Anime Character: 歌愛ユキ @ [Yukopi - 強風オールバック (feat.歌愛ユキ)](https://www.youtube.com/watch?v=D6DVTLvOupE)

<img src="img\model.png" alt="model" style="zoom: 67%;" />

As shown in picture, we model the system with the following quantities:

- In origin $O$, there is a mass point with mass $M$.
- $r \in(0,\infty)$: distance from $O$ to the center of the rod $C$.
- $\theta\in[0,2\pi]$: angle of $C$ in polar coordinate.
- $\phi \in[0,2\pi]$: rotation angle of the rod with respect to the radius $r$ of $C$.
- $l$: half length of the rod, so the total length of the rod is $2l$.
- $m$: mass of the rod, so the density of the rod is $m/2l$.
- $s\in[-l,l]$: distance between a point on the rod and $C$.

### Derivation

Using homogenous coordinate and applying transform matrix, we get the position of each point on the rod 

$$
\
\begin{bmatrix}x \\ y \\ 1\end{bmatrix} = 
\begin{bmatrix}
\cos\theta & -\sin\theta &   \\
\sin\theta &  \cos\theta &   \\ 
          &            & 1 
\end{bmatrix}
\begin{bmatrix}
1 &  &  r \\
 &  1 &   \\ 
          &            & 1 
\end{bmatrix}
\begin{bmatrix}
-\cos\phi & \sin\phi  &   \\
-\sin\phi & -\cos\phi &   \\ 
         &         & 1 
\end{bmatrix}
\begin{bmatrix}s \\ 0 \\ 1\end{bmatrix}
\\=
\begin{bmatrix}
\cos\theta & -\sin\theta &   \\
\sin\theta &  \cos\theta &   \\ 
          &            & 1 
\end{bmatrix}
\begin{bmatrix}
1 &  &  r \\
 &  1 &   \\ 
          &            & 1 
\end{bmatrix}
\begin{bmatrix}-s\cos\phi \\ -s\sin\phi \\ 1\end{bmatrix}
\\=
\begin{bmatrix}
\cos\theta & -\sin\theta &   \\
\sin\theta &  \cos\theta &   \\ 
          &            & 1 
\end{bmatrix}
\begin{bmatrix} -s\cos\phi+r \\ -s\sin\phi \\ 1 \end{bmatrix}
\\=
\begin{bmatrix}
- s\cos\phi\cos\theta + s\sin\phi\sin\theta + r\cos\theta \\
- s\cos\phi\sin\theta - s\sin\phi\cos\theta + r\sin\theta \\
1
\end{bmatrix}
\\=
\begin{bmatrix}
- s\cos(\phi+\theta) + r\cos\theta \\
- s\sin(\phi+\theta) + r\sin\theta \\
1
\end{bmatrix}
$$

Therefore, the speed of each point is
$$
\dot x = s\sin(\phi+\theta)(\dot\phi+\dot\theta) + \dot r\cos\theta - r\dot\theta\sin\theta\\
\dot y = - s\cos(\phi+\theta)(\dot\phi+\dot\theta) + \dot r\sin\theta +r\dot\theta\cos\theta
$$

The kinetic energy is the integral of the kinetic energy of every point on the rod
$$
K(\theta,\phi,r,\dot\theta,\dot\phi,\dot r) \\
= \int_{-l}^l \frac{1}{2}\frac{m}{2l}\big[(s\sin(\phi+\theta)(\dot\phi+\dot\theta) + \dot r\cos\theta - r\dot\theta\sin\theta)^2+ (- s\cos(\phi+\theta)(\dot\phi+\dot\theta) + \dot r\sin\theta +r\dot\theta\cos\theta)^2\big]ds \\
= \int_{-l}^l \frac{1}{2}\frac{m}{2l}\big[\\
s^2\sin^2(\phi+\theta)(\dot\phi+\dot\theta)^2 + \dot r^2\cos^2\theta + r^2\dot\theta^2\sin^2\theta + 2s\dot r(\dot\phi+\dot\theta)\sin(\phi+\theta)\cos\theta - 2sr\dot\theta(\dot\phi+\dot\theta)\sin(\phi+\theta)\sin\theta - 2r\dot r\dot\theta\sin\theta\cos\theta\\+ 
s^2\cos^2(\phi+\theta)(\dot\phi+\dot\theta)^2 + \dot r^2\sin^2\theta + r^2\dot\theta^2\cos^2\theta - 2s\dot r(\dot\phi+\dot\theta)\cos(\phi+\theta)\sin\theta - 2sr\dot\theta(\dot\phi+\dot\theta)\cos(\phi+\theta)\cos\theta + 2r\dot r\dot\theta\sin\theta\cos\theta
\\\big]ds\\=
\int_{-l}^l \frac{1}{2}\frac{m}{2l}\big[s^2(\dot\phi+\dot\theta)^2 + \dot r^2 + r^2\dot\theta^2 + 2s\dot r(\dot\theta+\dot\phi)\sin\phi - 2sr\dot\theta(\dot\theta+\dot\phi)\cos\phi \big]ds\\ =
\frac{ml^{2}}{6} (\dot{\phi}^{2} + 2\dot{\phi}\dot{\theta} + \dot{\theta}^{2}) + \frac{m}{2} \left(r^{2} \dot{\theta}^{2} + \dot{r}^{2}\right) \\=
\frac{m}{6} \begin{bmatrix}\dot r & \dot\phi & \dot\theta\end{bmatrix}
\begin{bmatrix}
3  &     &    \\
   & l^2 & l^2 \\
   & l^2 & l^2 + 3r^2
\end{bmatrix}
\begin{bmatrix}\dot r \\ \dot\phi \\ \dot\theta\end{bmatrix}
$$

which has a concise form.

To compute the potential energy, we need to know the distance between each point on the rod and the origin. We can use the Law of cosines to get this distance

<img src="img\cos.png" alt="cos" style="zoom: 33%;" />

which is $\sqrt{r^2+s^2-2rs\cos\phi}$.

The gravitational potential of a mass is $U = -G\frac{Mm}{r}$, so the total potential is the integral of the potential of every point in the rod
$$
U(\theta,\phi,r) = \int_{-l}^l -G\frac{Mm/2l}{\sqrt{r^2+s^2-2rs\cos\phi}}ds
$$

Fortunately, this integral has a closed form. Because

$$
\int \frac{1}{\sqrt{s^2+as+b}}ds = \ln (2\sqrt{s^2+as+b}+2s+a)
$$

in our integral, $a = -2rcos\phi, b = r^2$, so
$$
\int -G\frac{Mm/2l}{\sqrt{r^2+s^2-2rs\cos\phi}}ds = -\frac{GMm}{2l}\ln(2\sqrt{r^2+s^2-2rs\cos\phi}+2s-2r\cos\phi)\\
U(\theta,\phi,r) = -\frac{GMm}{2l}\ln(\frac{\sqrt{r^2+l^2-2rl\cos\phi}+l-r\cos\phi}{\sqrt{r^2+l^2+2rl\cos\phi}-l-r\cos\phi})
$$

which is actually independent with $\theta$.

So the Lagrangian $L = K - U$ is
$$
L = \frac{m}{6} \begin{bmatrix}\dot r & \dot\phi & \dot\theta\end{bmatrix}
\begin{bmatrix}
3  &     &    \\
   & l^2 & l^2 \\
   & l^2 & l^2 + 3r^2
\end{bmatrix}
\begin{bmatrix}\dot r \\ \dot\phi \\ \dot\theta\end{bmatrix} - \int_{-l}^l -G\frac{Mm/2l}{\sqrt{r^2+s^2-2rs\cos\phi}}ds
$$

The general momentum and the derivative of $L$ w.r.t. general coordinates $q$ is 

$$
\frac{\part L }{\part\dot q} = \begin{bmatrix}p_r \\ p_\phi \\ p_\theta\end{bmatrix} =\begin{bmatrix}\frac{\part L }{\part\dot r} \\ \frac{\part L }{\part\dot\phi} \\ \frac{\part L }{\part\dot\theta}\end{bmatrix} = \frac{m}{3}
\begin{bmatrix}
3  &     &    \\
   & l^2 & l^2 \\
   & l^2 & l^2 + 3r^2
\end{bmatrix}
\begin{bmatrix}\dot r \\ \dot\phi \\ \dot\theta\end{bmatrix}\\
\frac{\part L }{\part q} = \begin{bmatrix}\frac{\part L }{\part r} \\ \frac{\part L }{\part \phi} \\ \frac{\part L }{\part \theta}\end{bmatrix} = \begin{bmatrix}
mr\dot\theta^2 - \frac{\part U }{\part r} \\  - \frac{\part U }{\part \phi} \\ -\frac{\part U }{\part \theta}
\end{bmatrix} = 
\begin{bmatrix}
mr\dot\theta^2 - \frac{\part U }{\part r} \\  - \frac{\part U }{\part \phi} \\ 0
\end{bmatrix}
$$

So according to Euler- Lagrange Equation, we get an ODE system, notice $\frac{\part U}{\part \theta} = 0$ because $U$ is independent with $\theta$
$$
\frac{d}{dt}\frac{\part L}{\part \dot q} = \frac{\part L}{\part q}\\\Rightarrow
\frac{d}{dt}\frac{m}{3}
\begin{bmatrix}
3  &     &    \\
   & l^2 & l^2 \\
   & l^2 & l^2 + 3r^2
\end{bmatrix}
\begin{bmatrix}\dot r \\ \dot\phi \\ \dot\theta\end{bmatrix}
 = \begin{bmatrix}
mr\dot\theta^2 - \frac{\part U }{\part r} \\  - \frac{\part U }{\part \phi} \\ -\frac{\part U }{\part \theta}
\end{bmatrix} \\\Rightarrow
\frac{d}{dt} \begin{bmatrix}r \\ \phi \\ \theta \\ \dot r \\ \dot\phi \\ \dot\theta\end{bmatrix} = 
\begin{bmatrix}\dot r \\ \dot\phi \\ \dot\theta \\
r\dot\theta^2 - \frac{1}{m}\frac{\part U }{\part r}\\
-\frac{3}{ml^2}\frac{\part U }{\part \phi} + \frac{1}{mr^2}(\frac{\part U }{\part \theta} - \frac{\part U }{\part \phi}) + \frac{2}{r}\dot r\dot\theta\\
\frac{1}{mr^2}(\frac{\part U }{\part \phi} - \frac{\part U }{\part \theta}) -\frac{2}{r}\dot r\dot\theta
\end{bmatrix} \\\Rightarrow
\frac{d}{dt} \begin{bmatrix}r \\ \phi \\ \theta \\ \dot r \\ \dot\phi \\ \dot\theta\end{bmatrix} = 
\begin{bmatrix}\dot r \\ \dot\phi \\ \dot\theta \\
r\dot\theta^2 - \frac{1}{m}\frac{\part U }{\part r}\\
-(\frac{3}{ml^2}+\frac{1}{mr^2})\frac{\part U }{\part \phi} + \frac{2}{r}\dot r\dot\theta\\
\frac{1}{mr^2}\frac{\part U }{\part \phi} -\frac{2}{r}\dot r\dot\theta
\end{bmatrix}
$$

### Simulation

We use Forward Euler to simulate the dynamics / solve the ODE

for an ODE system
$$
\dot {\bold y} = \bold f(\bold{y}, t)
$$
the Forward Euler is using
$$
\bold y_{n+1} = \bold y_n + \Delta t \cdot \bold f(\bold y_n, t_n)
$$
to update $\bold y$.

In our ODE, $\bold f$ do have a closed form. However, the partial derivative of the general potential $U$ is very hard to compute and is very complex ($U(\theta,\phi,r) = -\frac{GMm}{2l}\ln(\frac{\sqrt{r^2+l^2-2rl\cos\phi}+l-r\cos\phi}{\sqrt{r^2+l^2+2rl\cos\phi}-l-r\cos\phi})$). Therefore we use the autodiff system in taichi to compute the partial derivative of $U$.

### Result

The simulation is very unstable because we use the forward Euler. But finally we get some reasonable results.

## Update

The numerical instability is solved now. However, the analysis to the instability is interesting. **Please check the last submission for the unstable simulation result.**

###  Instability Behavior

In previous implementation, we found that when the rod is pointing towards the origin (that is when $\phi = 0$), the angular velocity of the rod would probably behave strangely, including suddenly vanishing, changing direction, or exploding. These strange behavior can be clearly seen from the previous video.

### Analysis of instability

The reason is that, when $\phi\to 0$, the general potential energy 
$$
U(\theta,\phi,r) = -\frac{GMm}{2l}\ln(\frac{\sqrt{r^2+l^2-2rl\cos\phi}+l-r\cos\phi}{\sqrt{r^2+l^2+2rl\cos\phi}-l-r\cos\phi})
$$
would behavior numerically unstable. Specifically, when $\phi= 0$, we have
$$
\frac{\sqrt{r^2+l^2-2rl\cos\phi}+l-r\cos\phi}{\sqrt{r^2+l^2+2rl\cos\phi}-l-r\cos\phi} = \frac{r-l+l-r}{r+l-l-r} = \frac{0}{0}
$$
Mathematically, we can use L'Hôpital's rule to compute the limit, which is $\frac{r+l}{r-l}$. However, in numerical computing this can leads to unstable behavior, especially when we take the numerical derivative of it. 

In the previous implementation, we use autodiff, which is a kind of numerical derivative. So near 0/0, the derivative would probability be a very big value and thus lead to immediately changing of the angular velocity.

### Solution

Instead of using autodiff, we compute the partial derivative manually. Though the derivative is pretty complex and took us 1 hour to derive, the result is simple and eliminate the instability. Here are derivations:

Because
$$
p(s) = \frac{\part}{\part r}\ln(2\sqrt{r^2+s^2-2rs\cos\phi}+2s-2r\cos\phi)
\\=\frac{{\frac{{r - s\cos\phi}}{{\sqrt{r^2 - 2rs\cos\phi + s^2}}} - \cos\phi}}{{\sqrt{r^2 - 2rs\cos\phi + s^2} - r\cos\phi + s}} \\= 
\frac{(\sqrt{r^2 - 2rs\cos\phi + s^2} -s + r\cos\phi)(\frac{r - s\cos\phi}{\sqrt{r^2 - 2rs\cos\phi + s^2}} - \cos\phi))}{(r^2 - 2rs\cos\phi + s^2) - (s^2 - 2rs\cos\phi +r^2\cos^2\phi)} \\=
\frac{(\sqrt{r^2 - 2rs\cos\phi + s^2} + r\cos\phi - s)(\frac{r - s\cos\phi}{\sqrt{r^2 - 2rs\cos\phi + s^2}} - \cos\phi))}{r^2\sin^2\phi}\\=
\frac{r-s\cos\phi-\cos\phi\sqrt{r^2 - 2rs\cos\phi + s^2}+
\frac{r^2\cos\phi-rs\cos^2\phi-rs+s^2\cos\phi}{\sqrt{r^2 - 2rs\cos\phi + s^2}}
-r\cos^2\phi+s\cos\phi
}{r^2\sin^2\phi}\\=
\frac{r\sin^2\phi-
\frac{(r^2+s^2)\cos\phi - 2rs\cos^2\phi}{\sqrt{r^2 - 2rs\cos\phi + s^2}}+
\frac{(r^2+s^2)\cos\phi-rs\cos^2\phi-rs}{\sqrt{r^2 - 2rs\cos\phi + s^2}}
}{r^2\sin^2\phi}\\=
\frac{r\sin^2\phi-
\frac{rs\sin^2\phi}{\sqrt{r^2 - 2rs\cos\phi + s^2}}
}{r^2\sin^2\phi}\\=
\frac{1-
\frac{s}{\sqrt{r^2 - 2rs\cos\phi + s^2}}
}{r}
$$
we have
$$
\frac{\part U}{\part r} = -\frac{GMm}{2l}(p(l)-p(-l)) \\=
-\frac{GMm}{2l}(
\frac{
(1-\frac{l}{\sqrt{r^2 - 2rl\cos\phi + l^2}})-
(1-\frac{-l}{\sqrt{r^2 + 2rl\cos\phi + l^2}})
}{r})
\\=
-\frac{GMm}{2l}(\frac{-l(\frac{1}{\sqrt{r^2 - 2rl\cos\phi + l^2}}+\frac{1}{\sqrt{r^2 + 2rl\cos\phi + l^2}})}{r}) \\=
\frac{GMm}{2r}(\frac{1}{\sqrt{r^2 - 2rl\cos\phi + l^2}}+\frac{1}{\sqrt{r^2 + 2rl\cos\phi + l^2}})
$$
Because
$$
q(s) = \frac{\part}{\part \phi}\ln(2\sqrt{r^2+s^2-2rs\cos\phi}+2s-2r\cos\phi)
\\=\frac{{\frac{{rs\sin\phi}}{{\sqrt{r^2 - 2rs\cos\phi + s^2}}} + r\sin\phi}}{{\sqrt{r^2 - 2rs\cos\phi + s^2} - r\cos\phi + s}} \\= 
\frac{(\sqrt{r^2 - 2rs\cos\phi + s^2} -s + r\cos\phi)(\frac{rs\sin\phi}{\sqrt{r^2 - 2rs\cos\phi + s^2}} + r\sin\phi))}{r^2\sin^2\phi}\\=
\frac{rs\sin\phi+r\sin\phi\sqrt{r^2 - 2rs\cos\phi + s^2}+
\frac{r^2s\sin\phi\cos\phi-rs^2\sin\phi}{\sqrt{r^2 - 2rs\cos\phi + s^2}}
-rs\sin\phi+r^2\sin\phi\cos\phi
}{r^2\sin^2\phi}\\=
\frac{r^2\sin\phi\cos\phi+
\frac{r^3\sin\phi-2r^2s\sin\phi\cos\phi+rs^2\sin\phi}{\sqrt{r^2 - 2rs\cos\phi + s^2}}+
\frac{r^2s\sin\phi\cos\phi-rs^2\sin\phi}{\sqrt{r^2 - 2rs\cos\phi + s^2}}
}{r^2\sin^2\phi}\\=
\frac{r\sin^2\phi+
\frac{r^3\sin\phi-r^2s\sin\phi\cos\phi}{\sqrt{r^2 - 2rs\cos\phi + s^2}}
}{r^2\sin^2\phi}\\=
\frac{\sin\phi+
\frac{r^2-rs\cos\phi}{\sqrt{r^2 - 2rs\cos\phi + s^2}}
}{r\sin\phi}
$$
we have
$$
\frac{\part U}{\part \phi} = -\frac{GMm}{2l}(q(l)-q(-l))
\\=
-\frac{GMm}{2lr\sin\phi}(
(\frac{r^2-rl\cos\phi}{\sqrt{r^2 - 2rl\cos\phi + l^2}})
-(\frac{r^2+rl\cos\phi}{\sqrt{r^2 + 2rl\cos\phi + l^2}})
)\\=
-\frac{GMm}{2l\sin\phi}(\frac{r-l\cos\phi}{\sqrt{r^2 - 2rl\cos\phi + l^2}} - \frac{r+l\cos\phi}{\sqrt{r^2 + 2rl\cos\phi + l^2}})
$$
plug into the ODE system, we get an analytically updating rule.

Notice that there is also a 0/0 in $\frac{\part U}{\part \phi}$: when $\phi = 0$, 
$$
\frac{1}{\sin\phi}(\frac{r-l\cos\phi}{\sqrt{r^2 - 2rl\cos\phi + l^2}} - \frac{r+l\cos\phi}{\sqrt{r^2 + 2rl\cos\phi + l^2}}) = \frac{1}{0}(\frac{r-l}{r-l} - \frac{r+l}{r+l}) = \frac{0}{0}
$$
But it didn't cause instability. Our conjecture is that, because we don't take numerical derivative, the 0/0 case has the measure 0, which means that we can get an valid value "almost everywhere".

### Updated Simulation

We implement the explicit Runge–Kutta methods of order 4 (RK4) additionally to the forward Euler method.

for an ODE system
$$
\dot {\bold y} = \bold f(\bold{y}, t)
$$
the RK4 is using
$$
\bold{y}_{n+1} = \bold{y}_n + \frac{\Delta t}{6} (\bold k_1 + 2 \bold k_2 + 2 \bold k_3 + \bold k_4)\\
where\\
\begin{cases}
\bold k_1 &= \bold f(\bold y_n, t_n), \\
\bold k_2 &= \bold f(\bold y_n + \frac{\Delta t}{2} \cdot \bold k_1, t_n + \frac{\Delta t}{2}), \\
\bold k_3 &= \bold f(\bold y_n + \frac{\Delta t}{2} \cdot \bold k_2, t_n + \frac{\Delta t}{2}), \\
\bold k_4 &= \bold f(\bold y_n + \Delta t \cdot \bold k_3, t_n + \Delta t).
\end{cases}
$$
to update $\bold y$.

### Updated Result

The simulation is now very stable unless when the rod touch the origin. That case is theoretically impossible.

We give the rod an initial velocity, and the rod can move approximately elliptically / parabolically depending on the initial value.

We also found that the system is chaotic. With near identical initial conditions, different rod diverge over time displaying the chaotic nature of the system. In the video, two rods only have 0.001 difference in initial angular velocity.

### Future Work

Apply implicit Euler to make it more numerical stable and a barrier function to avoid hitting the origin.
