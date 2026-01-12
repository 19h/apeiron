# Multiscale Entropy Analysis: A New Method to Detect Determinism in a Time Series

**A. Sarkar and P. Barat**

Variable Energy Cyclotron Centre  
1/AF Bidhan Nagar, Kolkata 700064, India

**PACS numbers:** 05.45.Tp, 89.75.-k, 82.40.Bj

---

## Abstract

In this letter we show that the Multiscale Entropy (MSE) analysis can detect the determinism in a time series.

---

## Introduction

The output variables (time series) from physical systems often exhibit complex fluctuations containing information about the underlying dynamics. An important problem in the study of a time series is determining whether the time series arises from a stochastic process or has a deterministic component that is generated from chaotic dynamics having finite number of degrees of freedom. Whether a time series has a deterministic component or not in turn dictates what approaches are appropriate for investigating the time series and its generating process. In this sense detecting the determinism in a time series is very important.

Several methods of nonlinear dynamical analysis have previously been developed to detect determinism in time series [1-3]. These methods are all based on the assumption that a trajectory in the state space reconstructed from a deterministic time series behaves similarly to nearby trajectories as time evolves. Hence, a large number of data points are required to have sufficient information of the nearby trajectories to compare their future behaviors. In addition, the application of these methods can lead to spurious results for nonstationary time series.

Recently Costa et al. [4] introduced a new method, Multiscale Entropy (MSE) analysis for measuring the complexity of finite length time series. In this paper we show that the MSE method can be used to detect the determinism in a time series. The MSE method measures complexity taking into account the multiple time scales. This computational tool can be quite effectively used to quantify the complexity of a natural time series. The MSE method uses Sample Entropy (SampEn) [5] to quantify the regularity of finite length time series. SampEn is largely independent of the time series length when the total number of data points is larger than approximately 750 [5].

Recently MSE has been successfully applied to quantify the complexity of many Physiologic and Biological signals [6, 7].

---

## Method

The MSE method is based on the evaluation of SampEn on the multiple scales. The prescription of the MSE analysis is: given a one-dimensional discrete time series, {x₁, ..., xᵢ, ..., xₙ}, construct the consecutive coarse-grained time series, {y⁽τ⁾}, determined by the scale factor, τ, according to the equation:

$$y_j^{(\tau)} = \frac{1}{\tau} \sum_{i=(j-1)\tau+1}^{j\tau} x_i$$
<p align="center">(1)</p>

where τ represents the scale factor and 1 ≤ j ≤ N/τ. The length of each coarse-grained time series is N/τ. For scale one, the coarse-grained time series is simply the original time series.

Next we calculate the SampEn for each scale using the following method. Let X = {x₁, ..., xᵢ, ..., xₙ} be a time series of length N.

Let **u**ₘ(i) = {xᵢ, xᵢ₊₁, ..., xᵢ₊ₘ₋₁}, 1 ≤ i ≤ N − m, be vectors of length m.

Let nᵢᵐ(r) represent the number of vectors **u**ₘ(j) within distance r of **u**ₘ(i), where j ranges from 1 to (N−m) and j ≠ i to exclude the self matches.

$$C_i^m(r) = n_i^m(r) / (N - m - 1)$$

is the probability that any **u**ₘ(j) is within r of **u**ₘ(i).

We then define:

$$U^m(r) = \frac{1}{N-m} \sum_{i=1}^{N-m} \ln C_i^m(r)$$
<p align="center">(2)</p>

The parameter Sample Entropy (SampEn) [5] is defined as:

$$\text{SampEn}(m, r) = \lim_{N \to \infty} \left[ -\ln \frac{U^{m+1}(r)}{U^m(r)} \right]$$
<p align="center">(3)</p>

For finite length N the SampEn is estimated by the statistics:

$$\text{SampEn}(m, r, N) = -\ln \frac{U^{m+1}(r)}{U^m(r)}$$
<p align="center">(4)</p>

Advantage of SampEn is that it is less dependent on time series length and is relatively consistent over broad range of possible r, m and N values. We have calculated SampEn for all the studied data sets with the parameters **m = 2** and **r = 0.15 × SD** (SD is the standard deviation of the original time series).

---

## Results

Costa et al. had tested the MSE method on simulated white and 1/f noises [4]. They have shown that for the scale one, the value of entropy is higher for the white noise time series in comparison to the 1/f noise. This may apparently lead to the conclusion that the inherent complexity is more in the white noise in comparison to the 1/f noise. However, the application of the MSE method shows that the value of the entropy for the 1/f noise remains almost invariant for all the scales while the value of entropy for the white noise time series monotonically decreases and for scales greater than 5, it becomes smaller than the corresponding values for the 1/f noise. This result explains the fact that the 1/f noise contains complex structures across multiple scales in contrast to the white noise.

With a view to understand the complexity of deterministic chaotic data we have applied the MSE method to the following synthetic chaotic data sets.

### 1. Logistic Map

$$x_{n+1} = a x_n (1 - x_n)$$

Parameters: a = 3.9

### 2. Henon Map

$$x_{n+1} = 1 - \alpha x_n^2 + y_n$$

$$y_{n+1} = \beta x_n$$

Parameters: α = 1.4, β = 0.3

### 3. Ikeda Map

$$x_{n+1} = a + c x_n \exp\left( ib - \frac{i}{1 + |x_n|^2} \right)$$

Parameters: a = 0.4, b = 6.0, c = 0.9

### 4. Quadratic Map

$$x_{n+1} = p - x_n^2$$

Parameters: p = 1.7904

### 5. Rössler Equation

$$\frac{dx}{dt} = -y - z$$

$$\frac{dy}{dt} = x + ay$$

$$\frac{dz}{dt} = b + z(x - c)$$

Parameters: a = 0.2, b = 0.2, c = 5.7

### 6. Lorenz Equation

$$\frac{dx}{dt} = -ax + ay$$

$$\frac{dy}{dt} = bx - y - xz$$

$$\frac{dz}{dt} = -cz + xy$$

Parameters: a = 10, b = 28, c = 8/3

---

## Discussion

The result of the MSE analysis on the chaotic data sets together with the white noise, fractional Brownian noise (with Hurst exponent 0.7) [8] and the 1/f noise is shown in Fig. 1. It is seen that the entropy measure for the deterministic chaotic time series increases on small scales and then gradually decreases indicating the reduction of complexity on the larger scales. This trend of the variation of the SampEn with scale is entirely different from the white noise, fractional Brownian noise and the 1/f noise [4]. Moreover, the variation of the SampEn for all chaotic data sets showed a similar behavior. This establishes the fact that the MSE analysis can be used to detect the determinism in a time series.

---

## Conclusion

In conclusion we have showed that the Sample Entropy is an important statistic for detecting determinism in a time series.

---

## References

1. KAPLAN D. T. and GLASS L., Phys. Rev. Lett. **68** (1992) 427.
2. WAYLAND R., BROONLEY D., PICKETT D. and PASSARNATE A., Phys. Rev. Lett., **70** (1993) 580.
3. SALVINO L. W. and CAWLEY R., Phys. Rev. Lett., **73** (1994) 1091.
4. COSTA M., GOLDBERGER A. L. and PENG C. –K., Phys. Rev. Lett., **89** (2002) 068102.
5. RICHMAN J. S. and MOORMAN J. R., Am. J. Physiol., **278** (2000) H2039.
6. COSTA M., PENG C. –K., GOLDBERGER A. L., and HAUSDORFF J. M. Physica A, **330** (2003) 53.
7. COSTA M., GOLDBERGER A. L. and PENG C. –K., Phys. Rev. E, **71** (2005) 021906.
8. MANDELBROT B. B., in *The Fractal Geometry of Nature*, (San Francisco, Ca: Freeman) 1982.

---

## Figure

**Fig. 1:** MSE analysis of the various simulated chaotic data and white noise, fractional Brownian noise, 1/f noise each with 20000 data points.

![MSE Analysis Results](figure1_description.md)

*The figure shows Sample Entropy (y-axis, range 0.0–2.6) vs Scale Factor (x-axis, range 0–20) for:*
- *Logistic Map*
- *Henon Map*
- *Ikeda Map*
- *Quadratic Map*
- *Rössler Equation*
- *Lorenz Equation*
- *White Noise*
- *Fractional Brownian Noise*
- *1/f Noise*

*Key observation: Deterministic chaotic time series show an initial increase in SampEn at small scales followed by gradual decrease, distinguishing them from stochastic processes.*
