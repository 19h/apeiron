# Multivariate Refined Composite Multiscale Entropy Analysis

**Anne Humeau-Heurtier**

*Université d'Angers, LARIS – Laboratoire Angevin de Recherche en Ingénierie des Systèmes, 62 avenue Notre-Dame du Lac, 49000 Angers, France*

**Email:** anne.humeau@univ-angers.fr

**DOI:** http://dx.doi.org/10.1016/j.physleta.2016.02.029

---

## Article Info

**Article history:**
- Received: 18 May 2015
- Received in revised form: 12 February 2016
- Accepted: 18 February 2016
- Communicated by: A.P. Fordy

**Keywords:** Complexity, Nonlinear dynamics, Entropy, Fractal, Multivariate embedding, Multiscale entropy

---

## Abstract

Multiscale entropy (MSE) has become a prevailing method to quantify signals complexity. MSE relies on sample entropy. However, MSE may yield imprecise complexity estimation at large scales, because sample entropy does not give precise estimation of entropy when short signals are processed. A refined composite multiscale entropy (RCMSE) has therefore recently been proposed. Nevertheless, RCMSE is for univariate signals only. The simultaneous analysis of multi-channel (multivariate) data often over-performs studies based on univariate signals. We therefore introduce an extension of RCMSE to multivariate data. Applications of multivariate RCMSE to simulated processes reveal its better performances over the standard multivariate MSE.

---

## Introduction

Several entropy measures have been proposed to assess the regularity of times series. Among them, we can cite the sample entropy [1]. Sample entropy is equal to the negative of the natural logarithm of the conditional probability that sequences close to each other for *m* consecutive data points will also be close to each other when one more point is added to each sequence [1]. However, sample entropy operates on a single scale. Real world data, as physiological data, exhibit high degree of structural richness. Studies based on a single scale are therefore not adapted for real world signals. Analyses on multiple time scales have become necessary.

In the 2000s, Costa et al. proposed the multiscale entropy (MSE) to quantify complexity over multiple scales [2,3]. The MSE algorithm is composed of two steps [2,3]:

1. A coarse-graining procedure to derive a set of time series representing the system dynamics on different time scales. The coarse-graining procedure for scale τ is obtained by averaging the samples of the time series inside consecutive but non-overlapping windows of length τ.
2. The computation of the sample entropy for each coarse-grained time series.

MSE has become a prevailing method to quantify the complexity of signals. It has been shown through several studies that MSE is able to underline the general loss of complexity behavior when a living system changes from a healthy state to a pathological state [2,3].

Nevertheless, the coarse-graining procedure used in the MSE algorithm shortens the length of the data that are processed: for an original time series of *N* samples, the length of the coarse-grained time series at a scale factor τ is *N*/τ. It has been reported that for an embedding dimension *m* = 2, the sample entropy is significantly independent of the time series length when the number of data points is larger than 750 [4]. For shorter time series, the variance of the entropy estimator may grow very fast with the reduction of the number of data points. Therefore, at large scales, the coarse-grained time series may not be adequately long to obtain an accurate value for the sample entropy. Moreover, for some cases, the sample entropy value may not be defined because no template vectors are matched to one another. These two drawbacks (inaccurate or undefined sample entropy values) lead to problems of accuracy and validity of MSE at large scales.

In order to overcome the accuracy concern of MSE, Wu et al. proposed the composite MSE (CMSE) [5]. In the CMSE algorithm, all coarse-grained time series for a scale factor τ are processed to compute their sample entropy (each of the τ coarse-grained time series corresponding to different starting points of the coarse-graining process is used in the CMSE algorithm whereas, in the conventional MSE algorithm, for each scale, only the first coarse-grained time series is taken into account). The CMSE value for a given scale is therefore defined as the mean of several entropy values [5]. Therefore, CMSE estimates entropy more accurately than MSE. Unfortunately, CMSE increases the probability of inducing undefined entropy. This is why a refined CMSE (RCMSE) algorithm has been proposed in 2014, see below [6].

However, MSE, CMSE, and RCMSE are able to process univariate data only. For multivariate time series, the three algorithms treat individual time series separately. This may be satisfactory only if the individual signals are statistically independent or at least uncorrelated, which is often not the case when real world signals from a given system are registered simultaneously. To overcome this shortcoming, an extension of the MSE algorithm to multivariate data has been proposed in 2011: the multivariate MSE (MMSE) [8,9]. MMSE is able to operate on any number of data channels and provides a robust relative complexity measure for multivariate data [8,9]. MMSE has been used in studies from different fields [10–13]. However, the same concerns as MSE are found in MMSE. This is why in this work we propose an extension of the RCMSE algorithm to a more general case. To this end, we introduce the multivariate RCMSE (MRCMSE), and evaluate its performances on synthetic multivariate processes.

---

## 1. Multivariate Refined Composite Multiscale Entropy

### 1.1. Refined Composite Multiscale Entropy

RCMSE aims at improving the CMSE algorithm because, as mentioned previously, CMSE estimates entropy more accurately than MSE but increases the probability of inducing undefined entropy [6,7].

For a discrete time series **x** = {*x*ᵢ}ᴺᵢ₌₁, the RCMSE algorithm is based on the following three steps [6]:

**Step 1.** The *k*th coarse-grained time series for a scale factor τ is defined as **y**^(τ)_k = {*y*_{k,j}}^{N/τ}_{j=1} where [5]:

$$y_{k,j}^{(\tau)} = \frac{1}{\tau} \sum_{i=(j-1)\tau+k}^{j\tau+k-1} x_i, \quad 1 \leq j \leq \frac{N}{\tau}, \quad 1 \leq k \leq \tau$$

**Step 2.** For each scale factor τ, and for all τ coarse-grained time series, the number of matched vector pairs *n*^{m+1}_{k,τ} and *n*^m_{k,τ} is computed, where *n*^m_{k,τ} represents the total number of *m*-dimensional matched vector pairs and is computed from the *k*th coarse-grained time series at a scale factor τ.

**Step 3.** RCMSE is then defined as [6]:

$$\text{RCMSE}(x, \tau, m, r) = -\ln\left(\frac{\sum_{k=1}^{\tau} n_{k,\tau}^{m+1}}{\sum_{k=1}^{\tau} n_{k,\tau}^{m}}\right)$$

Using the same notation, CMSE is defined as [6]:

$$\text{CMSE}(x, \tau, m, r) = \frac{1}{\tau} \sum_{k=1}^{\tau} \left( -\ln \frac{n_{k,\tau}^{m+1}}{n_{k,\tau}^{m}} \right)$$

The CMSE value is therefore undefined when one of the values *n*^{m+1}_{k,τ} or *n*^m_{k,τ} is zero. By opposition, RCMSE value is undefined only when all *n*^{m+1}_{k,τ} or *n*^m_{k,τ} are zeros. It has been reported that RCMSE outperforms CMSE in validity, accuracy of entropy estimation, independence of data length, and computational efficiency [6]. RCMSE has been used in recent studies [14].

### 1.2. Multivariate Multiscale Entropy

MMSE is an extension of the MSE algorithm to multivariate data. MMSE relies on the same steps as MSE [8,9]:

1. A coarse-graining procedure
2. A sample entropy computation for each coarse-grained time series

However, due to the multivariate nature of the data processed by MMSE, these two steps are adapted to multivariate signals. Thus, for the coarse-graining procedure, temporal scales are defined by averaging a *p*-variate time series {*x*_{l,i}}ᴺᵢ₌₁ (*l* = 1,..., *p* is the channel index and *N* is the number of samples in every channel) over non-overlapping time segments of increasing length. Thus, for a scale factor τ, a coarse-grained multivariate time series is computed as:

$$y_{l,j}^{(\tau)} = \frac{1}{\tau} \sum_{i=(j-1)\tau+1}^{j\tau} x_{l,i}$$

where 1 ≤ *j* ≤ *N*/τ, and the channel index *l* goes from 1 to *p*.

For the entropy computation, the multivariate sample entropy (MSampEn) is used for each coarse-grained multivariate. The MSampEn algorithm is an extension of the univariate sample entropy [1]. For a tolerance level *r*, MSampEn is calculated as the negative of the natural logarithm of the conditional probability that two composite delay vectors close to each other in a *m* dimensional space will also be close to each other when the dimensionality is increased by one. The detailed MSampEn algorithm can be found in [8,9].

### 1.3. Multivariate Refined Composite Multiscale Entropy

Based on RCMSE and MSampEn, we define the MRCMSE algorithm as follows:

**Step 1.** For a *p*-variate time series {*x*_{l,i}}ᴺᵢ₌₁, *l* = 1,..., *p*, where *p* denotes the number of variates (channels) and *N* is the number of samples in each variate, and for a scale factor τ, determine the coarse-grained multivariate time series {*y*^(τ)_{l,k,j}}^{N/τ}_{j=1} as:

$$y_{l,k,j}^{(\tau)} = \frac{1}{\tau} \sum_{i=(j-1)\tau+k}^{j\tau+k-1} x_{l,i}$$

where 1 ≤ *j* ≤ *N*/τ, 1 ≤ *k* ≤ τ, *l* = 1,..., *p*

**Step 2.** For each coarse-grained multivariate compute *B*^m(*r*) and *B*^{m+1}(*r*) as defined in Table 1 [8,9]. For the coarse-grained multivariate {*y*^(τ)_{l,k,j}}^{N/τ}_{j=1}, *l* = 1,..., *p*, these two quantities are denoted as *B*^m_{k,τ}(*r*) and *B*^{m+1}_{k,τ}(*r*), respectively.

**Step 3.** Compute:

$$\text{MRCMSE}(\tau, M, r, \tau, N) = -\ln\left(\frac{\sum_{k=1}^{\tau} B_{k,\tau}^{m+1}(r)}{\sum_{k=1}^{\tau} B_{k,\tau}^{m}(r)}\right)$$

---

## Table 1: Computation of B^m(r) and B^{m+1}(r)

*(from [8,9])*

**Step 1.** Form (*N* − *n*) composite delay vectors **X**^m(*i*) ∈ ℝ^m, where *i* = 1, 2,..., *N* − *n* and *n* = max{**M**} × max{**τ**}. For a *p*-variate time series {*x*_{l,i}}ᴺᵢ₌₁, *l* = 1,..., *p*, the composite delay vector **X**^m(*i*) ∈ ℝ^m is defined as:

$$\mathbf{X}^m(i) = [x_{1,i}, x_{1,i+\tau_1}, \ldots, x_{1,i+(m_1-1)\tau_1}, x_{2,i}, x_{2,i+\tau_2}, \ldots, x_{2,i+(m_2-1)\tau_2}, \ldots, x_{p,i}, x_{p,i+\tau_p}, \ldots, x_{p,i+(m_p-1)\tau_p}]$$

where **M** = [*m*₁,...,*m*_p] ∈ ℝ^p is the embedding vector, **τ** = [τ₁,...,τ_p] the time lag vector and *m* = Σᵖₗ₌₁ *m*_l

**Step 2.** Define the distance between any two vectors **X**^m(*i*) and **X**^m(*j*) as the maximum norm, that is:

$$d[\mathbf{X}^m(i), \mathbf{X}^m(j)] = \max_{l=1,\ldots,m}\{|x(i + l - 1) - x(j + l - 1)|\}$$

**Step 3.** For a given composite delay vector **X**^m(*i*) and a threshold *r*, count the number of instances *P*_i for which *d*[**X**^m(*i*), **X**^m(*j*)] ≤ *r*, *j* ≠ *i*, then calculate the frequency of occurrence:

$$B_i^m(r) = \frac{1}{N-n-1} P_i$$

and define:

$$B^m(r) = \frac{1}{N-n} \sum_{i=1}^{N-n} B_i^m(r)$$

**Step 4.** Extend the dimensionality of the multivariate delay vector defined in step 1 from *m* to (*m* + 1). A total of *p* × (*N* − *n*) vectors **X**^{m+1}(*i*) in ℝ^{m+1} are obtained, where **X**^{m+1}(*i*) corresponds to any embedded vector upon increasing the embedding dimension from *m*_α to *m*_α + 1 for a specific variable α.

**Step 5.** For a given **X**^{m+1}(*i*), calculate the number of vectors *Q*_i, such that *d*[**X**^{m+1}(*i*), **X**^{m+1}(*j*)] ≤ *r*, where *j* ≠ *i*, then calculate the frequency of occurrence:

$$B_i^{m+1}(r) = \frac{1}{p \times (N-n) - 1} Q_i$$

and define:

$$B^{m+1}(r) = \frac{1}{p \times (N-n)} \sum_{i=1}^{p \times (N-n)} B_i^{m+1}(r)$$

---

## 2. Results and Discussion

In order to analyze the behavior of MRCMSE on multivariate data, we generated a trivariate time series, where originally all the data channels were realizations of mutually independent white noise [8]. We then gradually decreased the number of variates representing white noise (from 3 to 0) and simultaneously increased the number of data channels representing independent 1/*f* noise (from 0 to 3), as already proposed in [8,9]. The total number of variates was always three.

**Experimental Parameters:**
- 50 independent realizations per trivariate data type
- 10,000 samples per variate per realization
- Scale factors: 1 to 20
- Shortest coarse-grained time series length: 500 samples
- Embedding dimension *m*_k = 2 for each channel
- Threshold *r* = 0.15 × (standard deviation of the normalized time series) for each channel

**Complexity Interpretation:** A multivariate time series is considered more structurally complex than another if, for most of the scale factors τ, its multivariate entropy values are higher than those of the other time series. When the multivariate entropy values decrease with the scale factor τ, the time series that is processed only contains information at the smallest scales. It is thus not structurally complex. This is the same as what is observed for the univariate MSE where sample entropy values of random white noise (uncorrelated) decrease with the scale factor whereas for 1/*f* noise (long-range correlated), the sample entropy values are constant over multiple scales.

### Key Findings

**Figure 1 Results (10,000 samples):**
- MRCMSE and MMSE curves are close to each other
- Higher number of variates representing 1/*f* noise → higher multivariate entropy value for a given scale factor τ
- Behavior consistent between MRCMSE and MMSE

**Correlated Bivariate Analysis (Figure 2a, 10,000 samples):**
MRCMSE handles within- and cross-channel correlations. At large scales, complexity ordering (highest to lowest):
1. Correlated bivariate 1/*f* noise
2. Uncorrelated 1/*f* noise
3. Correlated white noise
4. Uncorrelated white noise

This confirms: for white noise and 1/*f* noise, the complexity of multivariate processes with within- and cross-channel correlations is higher than the complexity of uncorrelated multivariate processes.

### Short Time Series Analysis (1,000 samples)

**Parameters:**
- 1,000 samples per variate
- Shortest coarse-grained time series: 50 samples
- Same embedding dimension and threshold as above

**Figure 3 & 4 Results:**
- For time series with at least one white noise channel: MRCMSE and MMSE means are similar
- For pure 1/*f* noise channels: some MMSE entropy values are **undefined** at large scale factors; MRCMSE values are **all defined**
- MRCMSE standard deviation values are **lower** than MMSE for all trivariate signals

**Figure 5 Results (10,000 samples):**
Even with long time series, MRCMSE standard deviation values remain lower than MMSE.

### Data Length Dependency

Comparing Figure 1 (10,000 samples) and Figure 3 (1,000 samples) reveals different curve orderings. Explanations:

1. **Univariate case:** RCMSE is dependent on data length [6]

2. **Multivariate sample entropy sensitivity:** The naive approach shows increasing trend as data length decreases; the full multivariate approach shows decreasing multivariate sample entropy as data length decreases (Figure 6). Multivariate sample entropy is a relative estimate, not an absolute parameter.

3. **Non-stationarity effects:** Since 1/*f* noise time series are not stationary, as data points decrease, the discrepancy between numerically calculated sample entropy and mean value for simulated time series increases faster for 1/*f* noise than for white noise [3].

### Minimum Data Length Requirements

From testing:
- **≥3,000 samples:** Curve ordering matches long time series (10,000 samples) for scale factors 1–20
- **<3,000 samples:** Curve ordering changes
- **2,000 samples:** From scale factor τ = 6, trivariate time series with three 1/*f* noise channels show lower entropy than those with two 1/*f* + one white noise channel

**Recommendation:** For embedding dimension *m*_k = 2, MRCMSE estimates are consistent for data length *N* ≥ 300 (the highest scale should have at least 300 points). For shorter time series, MRCMSE may not be adequate.

**Key Advantage:** For both short and long multivariate time series:
- MRCMSE standard deviation values < MMSE standard deviation values
- No undefined entropy values with MRCMSE when MMSE produces them
- **MRCMSE shows better precision than MMSE**

### Threshold Parameter Selection

For the multivariate case (as suggested for MMSE [8,9]), MRCMSE uses the multivariate generalization of threshold parameter *r*: *r* as a percentage of the total variation of the covariance matrix. With each channel normalized to unit variance, differences in variance among multivariate signals do not affect multivariate sample entropy computation [8,9].

**Note on Coarse-Graining:**
- Nikulin and Brismar [16]: Coarse-graining acts as smoothing and decimation; if *r* is constant with scales, MSE changes depend on both regularity and variation of coarse-grained time series
- Costa et al. [17]: Irregularity degree is an entropy property not entirely captured by standard deviation or correlation measures individually or combined; post-normalization variance modifications from coarse-graining reflect temporal structure of the original signal

Additional factors potentially affecting MSE signatures include: sample time of time series, correlation time, and period of possible nonlinear oscillations [18].

---

## 3. Conclusion

MSE has been proposed in the 2000s to quantify the complexity of time series over multiple scales. However, the coarse-graining procedure used in the MSE algorithm shortens the length of the data processed; the larger the scale factor, the shorter the data processed. This may lead to inaccurate or undefined sample entropy values. CMSE and RCMSE have been proposed to overcome these drawbacks. Unfortunately, these two algorithms are dedicated to univariate data only. MMSE has been proposed recently to take into account multivariate data. However, MMSE presents the same disadvantages as MSE.

We therefore proposed an extension of RCMSE to multivariate data and we show that it leads to lower standard deviation values than MMSE. **In this sense, MRCMSE outperforms MMSE.**

---

## References

[1] J.S. Richman, J.R. Moorman, Physiological time-series analysis using approximate entropy and sample entropy, Am. J. Physiol., Heart Circ. Physiol. 278 (2000) H2039–H2049.

[2] M. Costa, A.L. Goldberger, C.K. Peng, Multiscale entropy analysis of complex physiologic time series, Phys. Rev. Lett. 89 (2002) 068102.

[3] M. Costa, A.L. Goldberger, C.K. Peng, Multiscale entropy analysis of biological signals, Phys. Rev. E 71 (2005) 021906.

[4] M. Costa, C.K. Peng, A.L. Goldberger, J.M. Hausdorff, Multiscale entropy analysis of human gait dynamics, Physica A 330 (2003) 53–60.

[5] S.D. Wu, C.W. Wu, S.G. Lin, C.C. Wang, K.Y. Lee, Time series analysis using composite multiscale entropy, Entropy 15 (2013) 1069–1084.

[6] S.D. Wu, C.W. Wu, S.G. Lin, K.Y. Lee, C.K. Peng, Analysis of complex time series using refined composite multiscale entropy, Phys. Lett. A 378 (2014) 1369–1374.

[7] A. Humeau-Heurtier, The multiscale entropy algorithm and its variants: a review, Entropy 17 (2015) 3110–3123.

[8] M.U. Ahmed, D.P. Mandic, Multivariate multiscale entropy: a tool for complexity analysis of multichannel data, Phys. Rev. E 84 (2011) 061918.

[9] M.U. Ahmed, D.P. Mandic, Multivariate multiscale entropy analysis, IEEE Signal Process. Lett. 19 (2012) 91–94.

[10] Z.K. Gao, M.S. Ding, H. Geng, N.D. Jin, Multivariate multiscale entropy analysis of horizontal oil-water two-phase flow, Physica A 417 (2015) 7–17.

[11] S. Begum, S. Barua, M.U. Ahmed, Physiological sensor signals classification for healthcare using sensor data fusion and case-based reasoning, Sensors 14 (2014) 11770–11785.

[12] P. Li, C. Liu, X. Wang, L. Li, L. Yang, Y. Chen, C. Liu, Testing pattern synchronization in coupled systems through different entropy-based measures, Med. Biol. Eng. Comput. 51 (2013) 581–591.

[13] Q. Wei, D.H. Liu, K.H. Wang, Q. Liu, M.F. Abbod, B.C. Jiang, K.P. Chen, C. Wu, J.S. Shieh, Multivariate multiscale entropy applied to center of pressure signals analysis: an effect of vibration stimulation of shoes, Entropy 14 (2012) 2157–2172.

[14] M.C. Chang, C.K. Peng, H.E. Stanley, Emergence of dynamical complexity related to human heart rate variability, Phys. Rev. E 90 (2014) 062806.

[15] M.U. Ahmed, N. Rehman, D. Looney, T.M. Rutkowski, D.P. Mandic, Dynamical complexity of human responses: a multivariate data-adaptive framework, Bull. Pol. Acad. Sci., Tech. Sci. 60 (2012) 433–445.

[16] V.V. Nikulin, T. Brismar, Comment on "Multiscale entropy analysis of complex physiologic time series", Phys. Rev. Lett. 92 (2004) 089803.

[17] M. Costa, A.L. Goldberger, C.K. Peng, Comment on "Multiscale entropy analysis of complex physiologic time series", Phys. Rev. Lett. 92 (2004) 089804.

[18] R.A. Thuraisingham, G.A. Gottwald, On multiscale entropy analysis for physiological data, Physica A 366 (2006) 323–332.

---

*Physics Letters A (2016)*

*© 2016 Published by Elsevier B.V.*
