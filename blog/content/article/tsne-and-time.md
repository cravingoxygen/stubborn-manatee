---
title: "T-SNE and Time: Part One"
date: 2018-05-30T12:58:35+02:00
draft: true

featuredImage: "/images/patreon.png"
categories: [code,math]
tags: [t-SNE, high dimensions, sequences, time series, visualization]
author: "Elre"
---

#### What is t-SNE?

[t-Distributed Stochastic Neighbor Embedding (t-SNE)](https://lvdmaaten.github.io/tsne/) is a visualization technique for high dimensional datasets. Well, technically it is a dimensionality reduction technique that also serves as an excellent visualization technique.

t-SNE is based on [Stochastic Neighbor Embedding](https://www.cs.toronto.edu/~fritz/absps/sne.pdf), which is another dimensionality reduction technique proposed by Hinton and Roweis in 2002. 

The general aim is to group similar points together and to keep dissimilar points further apart, even in the low dimensional representation. The technique defines similarity in terms of the conditional probability that two points will be neighbours, where the probability of them being neighbours depends on their distance apart. The structure of the high dimensional points can then be imitated in by choosing low dimensional points that have the same relationship with their neighbours as their high dimensional counterparts.

More precisely, for every pair of high dimensional points, $x\_i$ and $x\_j$ the similarity between them is defined as the conditional probability, $p\_{j|i}$, that $x\_j$ will be $x\_i$'s neighbour, assuming that $x\_i$'s neighbours are distributed according to a Gaussian centered at $x\_i$. The similarity between $x\_i$ and $x\_j$ can thus be expressed as
$$
	p\_{j|i} = \frac{exp(-||x\_i - x\_j||^2 / 2 \sigma\_i^2)}{ \sum\_{k \neq i} exp(-||x\_i - x\_k||^2 / 2 \sigma\_i^2)}
$$
where $\sigma\_i$ denotes the variance of the Gaussian centered at $x\_i$. In this way, the distance between two high dimensional points is converted into a conditional probability, where closer points are more likely to be neighbours and further points become exponentially less likely to be neighbours.
For now, pretend that we know $\sigma\_i$ for all the high dimensional points.

For every high dimensional point $x\_i$, we must find a low dimensional point $y_i$, where the pair-wise similarities between $x_i$ and all its neighbours is the same as the pair-wise similarities between $y\_i$ and all of its neighbours. To do that, we need an expression for measuring the similarity between two low dimensional points $y_i$ and $y_j$:
$$
	q\_{j|i} = \frac{exp(-||y\_i - y\_j||^2 )}{ \sum\_{k \neq i} exp(-||y\_i - y\_k||^2 )}
$$
where we set the variance for the Gaussian centered at $y\_i$ equal to $\frac{1}{\sqrt{2}}$ to make things prettier.

To try and get the low and high dimensional probability distributions as similar as possible, SNE minimizes the Kullback-Leibler divergence over all the points (there's a very nice [post about Kullback-Leibner divergence involving dangerous space worms here](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)). The optimization is done by gradient descent with the cost function, $C$, as below:
$$
	C = \sum\_i KL(P\_i || Q\_i) = \sum\_i \sum\_j p\_{j|i} log \frac{p\_{j|i}}{q\_{j|i}}
$$
where $P\_i$ denotes the conditional probability over all $x\_j$ given $x\_i$ and $Q\_i$ denotes the conditional probability over all $y\_j$ given $y\_i$.

The problem here, is that the Kullback Leibler divergence is not symmetric. Two types of mistakes can occur:

1. Points that are close in high dimensions are mapped to low dimensional points that are far apart. 
2. Points that are far away in high dimensions are mapped to low dimensional points that are close together. 

Due to the cost function's asymmetry, the first type of mistake is very costly, but the second type of mistake has only a small cost.

t-SNE uses a symmetric cost function which is easier to optimize. Instead of considering $P\_i$ and $Q\_i$ for each $i$ separately, t-SNE defines join probability distributions $P$ and $Q$ (in the high and low dimensional spaces, respectively). Now, instead of optimizing the sum of the KL divergence for each $i$, we only need to optimize the KL divergence between $P$ and $Q$. The cost function then becomes:
$$
	C = KL(P || Q) =  \sum\_i \sum\_j p\_{ij} log \frac{p\_{ij}}{q\_{ij}}
$$
Great, a simpler, symmetric cost function. The only remaining question is how to define the joint probability distributions. The most obvious way to define the pairwise similarities is by
$$
	p\_{ij} = \frac{exp(-||x\_i - x\_j||^2 / 2 \sigma\_j^2)}{ \sum\_{k \neq i} exp(-||x\_i - x\_k||^2 / 2 \sigma\_j^2)}
$$
for the high dimensional case and 
$$
	q\_{ij} = \frac{exp(-||y\_i - y\_j||^2 )}{ \sum\_{k \neq i} exp(-||y\_i - y\_k||^2 )}
$$
for the low dimensional case. 

But as often happens, the obvious solutions are not ncessarilyy good solutions. If $x\_i$ is an outlier that is far away from all the other points, then $||x\_i - x\_j||^2$ is large and $p\_{ij}$ is very small for every other $x\_{j}$. The influence of point $x\_{i}$ on the cost function is determined by $\sum\_j p\_{ij} log \frac{p\_{ij}}{q\_{ij}}$. Each $p\_{ij} log \frac{p\_{ij}}{q\_{ij}}$ term goes to zero when $p\_{ij}$ decreases, as you can see from the plot below.

As a result, the chosen location for $y_i$ has very little influence over the value of the cost function. The low dimensional points assigned to outliers can thus be a poor representation of their high dimensional positions.

A less obvious, but reasonable way to define the joint probabilities for the high dimensional points is by:
$$
	p\_{ij} = \frac{p\_{j|i} + p\_{i|j}}{2n}
$$
Then, using the fact that [$\sum\_{k} p(A_k | B) = 1$](https://en.wikipedia.org/wiki/Conditional_probability_distribution#Properties), we see
$$
	\sum\_{j}^n p\_{ij} = \sum\_{j}^n \frac{p\_{j|i} + p\_{i|j}}{2n}\\\\= \sum\_{j}^n \frac{p\_{j|i}}{2n} + \sum\_{j}^n \frac{p\_{i|j}}{2n} \\\\= 1 + \sum\_{j}^n \frac{p\_{i|j}}{2n} > \frac{1}{2n}
$$
so that the accumulated influence of the point $x_i$ is always sufficiently large. 

Another problem occurs with the cursory definition for the low dimensional joint probabilities.

$$
	q\_{ij} = \frac{(1 + ||y\_i - y\_j||^2)^{-1}}{\sum\_{k \neq l}(1 + ||y\_k - y\_l||^2)^{-1}}
$$

#### Why t-SNE doesn't play well with time



#### Making t-SNE take time into consideration

#### Fantastical Implosions


\begin{cases}
\dot{x} & = \sigma(y-x) \newline
\dot{y} & = \rho x - y - xz \newline
\dot{z} & = -\beta z + xy
\end{cases}
