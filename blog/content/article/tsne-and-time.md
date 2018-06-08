---
title: "T-SNE and Time: Part One"
date: 2018-05-30T12:58:35+02:00
draft: true

featuredImage: "/images/patreon.png"
categories: [code,math]
tags: [t-SNE, high dimensions, sequences, time series, visualization]
author: "Elre"
---

### What is t-SNE?

[t-Distributed Stochastic Neighbor Embedding (t-SNE)](https://lvdmaaten.github.io/tsne/) is a visualization technique for high dimensional datasets. Well, technically it is a dimensionality reduction technique that also serves as an excellent visualization technique.

t-SNE is based on [Stochastic Neighbor Embedding](https://www.cs.toronto.edu/~fritz/absps/sne.pdf), which is another dimensionality reduction technique proposed by Hinton and Roweis in 2002. 

The general aim is to create a mapping from high dimensional data points to low dimensional data points where similar points are grouped together and dissimilar points are further apart. The technique defines similarity in terms of the conditional probability that two points will be neighbours, where the probability of them being neighbours depends on the distance between them. The structure of the high dimensional points can then be imitated in by choosing low dimensional points that have the same relationship with their neighbours as their high dimensional counterparts. The technique is thus based on distance preservation, but also preserves neighbourhood structure or topology by converting distances to probabilities.

More precisely, for every pair of high dimensional points, $x\_i$ and $x\_j$ the similarity between them is defined as the conditional probability, $p\_{j|i}$, that $x\_j$ will be $x\_i$'s neighbour, assuming that $x\_i$'s neighbours are distributed according to a Gaussian centered at $x\_i$. The similarity between $x\_i$ and $x\_j$ can thus be expressed as
$$
	p\_{j|i} = \frac{exp(-||x\_i - x\_j||^2 / 2 \sigma\_i^2)}{ \sum\_{k \neq i} exp(-||x\_i - x\_k||^2 / 2 \sigma\_i^2)}
$$
where $\sigma\_i$ denotes the variance of the Gaussian centered at $x\_i$. In this way, the distance between two high dimensional points is converted into a conditional probability, where closer points are more likely to be neighbours and further points become exponentially less likely to be neighbours.
For now, pretend that we know $\sigma\_i$ for all the high dimensional points. If this makes you uncomfortable, skip ahead to the end.

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

### The Beauty (And Practical Advantages) of Symmetry

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
But as often happens, the obvious solutions are not necessarily good solutions.

#### The Problem with $p\_{ij}$
 If $x\_i$ is an outlier that is far away from all the other points, then $||x\_i - x\_j||^2$ is large and $p\_{ij}$ is very small for every other $x\_{j}$. The influence of point $x\_{i}$ on the cost function is determined by $\sum\_j p\_{ij} log \frac{p\_{ij}}{q\_{ij}}$. Each $p\_{ij} log \frac{p\_{ij}}{q\_{ij}}$ term goes to zero when $p\_{ij}$ decreases, as you can see from the plot below.

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

#### The Problem with $q\_{ij}$

The cursory definition for the low dimensional joint probabilities is also inadequate: it is vulnerable to the crowding problem. Intuitively, and perhaps unsurprisingly, the crowding problem arises because there is more directions and more space in high dimensional spaces. 

###### *Directions*

Consider the following example. Let our high dimensional space be a square centered at 0 with side lengths of 2 units. Our high dimensional points are $x\_1, ...x\_5$, as shown in the figure below.
The low dimensional space will be the interval $\[-1, 1\]$.
Let $x\_1$'s embeded point $y\_1$ be at zero.

$x\_1$'s has four, distinct equidistant neighbours. To fully preserve the topology of the high dimensional points, $x\_1$ should also have four distinct equidistant neighbours in the low dimensional space. However, in 1-D, it is impossible to have more than two distinct equidistant neighbours.

There is thus loss of information when mapping from high dimensions to low dimensions. This shouldn't be very surprising: by definition, the lower dimensional space has fewer dimensions/directions, so information that is intrinsically characteristic of dimensionality will be lost in low dimensions.

###### *Space*

Hyper-volume grows exponentially with dimensionality (see below). Lower dimensions also have less available space/volume for the mapped points to occupy. 

SNE is a local technique, meaning that local structure in the high dimensional space is also preserved in the low dimensional space. This is a useful property, but it essentially leads to "scaling issues".
To accurately portray the local structure of $x\_i$'s neighbourhood, with minimal loss of information, requires a large portion of the available room in the low dimensional space. Other points which are not in the immediate vicinity of $x\_i$ in the high dimensional space end up being mapped very far away from $y\_i$ because all of the space near $x\_i$ is already occupied. 

Since the low dimensional distance is too large, the optimization algorithm will attempt to rectify the situation by essentially crushing all the points near $x\_i$ together in the center so that the points which were put too far away can be closer to $x\_i$. All information about local structure is lost.

>**Gradients as Springs**  
The movement of the points can be thought of in terms of springs. The point $y\_i$ is connected to all of its neighbours $y\_j$ by a spring that is either pulling them together or pushing them apart. The spring pulls two points together if the distance between them is too large (in proportion to the distance between their high dimensional counterparts. Conversely, the spring pushes the two points apart if the distance between them is too small. The force exerted by the spring is proportional to the distance between the points as well as the mismatch between the high and low probability distributions. This analogy holds very well mathematically and is based on a physical interpretation of the gradient:
$$
\frac{\partial C}{\partial y_i} = 4 \sum\_j (p\_{ij} - q\_{ij})(y\_i - y\_j)
$$
The direction of the spring's force is determined by $y\_i - y\_j$ and the magnitude of the force is proportional to $(p\_{ij} - q\_{ij})$ as well as the distance between the points, $y\_i - y\_j$.

t-SNE addresses the crowding problem by using a Student's t-distribution (hence the "t" in "t-SNE"). In particular, the Student's t-distribution with one degree of freedom is used.
$$
	q\_{ij} = \frac{(1 + ||y\_i - y\_j||^2)^{-1}}{\sum\_{k \neq l}(1 + ||y\_k - y\_l||^2)^{-1}}
$$
The Student's t-distribution has heavier tails than a Gaussian (see below). The Student's t-distribution makes a less sharp distinction between near and far-away neighbours. Intermediate points that are not in the immediate vicinity of $x\_i$ are mapped to more accurate low dimensional locations, thereby preventing the crowding problem.

Other heavy-tailed distributions or Student's t with more degrees of freedom may also do the job. However, the Student's t-distribution with one degree of freedom has the very nice property that $(1 + ||y\_i - y\_j||^2)^{-1}$ approaches an inverse square law when $||y\_i - y\_j||$ is large. 
This prevents points that are very far apart from exerting undue influence on the scale of the mapping (i.e. prevents the scale of the map from becoming very large when points are very far away). Another bonus is that the Student's t-distribution is easier to evaluate since it does not involve exponentials.

### Putting It All Together

Armed with our new, symmetric joint probability functions let us re-state the cost function:
$$
	C = KL(P || Q) =  \sum\_i \sum\_j p\_{ij} log \frac{p\_{ij}}{q\_{ij}}
$$
In order to throw gradient descent at the problem, we need to procure the cost function:
$$
\frac{\partial C}{\partial y_i} = 4 \sum\_j (p\_{ij} - q\_{ij})(y\_i - y\_j)(1 + ||y\_i - y\_j||^2)^{-1}
$$
(See the [Appendix of van der Maaten and Hinton's paper](https://lvdmaaten.github.io/tsne/) to see where this equation was conjured from).
The gradient descent update rule is given by
$$
	y\_i^t = y\_i^{t-1} + \eta \frac{\partial C}{\partial y\_i} + \alpha(t)(y\_i^{t-1}  - y\_i^{t-2} )
$$
where $\eta$ is the learning rate. The last term is a momentum term to speed up optimization. The step taken in the last iteration, $y\_i^{t-1}  - y\_i^{t-2}$, is scaled according to $\alpha(t)$. Usually, $\alpha(t)$ is higher for the first few iterations and lower for the rest of the search. To ensure that the steps converge, $0 \leq \alpha \leq 1$. The official implementations set $\alpha(t) = 0.8$ for $t < 250$ and $\alpha(t) = 0.5$ for $t \geq 250$. The learning rate can be constant or updated by an adaptive learning rate scheme.

>**It's the Little Things**  
Van der Maaten and Hinton also proposed two small tweaks to the original algorithm to improve performance. The first is called *early compression*, which forces the particles so stay close together at the beginning of the search. Since all the map points are close together, it is easy for clusters to move around, so different cluster configurations can be explored until the best topology is found. Early compression simply adds an L-2 penalty term to the cost function for the first part of the optimization, thereby encouraging points to stay together.
The second tweak is called *early exaggeration* which exaggerates the value of all the $p\_{ij}$'s for the first part of the search. The optimization algorithm corrects for this by making the $q\_{ij}$'s made much larger. This produces clusters with high inter-clusteral distances, where similar points are grouped tightly together, and low intra-clusteral distances so that clusters can easily be distinguished and move around.

### The Quest for $\sigma\_i$




#### Why t-SNE doesn't play well with time



#### Making t-SNE take time into consideration

#### Fantastical Implosions


\begin{cases}
\dot{x} & = \sigma(y-x) \newline
\dot{y} & = \rho x - y - xz \newline
\dot{z} & = -\beta z + xy
\end{cases}
