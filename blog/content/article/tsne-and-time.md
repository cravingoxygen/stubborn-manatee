---
title: "T-SNE and Time: Part One"
date: 2018-06-13T13:12:35+02:00
draft: false

featuredImage: "/images/patreon.png"
categories: [code,math]
tags: [t-SNE, high dimensions, sequences, time series, visualization]
author: "Elre"
---
This post is the first in a multi-part series on t-SNE and its use for dynamic high dimensional data. This post explains the basics of t-SNE. The next post will explain the difficulties of using t-SNE for iterative data if it is applied without further modification. The last post will discuss the author's approach to visualizing time-varying data with a modified t-SNE. The Golang source code for these posts will be made available.

### What is t-SNE?

[t-Distributed Stochastic Neighbor Embedding (t-SNE)](https://lvdmaaten.github.io/tsne/) is a visualization technique for high dimensional datasets. Well, technically it is a dimensionality reduction technique that also serves as an excellent visualization technique. Below is an image from van der Maaten and Hinton's paper, showing how t-SNE represents handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) in two dimensions.

<center>
{{< figure src="/images/tsne/tsne.png" caption="t-SNE Applied to 6000 Handwritten Digits from MNIST" width="100%">}}
</center>

If you were doing exploratory analysis of the MNIST dataset, t-SNE makes it immediately obvious that there are 10 different clusters and even gives an indication of similarities between clusters.
For comparison, here is a Sammon mapping's considerably more obscure representation of the same data:

<center>
{{< figure src="/images/tsne/sammon.png" caption="Sammon Map Applied to 6000 Handwritten Digits from MNIST" width="100%">}}
</center>

t-SNE is based on [Stochastic Neighbor Embedding (SNE)](https://www.cs.toronto.edu/~fritz/absps/sne.pdf), which is another dimensionality reduction technique proposed by Hinton and Roweis in 2002.  
SNE aims to create a mapping from high dimensional data points to low dimensional data points where similar points are grouped together and dissimilar points are further apart. The technique defines similarity in terms of the conditional probability that two points will be neighbours, where the probability of them being neighbours depends on the distance between them. The structure of the high dimensional points can be imitated by the mapped points by defining it so that low dimensional points have the same relationship with their neighbours as their high dimensional counterparts. The technique is thus based on distance preservation, but also preserves neighbourhood structure or topology.

More precisely, for every pair of high dimensional points, $x\_i$ and $x\_j$ the similarity between them is defined as the conditional probability, $p\_{j|i}$, that $x\_j$ will be $x\_i$'s neighbour, assuming that $x\_i$'s neighbours are distributed according to a Gaussian centered at $x\_i$. The similarity between $x\_i$ and $x\_j$ can thus be expressed as
$$
	p\_{j|i} = \frac{exp(-||x\_i - x\_j||^2 / 2 \sigma\_i^2)}{ \sum\_{k \neq i} exp(-||x\_i - x\_k||^2 / 2 \sigma\_i^2)}
$$
where $\sigma\_i$ denotes the variance of the Gaussian centered at $x\_i$. In this way, the high dimensional distance is converted into a conditional probability, where closer points are more likely to be neighbours and further points become exponentially less likely to be neighbours.
For now, pretend that we know $\sigma\_i$ for all the high dimensional points. If this makes you uncomfortable, skip ahead to the end of the post.

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
<center>
{{< figure src="/images/tsne/too_far.png" caption="The red point is mapped too far away from the blue point" width="100%">}}
</center>
2. Points that are far away in high dimensions are mapped to low dimensional points that are close together. 
<center>
{{< figure src="/images/tsne/too_close.png" caption="The red point is mapped too close to the blue point" width="100%">}}
</center>

Due to the cost function's asymmetry, the first type of mistake is very costly, but the second type of mistake has only a small cost. Since both types of mistakes lead to inaccurate interpretation of the data, this is not ideal.

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
<center>
{{< figure src="/images/tsne/xlogx.png" caption="Plot of $p_{ij} log \frac{p_{ij}}{q_{ij}}$ for various $q_{ij}$">}}
</center>

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

The cursory definition for the low dimensional joint probabilities is also inadequate: it is vulnerable to the crowding problem. Intuitively, and perhaps unsurprisingly, the crowding problem arises because there are more directions and more space in high dimensional spaces. 

###### *Directions*

Consider the following example. Let our high dimensional space be a square centered at 0 with side lengths of 2 units. Our high dimensional points are $x\_1, ...x\_5$, as shown in the figure below.
The low dimensional space will be the interval $\[-1, 1\]$.
Let $x\_1$'s embedded point $y\_1$ be at zero.
<center>
{{< figure src="/images/tsne/embed_x1.png" caption="Map $x_1$ from the 2-D space onto the interval $[-1,1]$">}}
</center>
$x\_1$'s has four, distinct equidistant neighbours. To fully preserve the topology of the high dimensional points, $x\_1$ should also have four distinct equidistant neighbours in the low dimensional space. However, in 1-D, it is impossible to have more than two distinct equidistant neighbours.
<center>
{{< figure src="/images/tsne/embed_rest.png" caption="Only two of the other points can be equi-distant from $x_1$ when mapped to $[-1,1]$">}}
</center>
There is thus loss of information when mapping from high dimensions to low dimensions. This shouldn't be very surprising: by definition, the lower dimensional space has fewer dimensions/directions, so information that is intrinsically characteristic of dimensionality will be lost in low dimensions.

###### *Space*

Hyper-volume grows exponentially with dimensionality. Lower dimensions thus have less available space/volume for the mapped points to occupy. SNE is a local technique, meaning that local structure in the high dimensional space is also preserved in the low dimensional space. This is a useful property, but it essentially leads to "scaling issues".

Consider the following example, where the high dimensional space is 2-D and $x\_i$ is represented by the red dot. The immediate neighbours of $x\_i$ are also dots. Additionally, there are three other clusters, which are moderately far away from $x\_i$, represented by triangles, squares, and crosses.
<center>
{{< figure src="/images/tsne/crowding_high_dim.png" width="100%">}}
</center>
To accurately portray the local structure of $x\_i$'s neighbourhood, with minimal loss of information, requires a large portion of the available room near $y\_i$ (red dot) in the low dimensional space. As shown below, the area immediately around $y\_i$ is taken up by all the other dots, since they are $x\_i$'s immediate neighbours.
<center>
{{< figure src="/images/tsne/crowding_neighbours.png" width="100%">}}
</center>
As a result, other points which are not in the immediate vicinity of $x\_i$ in the high dimensional space end up being mapped very far away from $y\_i$ because all of the space near $y\_i$ is already occupied. 
<center>
{{< figure src="/images/tsne/crowding_all_points.png" width="100%">}}
</center>
The low dimensional distance between $y\_i$ and the filled triangle, $y\_j$, is too large considering the distance between $x\_i$ and $x\_j$. 
<center>
{{< figure src="/images/tsne/crowding_too_far.png" width="100%">}}
</center>
The optimization algorithm will attempt to rectify the situation by moving $y\_i$ closer to $y\_j$.
<center>
{{< figure src="/images/tsne/crowding_squash_triangle.png" width="100%">}}
</center>
But $y\_i$ is too far away from all the triangles, the squares and the crosses! The sum of all these small movements ends up crushing all the other points towards $y\_i$. 
<center>
{{< figure src="/images/tsne/crowding_no_structure.png" width="100%">}}
</center>
This same situation occurs for each of the points, leading to all of the points being crushed into the center of the map, making it hard to distinguish different clusters. All information about local structure is lost.

>**Gradients as Springs**  
The movement of the points can be thought of in terms of springs. The point $y\_i$ is connected to all of its neighbours $y\_j$ by a spring that is either pulling them together or pushing them apart. The spring pulls two points together if the distance between them is too large (in comparison to the distance between their high dimensional counterparts). Conversely, the spring pushes the two points apart if the distance between them is too small. The force exerted by the spring is proportional to the distance between the points as well as the mismatch between the high and low probability distributions. This analogy holds very well mathematically and is based on a physical interpretation of the gradient:
$$
\frac{\partial C}{\partial y_i} = 4 \sum\_j (p\_{ij} - q\_{ij})(y\_i - y\_j)
$$
The direction of the spring's force is determined by $y\_i - y\_j$ and the magnitude of the force is proportional to $(p\_{ij} - q\_{ij})$ as well as the distance between the points, $y\_i - y\_j$.

#### New and Improved $q\_{ij}$
t-SNE addresses the crowding problem by using a Student's t-distribution (hence the "t" in "t-SNE"). In particular, the Student's t-distribution with one degree of freedom is used.
$$
	q\_{ij} = \frac{(1 + ||y\_i - y\_j||^2)^{-1}}{\sum\_{k \neq l}(1 + ||y\_k - y\_l||^2)^{-1}}
$$
The Student's t-distribution has heavier tails than a Gaussian (see below). The Student's t-distribution makes a less sharp distinction between near and far-away neighbours. Intermediate points that are not in the immediate vicinity of $x\_i$ are mapped to more accurate low dimensional locations, thereby preventing the crowding problem.
<center>
{{< figure src="/images/tsne/gaussian_t-sne_pdfs.png" >}}
</center>
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
where $\eta$ is the learning rate. The learning rate can be constant or updated by an adaptive learning rate scheme. The last term is a momentum term to speed up optimization. The step taken in the last iteration, $y\_i^{t-1}  - y\_i^{t-2}$, is scaled according to $\alpha(t)$. Usually, $\alpha(t)$ is higher for the first few iterations and lower for the rest of the search. To ensure that the steps converge, $0 \leq \alpha \leq 1$. The official implementations set $\alpha(t) = 0.8$ for $t < 250$ and $\alpha(t) = 0.5$ for $t \geq 250$. 

>**It's the Little Things**  
Van der Maaten and Hinton also proposed two small tweaks to the original algorithm to improve performance. The first is called *early compression*, which forces the particles so stay close together at the beginning of the search. Since all the map points are close together, it is easy for clusters to move around, so different cluster configurations can be explored until the best topology is found. Early compression simply adds an L-2 penalty term to the cost function for the first part of the optimization, thereby encouraging points to stay together.  
The second tweak is called *early exaggeration* which exaggerates the value of all the $p\_{ij}$'s for the first part of the search. The optimization algorithm corrects for this by making the $q\_{ij}$'s made much larger. This produces clusters with high intra-clusteral distances, where similar points are grouped tightly together, and low inter-clusteral distances so that clusters can easily be distinguished and move around.

### Next up
The next post in this series will consider dynamic or time-dependent problems, where the high dimensional data changes over a number of time-steps. We also discuss the use of t-SNE to visualize what is taking place in the high dimensional space over time.

### Appendix: The Quest for $\sigma\_i$
	
An important caveat of the discussion above is the value chosen for $\sigma\_i$. Recall that $\sigma\_i$ is the variance of the Gaussian that represents $x\_i$'s neighbourhood. If the high dimensional data is sufficiently interesting to warrant exploratory analysis, then it is unlikely that the same $\sigma\_i$ can be used for all the data points. If $x\_i$ is in a region with many other points nearby, then $\sigma\_i$ should be small so that only the nearest points are classified as neighbours. On the other hand, if $x\_i$ occurs in a sparse region, then $\sigma\_i$ should be larger since $x\_i$'s closest neighbours will be far away.

In order to calculate the appropriate $\sigma\_i$, we make use of *perplexity*, which can be interpreted as a measure for the effective number of neighbours that each data point should have. Mathematically, the perplexity is defined as
$$
 perp(P\_i) = 2^{H(P\_i)}
$$
where $P\_i$ denotes the probability distribution induced by a given value of $\sigma\_i$ over all $x\_j$, and $H(P\_i)$ denotes the Shannon entropy of $P\_i$:
$$
	H(P\_i) = - \sum\_j p\_{j|i} \log\_2 p\_{j|i}
$$
Thus, for any given value of $\sigma_i$, a probability distribution is induced over all the points $x\_j$. We can calculate the entropy, $H(P\_i)$, for this distribution and from the entropy, we can calculate the resulting perplexity. 
So to calculate the value for $\sigma\_i$ that will be used by t-SNE, we choose a value for the perplexity that seems right and then search for a corresponding $\sigma\_i$ that will bring about that perplexity (using something like a [binary search](https://en.wikipedia.org/wiki/Binary_search_algorithm)).

The chosen value for the perplexity depends on the dataset. The smaller the perplexity, the fewer neighbours each data point will have. If the perplexity is too small then the algorithm takes considerably longer to form clusters and the clusters may end up fragmented. For example, the visualization below used a perplexity of 2 for a dataset of 1000 points. The green group was incorrectly broken up into two separate clusters. Both the red and blue clusters were also fragmented. The data points in the fragments are all each other's neighbours, so there is very little force pulling a fragment towards the rest of its cluster.
<center>
{{< figure src="/images/tsne/tsne_k=2.gif">}}
</center>

>**The Benchmark**  
The visualizations use a very simple benchmark with five centroids. The five centroids are initialized uniform randomly in $[-50,50]^{10}$. Each data point is assigned to one of the centroids and is initialized within some small distance of its assigned centroid. t-SNE should thus very easily be able to pick out the five clusters.

The larger the chosen value for the perplexity is, the larger every point's neighbourhood will be. For comparison, the example below chose a perplexity of 10. The clusters form much faster; there is space between the different clusters and none of the clusters are fragmented.
<center>
{{< figure src="/images/tsne/tsne_k=10.gif">}}
</center>

>**Note**  
*Early exaggeration* is visible in the first 100 iterations, where the clusters tend to form very tight points. Later, when early exaggeration is removed, the clusters expand again to reveal their internal structure. These examples do not apply *early compression*.

Below is a visualization with a perplexity of 50. Since the 1000 points are divided into 5 clusters, each cluster contains 200 points. Every data point thus consider a quarter of its fellow cluster members to be neighbours. This allows the clusters to form even faster than before. The clusters are also tighter, with more space between different clusters. 
<center>
{{< figure src="/images/tsne/tsne_k=50.gif">}}
</center>
If the dataset consists of 1000 points, then a perplexity of 400 implies that half of a data point's neighbours belong to a different cluster. The resulting visualization make it difficult to discern any structure within the cluster.
<center>
{{< figure src="/images/tsne/tsne_k=400.gif">}}
</center>
If the perplexity is set to 1000, then every point considers every other point to be its neighbour. All the points attract each other (note how small the values along the axes are) and no structure is visible.
<center>
{{< figure src="/images/tsne/tsne_k=1000.gif">}}
</center>
