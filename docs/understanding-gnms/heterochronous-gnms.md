# Heterochronous Generative Network Models

## Motivation: why account for developmental time?

Classic [binary generative network models](binary-gnms.md) (GNMs) grow a synthetic connectome by balancing a topological "value" term (e.g. homophily) against a spatial "cost" term (Euclidean distance). These models reproduce hallmark *topological* properties of empirical connectomes but routinely misplace those properties in physical space—for example, hubs appear in anatomically implausible locations and modules drift away from canonical lobes. A key biological feature missing from the classical formulation is **heterochronicity**: different cortical and sub‑cortical areas begin wiring at different times.

Incorporating a third, time‑dependent term into the wiring rule allows GNMs to respect this developmental programme. Heterochronous GNMs therefore provide a principled bridge between biological mechanism and computational model.

------------------------------------------------------------------------

## How the heterochronous GNM modifies the standard algorithm

Let

-   $d_{ij}$ be the [distance transform](glossary.md#distance-transform-d) derived from the Euclidean distance matrix $D$, controlled by parameter $\eta$.
-   $k_{ij}$ be the [affinity transform](glossary.md#affinity-transform-k) derived from the affinity/homophily matrix $K$, controlled by parameter $\gamma$.
-   $h_{ij}(t)$ be the **heterochronicity transform** derived from a time‑varying heterochronicity matrix $H(t)$, controlled by parameter $\lambda$.

At iteration $t$ of network growth we compute *unnormalised* wiring probabilities, we must compute [wiring probabilities](glossary.md#wiring-probabilities) $P_{ij}$ that each pair of currently unconnected nodes will form a connection. The process is the same as the one described for [binary GNMs](binary-gnms.md#computing-wiring-probabilities) with the only difference that also the heterochronicity matrix $H$ is now multiplied to the distance and affinity matrices, $D$ and $K$.

------------------------------------------------------------------------

## Modelling the heterochronous gradient

The exact shape of the heterochronous gradient is unknown. However, we know that we want to capture a smooth change over space and time in the probability of a certain node to make a connection. for this reason, we can compute heterochronicity using a time‑varying cumulative Gaussian function. More specifically, the likelihood that node $i$ can establish any connection at time $t$ is given by

$$
 h_{it} = \exp\!\left[-\frac{(d_i - \mu_t)^2}{2\sigma^2}\right],
$$

where $d_i$ is the Euclidean distance of node $i$ from the origin point, $\sigma$ is the standard deviation of the Gaussian, and the centre $\mu_t$ changes across time according to

$$
 \mu_t = \frac{t}{T-1}\,\max_i d_i ,
$$

thus placing the centre of the Gaussian at equally spaced positions along the distance axis from the reference point out to the farthest node. Importantly, once a node becomes *active* and starts making connections, it remains active until the end of the heterochronous process:

$$
 h^{a}_{it} = \max_{t' \le t} h_{i t'}.
$$

Once the likelihood of connecting is defined for each node, an undirected matrix specifies the likelihood of any two nodes $i$ and $j$ connecting:

$$
 H_{ij,t} = \max\!\bigl(h^{a}_{it}, h^{a}_{jt}\bigr). 
$$

------------------------------------------------------------------------

## Model Fitting

When fitting heterochronous GNMs, we are interested in capturing not only the topological features of empirical networks but also their topography. For this reason, we should always use an Energy function that combines both types of statistics--topological and topographical--as illustrated in the section on [Fitting GNMs](fitting-gnms.md)

------------------------------------------------------------------------

## Summary

Heterochronous GNMs enrich the classical cost–benefit framework with a biologically motivated temporal gradient. A simple Gaussian activation wave suffices to capture developmental constraints, substantially improving both topological and topographical realism while retaining the interpretability and analytical tractability of GNMs.