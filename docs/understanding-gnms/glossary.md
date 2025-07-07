# Glossary of terms

### Adjacent
Two [nodes](#node) within a [graph](#graph) are said to be adjacent if there is an [edge](#edge) between them.

### Adjacency matrix, $A$
A matrix representation of a [network](#network). $A_{ij} = A_{ji} = 1$ when [node](#node) $i$ is [adjacent](#adjacent) to [node](#node) $j$ (*i.e.*, when there is an [edge](#edge) between $i$ and $j$), and $A_{ij} = A_{ji} = 0$ when they are not [adjacent](#adjacent) (*i.e.*, no such [edge](#edge) exists). Note that the adjacency matrices are [symmetric](#symmetric). 

### Affinity matrix, $K$
Matrix capturing non-geometric factors influencing the likelihood that [nodes](#node) form an [edge](#edge) between them. Computed using a [generative rule](#generative-rule).

### Affinity transform, $k$
Parametrised [transform](#transform) of the [affinity matrix](#affinity-matrix-k) which effects the [wiring probabilities](#wiring-probabilities-p_ij). When the affinity relationship type is [powerlaw](#powerlaw-relationship-type), the affinity transform is $k_{ij} = K_{ij}^\gamma$; when the affinity relationship type is [exponential](#exponential-relationship-type), the affinity transform is $k_{ij} = \exp( \gamma K_{ij} )$. The affinity transform is multiplied by the [distance transform](#distance-transform-d) and the [heterochronous transform](#heterochronicity-transform-h) to obtain (unnormalised) [wiring probabilities](#wiring-probabilities-p_ij).

### Alpha, $\alpha$
Parameter of the [weighted GNM](#weighted-generative-network-model). Controls the size of the update on the weights of the [network](#network) at each weighted update step; equivalent to the learning rate on the [weight optimisation criteron](#weight-optimisation-criterion) (*i.e.*, the [loss](#loss-l)), $L$. The update performed at each step is $W_{ij} \gets W_{ij} - \alpha \frac{\partial L}{\partial W_{ij}}$.

### Betweenness Centrality
A way of measuring the importance of a [node](#node) for communication flow through a [network](#network). For binary [networks](#network), the betweenness centrality of a [node](#node) $a$ is computed by first looking at all other pairs of [nodes](#node) in the [network](#network) which are [connected](#connected) to one another, $b$ and $c$. For each pair of other [connected](#connected) [nodes](#node), $b$ and $c$, we compute the fraction of [shortest paths](#shortest-path) between $b$ and $c$ which pass through $a$; for example, if all [shortest paths](#shortest-path) between $b$ and $c$ go through $a$ this is $1$, and if none of the [shortest paths](#shortest-path) go through $a$ then this is $0$. We then sum this fraction over all these pairs to compute the betweenness centrality of $a$.

### Binary Generative Network Model
The standard generative network model. This model generates [graphs](#graph) representing connectivity between brain regions by iteratively adding [edges](#edge). At each iteration, the (unnormalised) [wiring probability](#wiring-probabilities-p_ij) is computed by multiplying the [distance transform](#distance-transform-d) and the [affinity transform](#affinity-transform-k). We then sample from these [wiring probabilities](#wiring-probabilities-p_ij) to choose which [edge](#edge) to add.

### Characteristic path length
The average [shortest path](#shortest-path) length within a [graph](#graph). To compute characteristic path length, we consider all pairs of [nodes](#node) within the [graph](#graph) $a$, $b$, and compute the [shortest path](#shortest-path) length between $a$ and $b$. We then average over these pairs to get the characteristic path length of the [graph](#graph).

### Clustering coefficient
Measure of local clustering in a [network](#network). The clustering coefficient of a [node](#node) $a$ is equal to the proportion of [neighbours](#neighbour) of $a$ which are [neighbours](#neighbour) with each other. The clustering coefficient of a [network](#network) is the average clustering coefficient of all the [nodes](#node) in that [network](#network).

### Communicability
A measure of the total influence that one [node](#node) has on another. Given a [weight matrix](#weight-matrix-w) $W$, to compute the communicability we begin by computing normalised weights $\hat{W}$ via $\hat{W}_{ij} = \frac{W_{ij}}{\sqrt{s_i s_j}}$ where $s_i$ and $s_j$ are the [node strengths](#node-strength-s) of $i$ and $j$ respectively. The communicability is then the matrix exponential of $\hat{W}$, $C = \exp(\hat{W}) = I + \frac{1}{1!} \hat{W} + \frac{1}{2!} \hat{W}^2 + \frac{1}{3!} \hat{W}^3 + \dots$.

### Connected
Two [nodes](#node) within a [network](#network) are said to be connected if there is any [path](#path) which joins them, *i.e.*, any sequence of [nodes](#node) which begins with one [node](#node) and ends with the other, such that each [node](#node) in the sequence is [adjacent](#adjacent) to both the [nodes](#node) before and after. For example, if $a$ and $b$ are [neighbours](#neighbour) and $b$ and $c$ are [neighbours](#neighbour), then $a$ and $c$ are connected via the [path](#path) $a,b,c$.

### Correlation
Take a pair of [networks](#network) with the same set of $N$ [nodes](#node), but different [edges](#edge). Consider some nodal property, such as [clustering coefficient](#clustering-coefficient) or [betweenness centrality](#betweenness-centrality). Then if we denote that property for [node](#node) $i$ in the first [network](#network) by $X_i$, and for [node](#node) $i$ in the second [network](#network) by $Y_i$, the correlation of that graph property between the [networks](#network) is $$\rho = \frac{ \sum_i (X_i - \bar{X})(Y_i - \bar{Y}) }{ \sqrt{ ( \sum_i (X_i - \bar{X})^2  )( \sum_i (Y_i - \bar{Y})^2 ) } },$$ where $\bar{X} = \frac{1}{N} \sum_i X_i$ and $\bar{Y} = \frac{1}{N} \sum_i Y_i$ are the average nodal properties for each of the two [networks](#network).

### Cumulative distribution function
For a graph property, the cumulative distribution function at a point is the fraction of values of that graph property less than that point.

### Degree
The number of [edges](#edge) attached to a [node](#node); equal to the number of [neighbours](#neighbour) the [node](#node) has.

### Density
The fraction of possible [edges](#edge) in a [network](#network) which are actually present. Equal to the number of edges divided by $N(N-1)/2$.  

### Distance Matrix, $D$
Matrix storing the physical distances between [nodes](#node) in the [network](#network). $D_{ij}$ gives the distance in space between [nodes](#node) $i$ and $j$.

### Distance transform, $d$
Parametrised [transform](#transform) of the [distance matrix](#distance-matrix-d) which effects the [wiring probabilities](#wiring-probabilities-p_ij). When the distance relationship type is [powerlaw](#powerlaw-relationship-type), the distance transform is $d_{ij} = D_{ij}^\eta$; when the distance relationship type is [exponential](#exponential-relationship-type), the distance transform is $d_{ij} = \exp( \eta D_{ij} )$. The distance transform is multiplied by the [affinity transform](#affinity-transform-k) to obtain (unnormalised) [wiring probabilities](#wiring-probabilities-p_ij).

### Distribution
The collection of values taken by a graph property across the [network](#network). For example, the distribution of [clustering coefficients](#clustering-coefficient) is the collection of [clustering coefficient](#clustering-coefficient) values of each [node](#node) within the [network](#network); the [edge length](#edge-length) distribution is the collection of [edge lengths](#edge-length) of each [edge](#edge) within the [network](#network).

### Edge
Direct connection between two [nodes](#node) in a [graph](#graph).

### Edge length
The distance between [nodes](#node) connected by a single [edge](#edge). For example, if $i$ and $j$ are [neighbours](#neighbour), the edge length between [nodes](#node) $i$ and $j$ is $D_{ij}$, where $D$ is the [distance matrix](#distance-matrix-d). Note that edge length depends on the physical position of [nodes](#node) in space, and not (only) on the [topological](#topology) properties of the [graph](#graph).

### Eta, $\eta$
A parameter of the [binary GNM](#binary-generative-network-model). $\eta$ controls the influence of the [distance matrix](#distance-matrix-d) $D_{ij}$ on [wiring probabilities](#wiring-probabilities-p_ij). When the relationship type is [powerlaw](#powerlaw-relationship-type), the [distance transform](#distance-transform-d) is $d_{ij} = D_{ij}^\eta$; when the relationship type is [exponential](#exponential-relationship-type), the [distance transform](#distance-transform-d) is $d_{ij} = \exp( \eta D_{ij} )$.

### Evaluation criterion
Method of measuring the similarity between two [adjacency matrices](#adjacency-matrix-a) or two [weight matrices](#weight-matrix-w). For example, the maximum of a [KS distance](#kolmogorov-smirnov-ks-distance) between graph measures for each of the [graphs](#graph).

### Exponential relationship type
Method of determining the [distance](#distance-transform-d), [affinity](#affinity-transform-k), and [heterochronicity transforms](#heterochronicity-transform-h) from their respective matrices. Under the exponential relationship type, the [distance transform](#distance-transform-d) is $d_{ij} = \exp( \eta D_{ij} )$, the [affinity transform](#affinity-transform-k) is $k_{ij} = \exp( \gamma K_{ij} )$, and the [heterochronicity transform](#heterochronicity-transform-h) is $h_{ij} = \exp( \lambda H_{ij} )$.

### Functional connectivity
Connectivity between brain regions determined by the statistical relationships between the activities in the different regions. *cf.* [structural connectivity](#structural-connectivity)

### Gamma, $\gamma$
A parameter of the [binary GNM](#binary-generative-network-model). $\gamma$ controls the influence of the [affinity matrix](#affinity-matrix-k) $K_{ij}$ on [wiring probabilities](#wiring-probabilities-p_ij). When the affinity relationship type is [powerlaw](#powerlaw-relationship-type), the [affinity transform](#affinity-transform-k) is $K_{ij}^\gamma$; when the affinity relationship type is [exponential](#exponential-relationship-type), the [affinity transform](#affinity-transform-k) is $\exp( \gamma K_{ij} )$.

### Generative rule
Method for computing the [affinity matrix](#affinity-matrix-k) $K$ from the [adjacency matrix](#adjacency-matrix-a) $A$.

### Graph
Synonym for [network](#network). A set of [nodes](#node) with connections between them. Graphs can be either weighted or unweighted. Unweighted graphs are described by an [adjacency matrix](#adjacency-matrix-a), $A$, while weighted graphs are described via a [weight matrix](#weight-matrix-w), $W$.

### Heterochronicity
The influence of time on [wiring probabilities](#wiring-probabilities-p_ij).

### Heterochronicity Matrix, $H$
A matrix specifying how time influences [wiring probabilities](#wiring-probabilities-p_ij). If $H_{ij}$ is large on a particular time-step, it increases the probability of a connection between [nodes](#node) $i$ and $j$ being made on that time-step, assuming no connection already exists.

### Heterochronicity transform, $h$
Parametrised [transform](#transform) of the [heterochronous matrix](#heterochronicity-matrix-h) which effects the [wiring probabilities](#wiring-probabilities-p_ij). When the heterochronous relationship type is [powerlaw](#powerlaw-relationship-type), the heterochronous transform is $h_{ij} = H_{ij}^\lambda$; when the heterochronous relationship type is [exponential](#exponential-relationship-type), the heterochronous transform is $h_{ij} = \exp( \lambda H_{ij} )$. In the [heterochronous GNM](#heterochronous-generative-network-model), the (unnormalised) [wiring probabilities](#wiring-probabilities-p_ij) are obtained by multiplying not just the [distance transform](#distance-transform-d) and the [affinity transform](#affinity-transform-k), but additionally the heterochronous transform.

### Heterochronous Generative Network model
Modification of the GNM which allows for [heterochronicity](#heterochronicity) in the growth of the [network](#network). In the heterochronous GNM, the [wiring probabilities](#wiring-probabilities-p_ij) depend not just on the [distance matrix](#distance-matrix-d) and the current [affinity matrix](#affinity-matrix-k), but also on a [heterochronous matrix](#heterochronicity-matrix-h) which varies from step to step, making the formation of some connections more or less likely on some steps.

### Homophily
Any method of computing the [affinity matrix](#affinity-matrix-k) which assigns a higher affinity between pairs of [nodes](#node) which have a similar connection profile. This includes both the [Matching Index](#matching-index) and [Neighbours generative rules](#neighbours-generative-rule).

### Kolmogorov-Smirnov (KS) distance
A way of measuring how different two [distributions](#distribution) are, by taking the maximum difference between their [cumulative distribution functions](#cumulative-distribution-function).

### Lambdah, $\lambda$
A parameter of the [heterochronous binary GNM](#heterochronous-generative-network-model). $\lambda$ controls the influence of the [heterochronous matrix](#heterochronicity-matrix-h) $H_{ij}$ on [wiring probabilities](#wiring-probabilities-p_ij). When the heterochronous relationship type is [powerlaw](#powerlaw-relationship-type), the [heterochronous transform](#heterochronicity-transform-h) is $H_{ij}^\lambda$; when the affinity relationship type is [exponential](#exponential-relationship-type), the [affinity transform](#affinity-transform-k) is $\exp( \lambda H_{ij} )$.

### Loss, $L$
Synonym for [weight optimisation criterion](#weight-optimisation-criterion). For the [weighted GNM](#weighted-generative-network-model), an update is performed at each step which performs gradient descent on the loss with learning rate [$\alpha$](#alpha-alpha). 

### Matching Index
A [homophily](#homophily) [generative rule](#generative-rule) in which $K_{ij}$ is equal to the number of [neighbours](#neighbour) shared by $i$ and $j$, divided by the total number of [nodes](#node) which are [neighbours](#neighbour) of either $i$ or $j$.

### Neighbour
Two [nodes](#node) in a [graph](#graph) are said to be neighbours if there is an [edge](#edge) between them. In other words, $i$ and $j$ are neighbours if and only if $A_{ij} = 1$.

### Neighbours generative rule
A [homophily](#homophily) [generative rule](#generative-rule) in which $K_{ij}$ is equal to the number of [neighbours](#neighbour) shared by $i$ and $j$, *i.e.*, $K_{ij} = \sum_k A_{ik} A_{jk}$. Under this rule, $K_{ij}$ is larger the more [neighbours](#neighbour) $i$ and $j$ have in common.

### Network
Synonym for [graph](#graph). A set of [nodes](#node) with [edges](#edge) between them.

### Node
A point within a [network](#network). Within a [parcellation](#parcellation), each node is a distinct region of the brain.

### Node strength, $s$
The sum of the weights of the [edges](#edge) attached to a [node](#node). The node strength of [node](#node) $i$ is given by $\sum_j W_{ij}$. When the [weight matrix](#weight-matrix-w) is binary, the node strength is equal to the [degree](#degree) of a [node](#node).

### Omega, $\omega$
A parameter of the [weighted GNM](#weighted-generative-network-model), used to parametrise various loss functions for the [weight optimisation criteria](#weight-optimisation-criterion).

### (weight) optimisation criterion, $L$
[Loss](#loss-l) function used to update the [weight matrix](#weight-matrix-w) in the [weighted GNM](#weighted-generative-network-model). At each update step for the [weighted GNM](#weighted-generative-network-model), the weights are updated by performing gradient descent on the weight optimisation criterion $L$ with learning rates [$\alpha$](#alpha-alpha), $W_{ij} \gets W_{ij} - \alpha \frac{\partial L}{\partial W_{ij}}$.

### Parcellation
A division of part of the brain, typically the cortical surface, into distinct [nodes](#node). Parcellations can be performed on the basis of the connectivity profile between regions or the cytoarchitectural properties of each region.

### Path
A sequence of [nodes](#node) in a [graph](#graph) such that each [node](#node) is [connected](#connected) to the [nodes](#node) before and after in the sequence, and no [node](#node) is repeated. For example, $a,b,c$ is a path if $a$ and $b$ are [connected](#connected) and $b$ and $c$ are [connected](#connected). Paths can also be described by the sequence of [edges](#edge) traversed, *e.g.*, $(a,b),(b,c)$.

### Path length
The number of [edges](#edge) found in a [path](#path). For example, if $a$ and $b$ are [neighbours](#neighbour) then the [path](#path) $a,b$ has length $1$; if $a$ and $b$ are [neighbours](#neighbour) and $b$ and $c$ are [neighbours](#neighbour), then $a,b,c$ has length $2$, *etc.* Note that path length is a purely [topological](#topology) property, in the sense that it is agnostic to the physical position of [nodes](#node) in space or the physical distance between them; it is computed solely on the basis of the presence or absence of [edges](#edge) between [nodes](#node).

### Powerlaw relationship type
Method of determining the [distance](#distance-transform-d), [affinity](#affinity-transform-k), and [heterochronicity transforms](#heterochronicity-transform-h) from their respective matrices. Under the powerlaw relationship, the [distance transform](#distance-transform-d) is $d_{ij} = D^{\eta}_{ij}$, the [affinity transform](#affinity-transform-k) is $k_{ij} = K_{ij}^{\gamma}$, and the [heterochronicity transform](#heterochronicity-transform-h) is $h_{ij} = H_{ij}^{\lambda}$.

### Shortest path
For any two [nodes](#node), a shortest path between those [nodes](#node) is a [path](#path) which has smallest [path length](#path-length). For example, if $a$, $b$, and $c$ are all [connected](#connected) to one another, then both $a,c$ and $a,b,c$ are [paths](#path) between $a$ and $c$, but only $a,c$ would be a shortest path. Note that there can be multiple shortest paths between [nodes](#node). If two [nodes](#node) are unconnected, then the length of the shortest path between them is taken to be infinite.

### Small worldness
A [network](#network) has high small worldness when it displays high [clustering](#clustering-coefficient) but low [characteristic path length](#characteristic-path-length). To compute the small worldness of a binary [network](#network), we compute the (average) [clustering coefficient](#clustering-coefficient) of the [network](#network), $C$, and the [characteristic path length](#characteristic-path-length), $\ell$. We then construct a "control" [network](#network) which has the same number of [nodes](#node) and [edges](#edge), but the [edges](#edge) are placed at random, and compute the same quantities for the control [network](#network), $C_{\rm control}, \ell_{\rm control}$. The small worldness of the [network](#network) is then $\frac{C/C_{\rm control}}{\ell/\ell_{\rm control}}$.

### Structural connectivity
The pattern of connections made between regions of the brain by white matter tracts. *cf.* [functional connectivity](#functional-connectivity). 

### Symmetric
A matrix $M$ is symmetric when $M_{ij} = M_{ji}$ for all pairs of [nodes](#node) $i$ and $j$. [Distance](#distance-matrix-d), [affinity](#affinity-matrix-k), and [heterochronicity matrices](#heterochronicity-matrix-h) (and their [transforms](#transform)) are all symmetric matrices.

### Topology
Topological properties of a [graph](#graph) are those properties that depend only on the pattern of connectivity within the [graph](#graph), but not on the location of [nodes](#node) within physical space. *cf.* [topographical properties](#topography).

### Topography
Topographical properties of a [graph](#graph) are those properties which depend on the physical position of [nodes](#node) within space and the spatial relationships between them. *cf.* [topological properties](#topology).

### Transform
A parametrised modification of an original matrix, denoted by a lowercase. For example, the exponential transform of the [distance matrix](#distance-matrix-d), $D_{ij}$, is $d_{ij} = \exp( \eta D_{ij})$, with $\eta$ parametrising the transform.

### Vertex
Synonym for [node](#node).

### Weighted Generative Network Model
Variant of the Generative Network Model which generates weighted [networks](#network).

### Weight matrix, $W$
Matrix representing a weighted [graph](#graph). $W_{ij}$ gives the strength of the connection between $i$ and $j$, with $W_{ij} = 0$ if no connection is present between $i$ and $j$.

### Wiring probabilities, $P_{ij}$
The probability that, at a particular time step of the [binary GNM](#binary-generative-network-model), two [nodes](#node) connect to one another. Wiring probabilities are obtained by first multiplying the [distance](#distance-transform-d) and [affinity transforms](#affinity-transform-k) to obtain unnormalised probabilities, $\tilde{P}_{ij} = d_{ij} \times k_{ij}$. We then set the probability of existing connections and self-connections to zero: $\tilde{P}_{ij} \gets 0$ if $A_{ij} = 1$, and $\tilde{P}_{ii} \gets 0$. The unnormalised probabilities are then divided by their sum to get a probability distribution over all potential new connections, $P_{ij} = \frac{ \tilde{P}_{ij} }{ \sum_{ab} \tilde{P}_{ab} }$.