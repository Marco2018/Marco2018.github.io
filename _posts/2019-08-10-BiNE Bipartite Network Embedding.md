### BiNE: Bipartite Network Embedding

##### Related work

matrix factorization methods and neural network-based methods

**In MF methods**:

SVD, kernel PCA, spectral embeddings

most MF methods drawbacks: 1.computationally expensive, 2. sensitive to the proximity method.

**Neural network method**:

Deepwalk and Node2vec extend the idea of Skip-gram to model homogeneous network.

Add 1st-order and 2nd-order proximities to embed node: LINE, SDNE, GraRep



BINE Problem

$G=(U,V,E)$ 其中U,V分别表示不同的两类vertices

**Model**

joint probability:

$P(i,j)=\frac{w_{i,j}}{\sum_{e_{i,j}\in E}w_{ij}}$, $w_{ij}$ is the weight of edge $e_{ij}$

$\widehat{P}(i,j)=\frac{1}{+exp({-\vec{u_i}^T\vec{v_j})}}$ 

Minimize 

$O_1=KL(P||\widehat{P})=\sum_{e_{ij}\in E} P(i,j) log(\frac{P(i,j)}{\widehat{P}(i,j)})$



**Constructing Corpus of Vertex Sequence** 

If directly preform random walks on the network, could fail.

Consider performing random walks on two homogeneous networks that contain 2nd-order proximities between vertices of the same type.

the 2nd-order proximity between two vertices:

$w_{ij}^U=\sum_{k\in V}w_{ik}w_{jk}$

$w_{ij}^V=\sum_{k\in U}w_{ik}w_{jk}$