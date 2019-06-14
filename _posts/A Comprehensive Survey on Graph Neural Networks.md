### A Comprehensive Survey on Graph Neural Networks

deep learning has achieved great success on Euclidean data, there is an increasing number of applications where data are generated from the non-Euclidean domain.

Graph data applications : 1.user, products recommendations, 2.citation network 3. social network 4.chemistry drug discovery

The difficulty of graph data: 

1. complexity of graph data (each graph has a variable size of unordered nodes and each nodes has a different number of neighbors, causing some opeartions like convolution can not be done directly.)

   ![](./Fig1.png)

   

2. a core assumption of machine learning method is that instances are independent of each other. However, graph nodes are not independent.



A brief history of GNN

Early studies learn a target node's representation **by  propagating neighbor information**.

However it is computationally expensive.



GNN is very close to graph embedding, which can be classified into three groups:

**Matrix Factorization, Random Walks, Deep Learning Approaches** 

Graph Neural Network can be classified into 

![](./Fig3.png)

**Graph Convolution Network** generalize the operation of convolution from traditional data to graph data. The key is to learn a function f to generate a node's representation by aggregating its own feature $X_i$ and its neighbors' feature $X_j$ 

Graph Convolution Network fall into two categories:**Spectral-based approaches（光谱 removing noise from graph signals） and Spatial-based approaches (空间 aggregating feature information from neighbors)** 

**Spectral-based approach** 

graph laplacian matrix
$$
L=I_n -D^{-\frac{1}{2}}AD^{-\frac{1}{2}}
$$
D is diagonal matrix of node degrees $D_{ii}=\sum _j (A_{i,j})$ 

矩阵L 具有实对称半正定性质，so we can have $L=U\Lambda U^T$

$U=[u_0,u_1,...,u_n] \in R^{N×N} $ is the matrix of eigenvectors ordered by eigenvalues. $\Lambda_{ii}=\lambda_i$

The graph Fourier transform to a signal x: $F(x)=U^Tx$  

The inverse graph Fourier transform to a signal $\hat{x}$: $F^{-1}(x)=U\hat{x}$    $\hat{x}$ is the result of graph Fourier transform of x

the convolution of the input signal x with a filter $g \in R^N$ 
$$
x * Gg =F^{-1}(F(x)\bigodot F(g))
$$
$\bigodot$ is Hadamard product

if we denote a filter $g_{\theta} = diag(U^T x)$ 
$$
x*g_{\theta}=Ug_{\theta}U^Tx
$$
the key difference lies in the choice of filter $g_{ \theta}$ 

这些的意义是基于傅里叶变换提出了卷积操作，
$$
X^{k+1}=\overline{A}X^k\Theta
$$
$\overline{A}=I_N+D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$ ,$\Theta$ is diagonal matrix filled with learnable parameters.

This 1stChebNet filter bridges the gap between spectral-based methods and spatial-based methods. Each row of the output represents the latent representation of node.

But spectral methods drawbacks:

1. any perturbation of graph will result in a different eigenbasis,  like new nodes coming in.
2. the learned filters can't be applied to a new graph.
3. eigen-decomposition requires $O(N^3)$ computation and $O(N^2)$  memory.

**Spatial-based Graph** 

**GNNs** recursively update node latent representations until convergence.
$$
h_v^t = f(l_v,l_{co}[v]，h^{t-1}_{ne}[v],l_{ne}[v])
$$
l denotes label of a node.

**Gated Graph Neural Networks**
$$
h_v^t = GRU(h_v^{t-1},\sum_{u \in N(v)}Wh_u^t)
$$
GGNN uses back propagation to learn a better parameters. 

**GraphSage**
$$
h_v^t=\sigma(W^t·aggregate(h_v^{t-1},\{h_u^{t-1},u \in N(v)\}))
$$
with t=2 GraphSage already achieves high performance.

![](./Fig8.png)



**Graph Pooling Modules** 
$$
h_G=mean/max/sum(h_1^T,h_2^T,...,h_n^T)
$$
In terms of efficiency, the spectral-based model increases dramatically with the graph size.

In terms of generality, spectral-based models assumed a fixed graph, making them generalize poorly to new or different graphs.

In terms of flexibility, spectral-based models limited to work on undirected graphs.



**Graph Attention Networks** are similar to GCNs. The key difference is that graph attention networks employ attention mechanisms which **assign larger weights to more important nodes.**

![](./Fig6.png)



Attention 机制是一系列注意力分配的过程，也就是一系列权重系数。

#### Attention is all you need

$$
y_t=f(x_t,A,B)
$$

当A=B时，即为self attention
$$
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

$$
Q \in R^{n×d_k}, K \in R^{m×d_k},V \in R^{m×d_v}
$$

Attention层相当于将n×dk的序列Q编码成了n×dv的序列。



##### Multi-Head Attention

将Q,K,V通过参数矩阵进行映射然后再做Attention，把这个过程重复h次，将结果拼接起来。
$$
head_i=Attention(QW_i^Q,KW_i^K,VW_i^V)
$$

$$
W_i^Q \in R^{d_k×\overline{d_k}}, W_i^K \in R^{d_k×\overline{d_k}}  W_i^V \in R^{d_k×\overline{d_k}}
$$

$$
MultiHead(Q,K,V)=Concat(head_1,...,head_n)
$$

得到一个n×$(h_{\overline{d_v}})$ 的序列，就是多做几遍同样的事情，（参数并不共享），将结果拼接。

**Graph Attention Network(GAT)** 
$$
h_i^t=\sigma(\sum_{j\in N_i}\alpha(h_i^{t-1},h_j^{t-1})W^{t-1}h_j^{t-1})
$$
$\alpha()$ is an attention function which adaptively controls the contribution of a neighbor j to the node i. GAT uses multihead.
$$
h_i^t=||_{k=1}^K\sigma(\sum_{j\in N_i}\alpha_k(h_i^{t-1},h_j^{t-1})W^{t-1}h_j^{t-1})
$$
**Gated Attention Network GAAN**

比GAN多了一个对head的权重系数
$$
h_i^t=\phi_o(x_i\bigoplus ||_{k=1}^K g_i^k\sum_{j\in N_i}\alpha_k(h_i^{t-1},h_j^{t-1})\phi_v( h_j^{t-1}))
$$
$\phi_o(),\phi_v()$ is denote feedforward neural networks and $g_i^k$ is the attention weight of the $k^{th}$ attention head.



克罗内克乘积 Kronecker$\bigotimes$

![](./krone1.png)

![](./krone2.png)



##### Hadamard product

$$
\left[
 \begin{matrix}
   a_{11} & a_{12} & a_{13} \\
   a_{21} & a_{22} & a_{23} \\
   a_{31} & a_{32} & a_{33}
  \end{matrix}
  \right] \bigodot \left[
 \begin{matrix}
   b_{11} & b_{12} & b_{13} \\
   b_{21} & b_{22} & b_{23} \\
   b_{31} & b_{32} & b_{33}
  \end{matrix}
  \right] = \left[
 \begin{matrix}
   a_{11}b_{11} & a_{12}b_{12} & a_{13}b_{13} \\
   a_{21}b_{21} & a_{22}b_{22} & a_{23}b_{23} \\
   a_{31}b_{31} & a_{32}b_{32} & a_{33}b_{33}
  \end{matrix}
  \right]
$$





**Graph Auto-encoders** 

**Graph  Auto-encoder(GAE)**

the encoder is defined as:
$$
Z=GCN(X,A)
$$


decoder is:
$$
\overline{A}=\sigma(ZZ^T)
$$
Graph Auto-encoder is to encode graph into low dimensional vector.

DNGR and SDNE learn node embeddings only given the topological structures, while GAE, ARGA, NetRA, DRNE, learn node embeddings when both topological information and node content features are available. 

One challenge of graph auto-encoders is the sparsity if adjacency matrix A.

To tackle this issue,DNGR reconstructs a denser matrix namely the PPMI matrix, SDNE imposes a penalty to zero entries of the adjacency matrix, GAE reweighs the terms in the adjacency matrix, and NetRA linearizes Graphs into sequences.

**DNGR** 

learn a PPMI matrix:
$$
PPMI_{v_1,v_2}=max(log(\frac{count(v_1,v_2)·|D|}{count(v_1)count(v_2)}),0)
$$
$|D|=\sum_{v_1,v_2}count(v_1,v_2)$ , and $v_1,v_2 \in V$

##### SDNE

The goal for the first-order proximity is to drive representation of adjacent nodes close to each other as much as possible.
$$
L_{1st}=\sum_{i,j=1}^n A_{i,j}||h_i^{(k)}-h_j{(k)}||^2
$$
The goal for the second-order proximity is to preserve a node's neighborhood information.
$$
L_{2nd}=\sum_{i=1}^n||(\hat{x}_i-x_i) \bigodot b_i||^2
$$


**Graph Generative Networks** 

![](./GGN.png)

The goal of graph generative networks is to generate graphs given an observed set of graphs. 

 For instance, in molecular graph generation, some works model a string representation of molecular graphs called SMILES. In natural language processing, generating a semantic or a knowledge graph is often conditioned on a given sentence.



**Graph Spatial-temporal Networks** 

Graph spatial-temporal networks capture spatial and temporal dependencies of a spatial-temporal graph simultaneously. Spatial-temporal graphs have a global graph structure with inputs to each node which are changing across time.



##### Future Directions:

1. Go Deep

2. Receptive Field:  The receptive field of a node refers to a set of nodes including the central node and its neighbors.

3. Scalability :Most GNN methods do not scale well for large graphs

4. Dynamics and Heterogeneity: New node may enter into a network at any time, a node may exits at any time. In recommender system, products may have different types where their inputs may have different forms such as texts or images.

    