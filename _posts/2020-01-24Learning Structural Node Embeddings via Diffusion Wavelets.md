### Learning Structural Node Embeddings via Diffusion Wavelets



**GraphWave** is a scalable unsupervised method for learning node embeddings based on structural similarity in networks. 



![](./pics/wwavelet.png)



Graph Laplacian $L=D-A=U\Lambda U^T$ 

U be the eigenvector decomposition of the unnormalized graph Laplacian $L$  

let $\lambda_1<\lambda_2\leq \lambda_3 \leq ...\leq\lambda_N$ $\Lambda=Diag(\lambda_1,...,\lambda_N)$  denote the eigenvalues of L, let $g_s$be the filter kernel, we use heat kernel $g_s(\lambda)=e^{-\lambda s}$ , we need to find out the proper value of s

The spectral graph wavelet $Ψa$  is given by an N-dimensional vector:
$Ψ_a = U Diag(g_s (λ_1), . . . ,g_s (λ_N ))U^T δ_a$ 
where $δ_a$ is the one-hot vector for node a.



##### GraphWave Algorithm

$Ψ$ is a $N×N$ matrix, a-th column vector is the spectral graph wavelet for a heat kernel centered at node a.

$Ψ_{ma}$ represents the amount of energy that node a has received from node m.

We embed spectral graph wavelet coefficient distributions into 2d-dimensional space by calculating chracteristic function for each node's coefficient $Ψ_{a}$.

The characteristic function of a probability distribution X is defined as: $ϕ_X (t ) = E[e^{itX} ], t ∈ R. $

For a given node a and scale s, the empiricaal characteristic function of $Ψ_{a}$ is defined as:
$$
ϕ_a（t）=\frac{1}{N}\sum_{m=1}^Ne^{itΨ_{ma}}
$$
Finally, structural embedding vector $\chi_a$ of node a is obtained by sampling the -dimensional parametric function at d evenly spaced points $t_1, t_2,...,t_d$ 
$$
\chi_a=[Re(\phi_a(t_i)),Im(\phi_aa(t_i))]_{t_1,...,t_d}
$$
![](wwavelet1.png)

##### Distance between structural embeddings

$dist(a,b)=||\chi_a-\chi_b||_2$ 

complexity of $O(K|E|)$,  where K denotes the order Chebyshev polynomial approximation. The overall complexity of GraphWave is linear in the number of edges