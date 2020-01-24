### Graph Neural Network && Graph Convolutional Networks

##### GNN

GNN常见用例，判断某网络结构（分子结构）的类别：例如用来判断分子是否为毒药

![](./pics/1.png)



GNN训练过程如下：

![](./pics/2.png)

每个节点的初始特征为one hot向量，属于某个原子则对应的下标为1



迭代训练T次：

![](./pics/3.png)

![](./pics/4.png)









##### GCN：Graph Convolutional Networks

[<http://tkipf.github.io/graph-convolutional-networks/>]()

paper:[semi-supervised classification with graph convolutional networks](https://openreview.net/pdf?id=SJU4ayYgl)



![](./pics/5.png)

##### 算法流程：

graph：**G=（V,E）**

N×D feature matrix **X** (N: number of nodes, D: number of input features)

A representative description of the graph structure in matrix form; typically in the form of an adjacency matrix **A** 

卷积层的计算：
$$
H^{(l+1)}=f(H^l,A)
$$
其中$H^{(0)}=X$ $H^{(L)}=Z$

Z为最终的输出



卷积简单例子：
$$
f(H^{(l)},A)=σ(AH^{(l)}W^{(l)})
$$
σ 表示激活函数类似于RELU

较为复杂的卷积例子， [Kipf & Welling](http://arxiv.org/abs/1609.02907) (ICLR 2017)


$$
f(H^{(l)},A)=σ(\hat{D}^{-\frac{1}{2}}\hat{A}\hat{D}^{-\frac{1}{2}}H^{(l)}W^{(l)})
$$


其中$\hat{A}=A+I$ 

$\hat{D}$ 是$\hat{A}$的对角阵



