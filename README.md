<div align="center">
 
# A short introduction to the F-adjoint
<div align="left"> 
 


This project will give some  highlight on the notion of F-adjoint which has been recently introduced in the following arxiv preprint: "[Backpropagation and F-adjoint. arXiv preprint arXiv:2304.13820.](https://arxiv.org/abs/2304.13820)". For this purpose, we consider the simple case of a fully-connected deep multi-layer perceptron (MLP) composed of $L$ layers trained in a supervised setting.  We will denote such an architecture by $` A[N_0, \cdots, N_\ell,\cdots, N_L]`$ where $N_0$ is the size of the input layer, $N_\ell$ is the size of hidden layer $\ell$,
and $N_L$ is the size of the output layer; $L$ is defined as the depth of the  (MLP).

**Notations**

``` math
  \begin{array}{|l|l|}
  \hline\\
  Term & Description \\ \hline
 W^{\ell}\in\mathbb{R}^{N_{\ell}\times (N_{\ell-1}+1)} & \mathrm{Weight\ matrix\ of\ the\ layer}\ \ell\  \mathrm{with\ bias} \\ \hline
 W_\sharp^{\ell}\in\mathbb{R}^{N_{\ell}\times N_{\ell-1}} & \mathrm{Weight\ matrix\ of\ the\ layer}\ \ell\  \mathrm{without\ bias}  \\ \hline
 Y^{\ell}\in\mathbb{R}^{N_{\ell}}  & \mathrm{Preactivation\ vector\ at\ layer}\ {\ell}, Y^{\ell} = W^{\ell}X^{\ell-1} \\ \hline
 X^{\ell}\in\mathbb{R}^{N_{\ell}}\times\{1\} & \mathrm{Activition\ vector\ at\ the\ layer} \ {\ell} \  \mathrm{with\ bias}, X^{\ell} =\sigma^{\ell}(Y^{\ell})\\ \hline
 \sigma^\ell:\  \mathbb{R}^{N_{\ell}}\ni Y^{\ell}\longmapsto\sigma^{\ell}(Y^{\ell})\in\mathbb{R}^{N_{\ell}} & \mathrm{Coordinate-wise\ activition\ function\ of\ the\ layer}\ {\ell}\\ \hline
   \end{array}$$
```

# **Definitions**

We   shall recall and revise the  definition of the this notion and provide some straightforward properties and improvements of this adjoint-like representation.



## Definition of an $F$-propagation

Let $X^0\in\mathbb{R}^{N_0}$ be a given data, $\sigma$ be a coordinate-wise map from $\mathbb{R}^{N_\ell}$ into $\mathbb{R}^{N_{\ell}}$ and $W^{\ell}\in \mathbb{R}^{{N_{\ell}}\times{N_{\ell-1}}}$ for all ${1\leq \ell\leq L}$. We say that we have a two-step recursive F-propagation   $F$  through the (MLP) $A[N_0,\cdots, N_L]$ if   one has the following family of vectors
``` math
F(X^0):=\begin{Bmatrix}X^0, Y^{1},X^{1},\cdots,X^{L-1},Y^{L},X^{L}\end{Bmatrix}\  \mathrm{where}\  Y^\ell=W^{\ell}X^{\ell-1}, \ X^\ell=\sigma(Y^\ell),\ \ell=1,\cdots, L.
``` 
Before going further, let us point that in the above definition the prefix "F" stands for "Feed-forward".

As a consequense, one may rewrite the $F$-propagation algorithm as follows:

1. Require: $`X^0,W,\sigma `$
2. Ensure:  F-propagation  ($` F(X^0)`$) 

      Function: $F$-propagation ($F$)
    
      1.  $`F\leftarrow\{X^0\}`$ 
      2.  For $\ell=1$ to $L$:
                            
            $Y^\ell:= W^\ell X^{\ell-1}$
                 
            $X^\ell:=\sigma(Y^{\ell})$
            
            $F\leftarrow Y^\ell,X^\ell$
            
            End For
          
  Return $F$

## Definition of the $F$-adjoint of an $F$-propagation

Let $`X^0\in\mathbb{R}^{N_0}`$ be a given data and let  $`X^L_*\in\mathbb{R}^{N_L}`$ be a given vector.  We define the F-adjoint propagation  ${F}_{*}$, through the (MLP) $`A[N_0,\cdots, N_L]`$, associated to the F-propagation  $F(X^0)$  as follows
``` math
F_{*}(X^{0}, X^{L}_{*}):=\begin{Bmatrix} X^{L}_{*}, Y^{L}_{*}, X^{L-1}_{*},\cdots, X^{1}_{*},Y^{1}_{*}, X^{0}_{*} \end{Bmatrix}\  \mathrm{where}\  Y^\ell_{*}=X^{\ell}_{*}\odot {\sigma}'(Y^\ell), \ X^{\ell-1}_{*}=(W_\sharp^\ell)^\top Y^\ell_{*},\ \ell=L,\cdots, 1.
``` 

Also, one may write a similar algorithm for the $F$-adjoint propagation as:

1. Require: $`F(X^0),X^L_*,W_\sharp, \sigma' `$
2. Ensure:  $`F_*`$-propagation ($`F_*(X^0, X_*^L)`$)

      Function: $F_*$-propagation ($`F_*`$)
    
      1.  $`F_* \leftarrow \{X_*^L\}`$
      2.  For $\ell= L$ to $1$:

            $Y^\ell_* := X^\ell_*\odot\sigma'(Y^\ell)$

            $X^{\ell-1}:={(W_\sharp^\ell)}^\top Y^\ell_* $                          
                    
            $F_* \leftarrow Y^\ell_* $, $X_*^{\ell-1}$
            
            End For
          
  Return $F_*$



## Reference

<div id="refs" class="references">


Boughammoura, A. (2023). Backpropagation and F-adjoint. arXiv preprint arXiv:2304.13820.(https://arxiv.org/abs/2304.13820)

</div>
