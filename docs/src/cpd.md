
# CPD 

Here is how to approximate a tensor `A` by a CPD of rank `r`.

```julia-repl
julia> using TensorKitchen
julia> A = randn(20, 15, 10)
julia> r = 35
julia> res = cpd(A, r)
CPDResult{Float64}
  Order:        3
  Dimensions:   (20, 15, 10)
  Rank:         35
  Rel. error:   0.4359141301703327
```

Now, `res` contains a CP approximation of the 3-way tensor `A`,

```math
\hat A = \sum_{i=1}^r \lambda_i\, a_i \otimes b_i \otimes c_i
```
It approximates `A` with relative error about `0.436`.

We access the decomposition as follows.
```julia
λ = weights(res)
U = factors(res)
```
Here, `U` is a triple of matrices $(A,B,C)$, where the columns of $A$ are the $a_i$ and so on. These are called *factor matrices*.

We get the whole reconstructed tensor by 
```julia
Â = reconstruct(res)
```


## CPD Docs

```@docs
cpd
nncpd
CPDResult
RankOneTensor
weights(::CPDResult)
comp_weight(::CPDResult)
factors(::CPDResult)
reconstruct(::CPDResult)
reconstruct_cp_rank1
reconstruct_cpd_rankr
normalize_components!
normalize_components
```
