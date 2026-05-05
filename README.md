
[![][docs-img]][docs-url] 

# TensorKitchen.jl


<img src="docs/logo.png" width = "450px">

<br>

**TensorKitchen.jl** is a Julia package for tensor decompositions.

---

The package is currently at a pre-alpha stage. 

The implementation is based on combining algebraic algorithms like ALS (see, e.g., the [textbook by Kolda and Ballard](https://users.wfu.edu/ballard/pdfs/tensor_textbook.pdf)) and Riemannian optimization from [Manopt.jl](https://manoptjl.org/stable/).

What currently works is 

- Canonical Polyadic Decomposition (CPD)
- Tucker Decomposition
- Nonnegative Canonical Polyadic Decomposition (NNCPD)
- Block Term Decomposition (BTD)
- Join Decompositions

See [PIPELINE.md](docs/src/PIPELINE.md) for the current execution flow.

---

The next updates will include 

- Documentation
- Improved User Interface
- ProgressMeter
- GPU Support 
- LL1 Decomposition (3-way specialized BTD)
- Symmetric CP / Waring Decomposition
- Partially Symmetric CP
- Tensor Trains


<br>

## Canonical Polyadic Decomposition (CPD)

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

<br>

## Tucker Decomposition

Approximating `A` by a Tucker decomposition
```math
\hat A = C \times_1 U \times_2 V \times_3 W
```
with multilinear rank `mlrank` can be computed as follows.

```julia-repl
julia> mlrank = (5, 4, 3)
julia> tucker_res = tucker(A, mlrank)
TuckerResult{Float64, 3}
  Original size:    (20, 15, 10)
  Core size:        (5, 4, 3)
  Multilinear rank: (5, 4, 3)
  Compression:      12.0x
```

The core $C$ and the factor matrices $(U, V, W)$ of the decomposition can be accessed as follows.

```julia
core(tucker_res)
factors(tucker_res)
```

<br>

## Nonnegative Canonical Polyadic Decomposition (NNCPD)

To compute a nonnegative CP decomposition, either set `nonnegative = true` in `cpd`

```julia-repl 
julia> B = abs.(A)
julia> cpd(B, r; nonnegative = true)
CPDResult{Float64}
  Order:        3
  Dimensions:   (20, 15, 10)
  Rank:         35
  Rel. error:   0.3765605093526155
``` 

or run 

```julia
nncpd(B, r)
```

We access the decomposition as follows.

```julia
weights(nn_res)
factors(nn_res)
```

<br>

## Block Term Decomposition (BTD)

A block term decomposition (BTD) with `r` blocks writes
```math
\hat A = \sum_{i=1}^r A_i,
```
where each block $A_i$ is represented as a Tucker decomposition. At present, only homogeneous BTDs are supported, that is, all blocks must have the same multilinear rank.

To compute a block term decomposition of A with 10 blocks, each of multilinear rank (5, 4, 3), use

```julia-repl
julia> r = 10
julia> mlrank = (5, 4, 3)
julia> btd_res = btd(A, r, mlrank)
BTDResult{Float64}
  Blocks:       10
  Rel. error:   0.2551559591470521
```

The blocks of `btd_res` can be obtained as follows:

```julia
blocks = blocks(btd_res)
```

Each block is represented as a Tucker decomposition, so we can access its core and factor matrices via:

```julia
blk = blocks[1]
core(blk)
factors(blk)
```

<br>

## Join Decompositions

A *Join Decomposition* of a vector $x\in\mathbb R^N$ is a decomposition of the form $x = x_1+\cdots+x_r$, where $x_i\in M_i$ and $M_i\subset \mathbb R^N$ is a given embedded manifold. 

For instance, we can approximate a point $p = (1.2, 0.4)\in\mathbb R^2$ by $x=x_1+x_2$, where $x_1,x_2\in S^1$ are points on the circle:

```julia-repl 
julia> using Manifolds
julia> p = [1.2, 0.4]
julia> S = Sphere(1)
julia> join_res = approx((S, S), p)
ApproxResult{Float64}
  Components:   2
  Rel. error:   0.0007114699550529457
```

We access the decomposition as follows.

```julia
components(join_res)
reconstruct(join_res)
```
<br>





[docs-img]: https://img.shields.io/badge/docs-online-blue.svg
[docs-url]: https://tensorkitchen.github.io/TensorKitchen.jl/