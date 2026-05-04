# Join Decomposition 

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

## Join Decomposition Docs

```@docs
approx
ApproxResult
reconstruct(::ApproxResult)
join_product
SegreProduct
SqEuclidean
pullback_metric_inverse
sm_cost_nn_quadratic
sm_egrad_nn_quadratic
sm_cost_nn_rank1
sm_egrad_nn_rank1
sm_cost_double_circle
sm_egrad_double_circle
sm_double_circle_at_zero
egrad_to_rgrad
egrad_to_rgrad!
```
