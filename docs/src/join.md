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

## Generic Join Approximation

`approx(...)` is the main frontend for join decomposition. It works in two stages:

1. build an initial point
2. refine it with the selected solver

### Supported Forms

- `approx(model; kwargs...)`
  Solve an already constructed `JoinModel`.
- `approx(manifolds, target; kwargs...)`
  Build a join from a tuple/vector of component manifolds.
- `approx(M::ProductManifold, target; kwargs...)`
  Use the product-manifold factors as join components.
- `approx(base, r, target; kwargs...)`
  Repeat a single base manifold `r` times to build a join.
- `approx(base, target; kwargs...)`
  Build a single-component join.

Examples:

```julia-repl
julia> target = [1.2, 0.4, -0.3]
julia> model = JoinModel(Manifolds.Sphere(2), target)
julia> approx(model; maxiter = 50, verbose = false)
ApproxResult{Float64}
  Components:   1
  Rel. error:   0.23076923076923075
```

```julia-repl
julia> target = randn(2, 3)
julia> approx((Manifolds.Segre((2, 3)), Manifolds.Segre((2, 3))), target; verbose = false)
CPDResult{Float64}
```

```julia-repl
julia> target = [1.2, 0.4, -0.3]
julia> approx(Manifolds.Sphere(2), 2, target; verbose = false)
ApproxResult{Float64}
```

### Routing

With `dispatch = :auto`:

- uniform `Manifolds.Segre` components route to `cpd(...)`
- uniform `Manifolds.Tucker` components route to `btd(...)`
- otherwise the generic `JoinModel(...)` path is used and `ApproxResult` is returned

You can also force the route explicitly:

- `dispatch = :generic`
- `dispatch = :cpd`
- `dispatch = :btd`

Forced routing is validated:

- `dispatch = :cpd` requires all components to be `Manifolds.Segre` with identical `factor_dims`
- `dispatch = :btd` requires all components to be `Manifolds.Tucker` with identical `factor_dims` and compatible multilinear rank

### Generic Join Options

For the generic join path:

- `init = :random`
  Default initializer. Built-in initializer support depends on the component manifolds.
- `solver = :rgd`
  Supported generic-join solver options are:
  - `:rgd`
  - `:rgd_fixed`
  - `:rcg`
  - `:lbfgs`

Other common options:

- `p0 = nothing`
- `maxiter = 500`
- `stepsize = 1.0`
- `tol = 1e-6`
- `gradient_mode = :riemannian`
- `verbose = true`
- `vector_transport_method = nothing`

Built-in `init` support by component family:

- `Sphere`: `:random`, `:deterministic`, `:target`
- `Segre`: `:random`, `:deterministic`
- `Tucker`: `:random`, `:tucker`, `:tucker_diag`, `:sthosvd`
- other manifolds: `:random`

### Notes

- Generic joins require every component manifold to embed into the same flattened ambient length as `target`.
- `solver = :als` is not available for a truly generic `JoinModel`. ALS may still be available when `approx(...)` auto-routes to `cpd(...)` or `btd(...)`.

## Join Decomposition Docs

```@docs
approx
ApproxResult
reconstruct(::ApproxResult)
join_product
SegreProduct

```
