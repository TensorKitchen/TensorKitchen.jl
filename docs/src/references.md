# References

#### General Tensor Decomposition

- **Tensor decompositions (CP, Tucker):** T. G. Kolda and B. W. Bader, "Tensor decompositions and applications," *SIAM Review*, vol. 51, no. 3, pp. 455–500, 2009.

#### Tucker Methods

- **HOSVD:** L. De Lathauwer, B. De Moor, and J. Vandewalle, "A multilinear singular value decomposition," *SIAM J. Matrix Anal. Appl.*, vol. 21, no. 4, pp. 1253–1278, 2000.
- **ST-HOSVD:** N. Vannieuwenhoven, R. Vandebril, K. Meerbergen, "A new truncation strategy for the higher-order singular value decomposition," *SIAM J. Sci. Comput.*, vol. 34, no. 2, pp. A1027–A1052, 2012.
- **HOOI:** L. De Lathauwer, B. De Moor, and J. Vandewalle, "On the best rank-1 and rank-(R_1,R_2,...,R_N) approximation of higher-order tensors," *SIAM J. Matrix Anal. Appl.*, vol. 21, no. 4, pp. 1324–1342, 2000.

#### Block and Structured Models (BTD / LL1)

- **Block-term decomposition (BTD):** L. De Lathauwer, "Decompositions of a higher-order tensor in block terms—Part I: Lemmas for partitioned matrices," *SIAM J. Matrix Anal. Appl.*, vol. 30, no. 3, pp. 1022–1032, 2008.
- L. De Lathauwer, "Decompositions of a higher-order tensor in block terms—Part II: Definitions and uniqueness," *SIAM J. Matrix Anal. Appl.*, vol. 30, no. 3, pp. 1033–1066, 2008.
- **BTD-ALS:** L. De Lathauwer and D. Nion, "Decompositions of a higher-order tensor in block terms—Part III: Alternating least squares algorithms," *SIAM Journal on Matrix Analysis and Applications*, vol. 30, no. 3, pp. 1067–1083, 2008. [PDF](http://dimitri.nion.free.fr/Publications/Revues/DeLatNion_TensorBlock3.pdf).

#### Join decompositions

- **Conditioning of join decompositions:** P. Breiding and N. Vannieuwenhoven, "The condition number of join decompositions," *SIAM Journal on Matrix Analysis and Applications*, vol. 39, no. 1, pp. 287–309, 2018. [arXiv:1611.08117 (PDF)](https://arxiv.org/pdf/1611.08117).

#### Riemannian Optimization and Julia Ecosystem

- **Riemannian trust-region / Gauss–Newton for canonical rank (CP) approximation:** P. Breiding and N. Vannieuwenhoven, "A Riemannian Trust Region Method for the Canonical Tensor Rank Approximation Problem," *SIAM Journal on Optimization*, vol. 28, no. 3, pp. 2435–2465, 2018. [arXiv:1709.00033 (PDF)](https://arxiv.org/pdf/1709.00033).
- **Riemannian optimization:** P.-A. Absil, R. Mahony, and R. Sepulchre, *Optimization Algorithms on Matrix Manifolds*. Princeton University Press, 2008.
- **Julia manifold optimization ecosystem:** R. Bergmann *et al.*, [ManifoldsBase.jl](https://github.com/JuliaManifolds/ManifoldsBase.jl), [Manifolds.jl](https://github.com/JuliaManifolds/Manifolds.jl), and [Manopt.jl](https://manoptjl.org).