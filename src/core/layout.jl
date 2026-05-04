# core/layout.jl — Point/tangent container layout helpers

@inline point_parts(p) = hasproperty(p, :x) ? p.x : p
@inline parts_tuple(p) = (_p = point_parts(p); _p isa Tuple ? _p : Tuple(_p))
@inline wrap_like_point(p, vals::Tuple) =
    hasproperty(p, :x) ? ArrayPartition(vals...) : vals
@inline _unwrap_part(x) = hasproperty(x, :x) ? x.x : x

# Pack canonical rank-r CP tangents in one place so the hot gradient paths do
# not each rebuild the same nested ArrayPartition layout by hand.
function wrap_rankr_canonical_tangent_like(p, grad_λ, gradU, r::Int)
    grad_modes = ntuple(m -> ntuple(k -> Vector(@view gradU[m][:, k]), r), length(gradU))
    if hasproperty(p, :x)
        mode_parts = ntuple(m -> ArrayPartition(grad_modes[m]...), length(grad_modes))
        return ArrayPartition(grad_λ, mode_parts...)
    end
    return (grad_λ, grad_modes...)
end
