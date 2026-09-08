# User-facing convenience API: save_result, load_result
export save_result, load_result

import Serialization


include("api/approx.jl")
include("api/cpd.jl")
include("api/nncpd.jl")
include("api/btd.jl")
include("api/tucker.jl")


"""
    save_result(path, result)

Save a TensorKitchen result or experiment record to a Julia binary file.
Only the object passed as `result` is stored; input data and preprocessing
metadata must be included explicitly when they are needed. An existing file at
`path` is overwritten. Returns `path` after the file is written.

```julia
result = tucker(randn(20, 15, 10), (5, 4, 3))
save_result("tucker_result.jls", result)
```
"""
function save_result(path::AbstractString, result)
    open(path, "w") do io
        Serialization.serialize(io, result)
    end
    return path
end

"""
    load_result(path)

Load an object previously written with [`save_result`](@ref). Only load files
from sources you trust. The file uses Julia's native serialization format and
is intended for Julia workflows rather than cross-language interchange.

```julia
result = load_result("tucker_result.jls")
A_approx = reconstruct(result)
```
"""
function load_result(path::AbstractString)
    open(path, "r") do io
        return Serialization.deserialize(io)
    end
end
