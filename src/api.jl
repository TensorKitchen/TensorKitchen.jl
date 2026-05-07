# User-facing convenience API: save_result, load_result
export save_result, load_result


include("api/approx.jl")
include("api/cpd.jl")
include("api/nncpd.jl")
include("api/btd.jl")
include("api/tucker.jl")


"""
    save_result(path::AbstractString, result)

Save a TensorKitchen result object to `path` for later use.

* This uses Julia's built-in `Serialization` format, so the file is a Julia-native binary file. 
* It is intended for saving results during local experiments, benchmarks, and development workflows.
* Common file extensions are `.jls`, `.julia`, `.bin`, or `.tkresult`; the extension is only a convention.

# Examples

```julia
A = randn(20, 15, 10)
res = cpd(A, 35)

save_result("cpd_rank35.jls", res)
```

* You can also save richer records by serializing a `NamedTuple` of the result and additional metadata.

```julia
record = (
    method = :cpd,
    rank = 35,
    input_size = size(A),
    result = res,
)
save_result("experiment_cpd_rank35.jls", record)
``` 
"""
function save_result(path::AbstractString, result)
    open(path, "w") do io
        Base.Serialization.serialize(io, result)
    end
    return path
end

"""
    load_result(path::AbstractString)

Load a previously saved TensorKitchen result or experiment record from `path` for later use.

* The file must be written with `save_result`, or otherwise created using Julia's `Serialization.serialize`.

# Examples

```julia
res = load_result("cpd_rank35.jls")

weights(res)
factors(res)
reconstruct(res)
```
If the saved object was an experiment record, you can access the stored fields:

```julia
record = load_result("experiment_cpd_rank35.jls")

record.method
record.rank
record.input_size
record.result
record.result.weights
record.result.factors
reconstruct(record.result)
```
"""
function load_result(path::AbstractString)
    open(path, "r") do io
        return Base.Serialization.deserialize(io)
    end
end
