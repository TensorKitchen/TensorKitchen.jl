# User-facing convenience API: tucker, save_result, load_result
export save_result, load_result


include("api/approx.jl")
include("api/cpd.jl")
include("api/nncpd.jl")
include("api/btd.jl")
include("api/tucker.jl")


"""
    save_result(path, result)

Serialize a result to disk (Julia Serialization).
"""
function save_result(path::AbstractString, result)
    open(path, "w") do io
        Base.Serialization.serialize(io, result)
    end
    return path
end

"""
    load_result(path)

Load a result serialized by `save_result`.
"""
function load_result(path::AbstractString)
    open(path, "r") do io
        return Base.Serialization.deserialize(io)
    end
end
