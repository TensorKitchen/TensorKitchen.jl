# User-facing convenience API: tucker, save_result, load_result
export save_result, load_result


include("api/approx.jl")
include("api/cpd.jl")
include("api/nncpd.jl")
include("api/btd.jl")
include("api/tucker.jl")


function save_result(path::AbstractString, result)
    open(path, "w") do io
        Base.Serialization.serialize(io, result)
    end
    return path
end

function load_result(path::AbstractString)
    open(path, "r") do io
        return Base.Serialization.deserialize(io)
    end
end
