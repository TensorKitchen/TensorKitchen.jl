using Documenter, TensorKitchen

makedocs(
    sitename = "TensorKitchen.jl",
    warnonly = true,
    pages = [
        "Home" => "index.md",
        "CPD" => "cpd.md",
        "Tucker" => "tucker.md",
        "BTD" => "btd.md",
        "Join" => "join.md",
        "Utilities" => "utils.md",
        "Pipeline" => "PIPELINE.md",
        "References" => "references.md",
    ],
)
