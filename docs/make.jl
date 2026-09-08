using Documenter, TensorKitchen

makedocs(
    sitename = "TensorKitchen.jl",
    format = Documenter.HTML(edit_link = "main"),
    warnonly = false,
    pages = [
        "Home" => "index.md",
        "CPD" => "cpd.md",
        "Tucker" => "tucker.md",
        "BTD" => "btd.md",
        "Join" => "join.md",
        "Utilities" => "utils.md",
        "Choosing a decomposition" => "PIPELINE.md",
        "Advanced guide" => [
            "Overview" => "advanced/index.md",
            "CPD methods" => "advanced/cpd.md",
            "Tucker methods" => "advanced/tucker.md",
            "BTD methods" => "advanced/btd.md",
            "Optimization methods" => "advanced/optimization.md",
            "Join models" => "advanced/join.md",
        ],
        "References" => "references.md",
    ],
)
