using Documenter, TensorKitchen

makedocs(
    sitename = "TensorKitchen.jl",
    build = "build",
    format = Documenter.HTML(
        sidebar_sitename = false,
    ),
    pages = [
        "Home" => "index.md",
        "CPD" => "cpd.md",
        "Tucker" => "tucker.md",
        "BTD" => "btd.md",
        "Join" => "join.md",
        "Utilities" => "utils.md",
        "Pipeline" => "PIPELINE.md",
    ],
)