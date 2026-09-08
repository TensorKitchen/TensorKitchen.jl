# Saving and Loading Results

Use `save_result` to store a fitted decomposition and `load_result` to restore
it in a later Julia session.

```julia
using TensorKitchen

A = randn(20, 15, 10)
result = tucker(A, (5, 4, 3))

save_result("tucker_result.jls", result)
loaded = load_result("tucker_result.jls")

A_approx = reconstruct(loaded)
```

`save_result` takes a file path and any TensorKitchen result. It returns the
path that was written. `load_result` takes that path and returns the saved
Julia object. If the path already exists, `save_result` overwrites it.

Only the object passed to `save_result` is stored. The original input tensor,
preprocessing choices, selected ranks, and other experiment metadata are not
included automatically. Store a small experiment record when those details are
needed for reproducibility:

```julia
record = (
    result = result,
    method = :tucker,
    ranks = (5, 4, 3),
    input_size = size(A),
    preprocessing = :none,
    julia_version = string(VERSION),
    tensorkitchen_version = string(pkgversion(TensorKitchen)),
)

save_result("tucker_experiment.jls", record)
loaded_record = load_result("tucker_experiment.jls")

loaded_result = loaded_record.result
```

The file uses Julia's native binary serialization format. Use it for Julia
workflows rather than cross-language interchange, and only load files from
sources you trust. For long-lived experiment files, record the Julia and
TensorKitchen versions as shown above.

The saved decomposition is compact, but `reconstruct(loaded_result)` allocates
an approximation with the original tensor dimensions. For large tensors,
inspect `core`, `factors`, `blocks`, or `components` before requesting a dense
reconstruction.

## API reference

```@docs
save_result
load_result
```
