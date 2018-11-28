# julia.v0.7-jupyter

## included packages:
- `IJulia`
- tabular data: `DataFrames`, `CSV`, `Query`
- JuliaPro:
    - General programming: `DataStructures`, `LightGraphs`,
    - General Math: `Calculus`, `StatsBase`, `Distributions`, `HypothesisTests`, `GLM`,
    - Optimization: `JuMP`, `Optim`, `Roots`,
    - Machine Learning: `Knet`, `Clusterig`, `DecisionTree`,
    - Interaction with other languages: `PyCall`,
    - File and data formats: `JSON`, `HDF5`, `JLD`,
- Visualization: `Plots`, `PyPlot`, `PlotlyJS`,
- Databases: `ODBC`, `JDBC`, `SQLite`
- Misc: `ProgressMeter`

## install custom packages

create your own `install_julia_packages.jl` and
add `julia /userhome/install_julia_packages.jl` before
the `jupyter notebook` command.

`install_julia_packages.jl` file template:

```julia
using Pkg
Pkg.add.([
    "Package 1",
    "Package 2",
    ...
    ])
```
