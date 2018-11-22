using Pkg
Pkg.add.([
          # tabular
          "DataFrames", "CSV", "Query"

          # JuliaPro
          "DataStructures", "LightGraphs", "Calculus", "DataFrames",
          "StatsBase", "Distributions", "HypothesisTests", "GLM",
          "JuMP", "Optim", "Roots", "ODBC", "JDBC",
          "Knet", "Clusterig", "DecisionTree", "PyCall",
          "JSON", "HDF5", "JLD", "QuantEcon",

          # JuliaPlots
          "Plots", "PyPlot", "PlotlyJS",

          # DSP
          "DSP", "Wavelets", "Deconvolution",

          # "MXNet"  # not supported yet

          # MISC
          "ProgressMeter",
         ])
