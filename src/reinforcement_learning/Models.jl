module Models

using Reexport

include("DataManipulate.jl")
@reexport using .DataManipulate
#export transform_data!

# Reinforcement Learning
include("RLModels.jl")
@reexport using .RLModels

# Dynamic fitting
include("Evaluate/Fit.jl")
export evaluate_relation, fit_RL_base, fit_RL_detrend_miniblock
export model_recovery, model_evaluation
export fit_and_evaluate, fit_and_evaluate_miniblock

end # module
