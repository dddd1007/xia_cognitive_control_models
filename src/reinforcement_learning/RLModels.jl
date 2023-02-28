#=
# Reinforcement Learning Models for Lingwang's lab

Author: Xiaokai Xia (xia@xiaokai.me)
Date: 2020-09-28
Version: 0.0.009

This model try to using reinforcement learning and Softmax to simulate the learning process in
Simon Tasks.

There are two models with four different methods:

## Two models

1. Reinforcement learning are used to learning the value of each rules
2. Softmax are used to select action by the value which RL learnt

## Four methods

- What to learn
    - Abstract concepts (Con, Inc)
    - S-R association
- The information attenuation
    - Have a Decay
    - Without Decay
- How to treat the error trials
    - Have a CCC to change action
    - Without CCC
- How to react to error trials
    - Change RL model's learning rate
    - Change Softmax model's Q-value
    - Both
=#

#============================================================================
# Module0: Basic calculate elements of RLModels                             #
============================================================================#
module RLModels

using GLM, DataFrames, StatsBase

export ExpEnv, RealSub, RLLearner
export init_env_sub, evaluate_relation
export update_options_weight_matrix, init_param
export calc_CCC

#####
##### 定义类型系统
#####

"""
    ExpEnv

The **experiment environment** which the RLLearner will to learn.
"""
struct ExpEnv
    stim_task_related::Array{Int64,1}
    stim_task_unrelated::Array{Int64,1}
    stim_correct_action::Array{Int64,1}
    stim_action_congruency::Array{Int64,1}
    env_type::Array{String,1}
    sub_tag::Array{String,1}
end

"""
    RealSub

All of the actions the **real subject** have done.
"""
struct RealSub
    response::Array{Int64,1}
    RT::Array{Float64,1}
    corrections::Array{Int64,1}
    sub_tag::Array{String,1}
    prop_seq::Array{Int64,1}
end

# 环境中的学习者
abstract type RLLearner end

#####
##### 定义函数
#####

"""
init_env_sub(
    transformed_data::DataFrame,
    env_idx_dict::Dict,
    sub_idx_dict::Dict
)

Init the env and subject objects for simulation.

# Examples
```julia
# Define the trasnform rule
begin
    env_idx_dict = Dict("stim_task_related" => "color", "stim_task_unrelated" => "location",
		                "stim_action_congruency" => "contigency",
		                "env_type" => "condition", "sub_tag" => "Subject")
	sub_idx_dict = Dict("response" => "Response", "RT" => "RT",
		                "corrections" => "Type", "sub_tag" => "Subject")
end
# Excute the transform
env, sub = init_env_realsub(transformed_data, env_idx_dict, sub_idx_dict, task_rule)
```
"""
function init_env_sub(transformed_data::DataFrame, env_idx_dict::Dict, sub_idx_dict::Dict)
    exp_env = ExpEnv(transformed_data[!, env_idx_dict["stim_task_related"]],
                     transformed_data[!, env_idx_dict["stim_task_unrelated"]],
                     transformed_data[!, env_idx_dict["correct_action"]],
                     transformed_data[!, env_idx_dict["stim_action_congruency"]],
                     transformed_data[!, env_idx_dict["env_type"]],
                     transformed_data[!, env_idx_dict["sub_tag"]])
    real_sub = RealSub(# Because of the miss action, we need the tryparse()
                       # to parse "miss" to "nothing"
                       tryparse.(Float64, transformed_data[!, sub_idx_dict["response"]]),
                       tryparse.(Float64, transformed_data[!, sub_idx_dict["RT"]]),
                       transformed_data[!, sub_idx_dict["corrections"]],
                       transformed_data[!, sub_idx_dict["sub_tag"]],
                       transformed_data[!, sub_idx_dict["prop_seq"]])
    println("The env and sub info of " *
            transformed_data[!, env_idx_dict["sub_tag"]][1] *
            " is generated!")

    return (exp_env, real_sub)
end

# 初始化更新价值矩阵和基本参数
function init_param(env, learn_type)
    total_trials_num = length(env.stim_task_unrelated)

    if learn_type == :sr
        options_weight_matrix = zeros(Float64, (total_trials_num + 1, 4))
        options_weight_matrix[1, :] = [0.5, 0.5, 0.5, 0.5]
    elseif learn_type == :ab
        options_weight_matrix = zeros(Float64, (total_trials_num + 1, 2))
        options_weight_matrix[1, :] = [0.5, 0.5]
    end

    p_softmax_history = zeros(Float64, total_trials_num)
    PE_history = zeros(Float64, total_trials_num)

    return (total_trials_num, options_weight_matrix, p_softmax_history, PE_history)
end

#### 定义工具性的计算函数

# 定义更新价值矩阵的函数

# 具体SR联结学习的价值更新函数
function update_options_weight_matrix(weight_vector::Array{Float64,1}, α::Float64,
    decay::Float64, sub_selection::Tuple; dodecay=true)

    # Convert the vector to weight matrix which easy to update
    weight_matrix = reshape(weight_vector, 2, 2)'

    # Get each idx of options
    sub_selection_idx = CartesianIndex(sub_selection) + CartesianIndex(1, 1)
    sub_unselection_idx = CartesianIndex(sub_selection[1], abs(sub_selection[2] - 1)) + CartesianIndex(1, 1)
    op_stim_loc = abs(sub_selection[1] - 1) + 1

    selection_value = weight_matrix[sub_selection_idx] +
         α * (1 - weight_matrix[sub_selection_idx])
    unselection_value = 1 - selection_value

    weight_matrix[sub_selection_idx] = selection_value
    weight_matrix[sub_unselection_idx] = unselection_value

    if dodecay
        weight_matrix[op_stim_loc, :] = weight_matrix[op_stim_loc, :] .+
           decay .*
           (0.5 .- weight_matrix[op_stim_loc, :])
    end

    return reshape(weight_matrix', 1, 4)
end

# 抽象概念的价值更新函数
function update_options_weight_matrix(weight_vector::Array{Float64,1}, α::Float64,
                                      correct_selection::Int; doreduce=true, debug=false)
    correct_selection_idx = correct_selection + 1
    op_selection_idx = 2 - correct_selection

    if debug
        println("True selection is " * repr(correct_selection_idx))
        println("The value is " * repr(weight_vector[correct_selection_idx]))
    end

    weight_vector[correct_selection_idx] = weight_vector[correct_selection_idx] +
                                           α * (1 - weight_vector[correct_selection_idx])

    if doreduce
        weight_vector[op_selection_idx] = 1 - weight_vector[correct_selection_idx]
    end

    return weight_vector
end

# 定义计算冲突程度的函数
function calc_CCC(weight_vector::Array{Float64,1}, stim_loc_sub_selection::Tuple)
    weight_matrix = reshape(weight_vector, 2, 2)'

    sub_selection_idx = CartesianIndex(stim_loc_sub_selection) + CartesianIndex(1, 1)

    return CCC = 2 * weight_matrix[sub_selection_idx] - 1
end

function calc_CCC(weight_vector::Array{Float64,1}, sub_action::Int)
    return CCC = 2 * weight_vector[sub_action] - 1
end

#============================================================================
# Module1: RLModels with Softmax                                            #
============================================================================#
# module WithSoftMax

# using ..RLModels

# #####
# ##### 定义类型系统
# #####

# # 环境中的学习者, 在基本条件下
# """
#     RLLearner_basic

# A RLLearner which learnt parameters from the experiment environment.
# """
# struct RLLearner_basic <: RLLearner
#     α_v::Float64
#     β_v::Float64
#     α_s::Float64
#     β_s::Float64
#     decay::Any
# end

# # 环境中的学习者, 在错误试次下学习率不同
# struct RLLearner_witherror <: RLLearner
#     α_v::Float64
#     β_v::Float64
#     α_s::Float64
#     β_s::Float64

#     α_v_error::Float64
#     β_v_error::Float64
#     α_s_error::Float64
#     β_s_error::Float64

#     decay::Any
# end

# # 存在冲突控制的学习者
# struct RLLearner_withCCC <: RLLearner
#     α_v::Float64
#     β_v::Float64
#     α_s::Float64
#     β_s::Float64

#     α_v_error::Float64
#     β_v_error::Float64
#     α_s_error::Float64
#     β_s_error::Float64

#     α_v_CCC::Float64
#     β_v_CCC::Float64
#     α_s_CCC::Float64
#     β_s_CCC::Float64

#     CCC::Float64
#     decay::Any
# end

# struct RLLearner_withCCC_no_error <: RLLearner
#     α_v::Float64
#     β_v::Float64
#     α_s::Float64
#     β_s::Float64

#     α_v_CCC::Float64
#     β_v_CCC::Float64
#     α_s_CCC::Float64
#     β_s_CCC::Float64

#     CCC::Float64
#     decay::Any
# end

# #### Define the data update functions

# # 定义SR学习中的决策过程
# function sr_softmax(options_vector::Array{Float64,1}, β, true_selection::Tuple, debug=false)
#     options_matrix = reshape(options_vector, 2, 2)'

#     op_selection_idx = CartesianIndex(true_selection[1], abs(true_selection[2] - 1)) +
#                        CartesianIndex(1, 1)
#     true_selection_idx = CartesianIndex(true_selection) + CartesianIndex(1, 1)

#     if debug
#         println(options_matrix)
#         println("True selection is " * repr(options_matrix[true_selection_idx]))
#         println("Op selection is " * repr(options_matrix[op_selection_idx]))
#     end

#     return exp(β * options_matrix[true_selection_idx]) /
#            (exp(β * options_matrix[true_selection_idx]) +
#             exp(β * options_matrix[op_selection_idx]))
# end

# # 定义参数选择过程的函数
# function get_action_para(env::ExpEnv, agent::RLLearner_basic, realsub::RealSub, idx::Int)
#     if env.env_type[idx] == "v"
#         β = agent.β_v
#         α = agent.α_v
#     elseif env.env_type[idx] == "s"
#         β = agent.β_s
#         α = agent.α_s
#     end

#     return (α, β)
# end

# function get_action_para(env::ExpEnv, agent::RLLearner_witherror, realsub::RealSub, idx::Int)
#     if env.env_type[idx] == "v"
#         if realsub.corrections[idx] == 1
#             β = agent.β_v
#             α = agent.α_v
#         elseif realsub.corrections[idx] == 0
#             β = agent.β_v_error
#             α = agent.α_v_error
#         end
#     elseif env.env_type[idx] == "s"
#         if realsub.corrections[idx] == 1
#             β = agent.β_s
#             α = agent.α_s
#         elseif realsub.corrections[idx] == 0
#             β = agent.β_s_error
#             α = agent.α_s_error
#         end
#     end

#     return (α, β)
# end

# function get_action_para(env::ExpEnv, agent::RLLearner_withCCC, realsub::RealSub, idx::Int,
#                          conflict)
#     if env.env_type[idx] == "v"
#         if realsub.corrections[idx] == 1 && conflict ≥ agent.CCC
#             β = agent.β_v
#             α = agent.α_v
#         elseif realsub.corrections[idx] == 1 && conflict < agent.CCC
#             β = agent.β_v_CCC
#             α = agent.α_v_CCC
#         elseif realsub.corrections[idx] == 0
#             β = agent.β_v_error
#             α = agent.α_v_error
#         end
#     elseif env.env_type[idx] == "s"
#         if realsub.corrections[idx] == 1 && conflict ≥ agent.CCC
#             β = agent.β_s
#             α = agent.α_s
#         elseif realsub.corrections[idx] == 1 && conflict < agent.CCC
#             β = agent.β_s_CCC
#             α = agent.α_s_CCC
#         elseif realsub.corrections[idx] == 0
#             β = agent.β_s_error
#             α = agent.α_s_error
#         end
#     end

#     return (α, β)
# end

# ##### 定义强化学习相关函数

# # 学习具体SR联结的强化学习过程
# function rl_learning_sr(env::ExpEnv, agent::RLLearner, realsub::RealSub; dodecay=true)

#     # Check the subtag
#     if env.sub_tag != realsub.sub_tag
#         return println("The env and sub_real_data not come from the same one!")
#     end

#     # init learning parameters list
#     total_trials_num, options_weight_matrix, p_softmax_history = init_param(env, :sr)

#     # Start learning
#     for idx in 1:total_trials_num
#         if isa(agent, RLLearner_withCCC)
#             conflict = calc_CCC(options_weight_matrix[idx, :],
#                                 (env.stim_task_unrelated[idx],
#                                  env.stim_correct_action[idx]))
#             α, β = get_action_para(env, agent, realsub, idx, conflict)
#         else
#             α, β = get_action_para(env, agent, realsub, idx)
#         end

#         ## Decision
#         p_softmax_history[idx] = sr_softmax(options_weight_matrix[idx, :], β,
#                                             (env.stim_task_unrelated[idx],
#                                              realsub.response[idx]))

#         ## Update
#         options_weight_matrix[idx + 1, :] = update_options_weight_matrix(options_weight_matrix[idx,:],
#                                                                          α, agent.decay,
#                                                                          (env.stim_task_unrelated[idx],
#                                                                           realsub.response[idx]), dodecay=dodecay)

#    end

#     options_weight_result = options_weight_matrix[2:end, :]
#     return Dict(:options_weight_history => options_weight_result,
#                 :p_softmax_history => p_softmax_history)
# end

# end # RLModels_SoftMax

#=============================================================================
# Module2: RLModels without selection                                        #
=============================================================================#
module NoSoftMax

using ..RLModels

#####
##### Define the Class System
#####

"""
    RLLearner_basic

A RLLearner which learnt parameters from the experiment environment.
"""

# 环境中的学习者, 在基本条件下
struct RLLearner_basic <: RLLearner
    α_v::Float64
    α_s::Float64
    decay::Float64
end

# 环境中的学习者, 在错误试次下学习率不同
struct RLLearner_witherror <: RLLearner
    α_v::Float64
    α_s::Float64

    α_v_error::Float64
    α_s_error::Float64

    decay::Float64
end

# 存在冲突控制的学习者
struct RLLearner_withCCC <: RLLearner
    α_v::Float64
    α_s::Float64

    α_v_error::Float64
    α_s_error::Float64

    α_v_CCC::Float64
    α_s_CCC::Float64

    CCC::Float64
    decay::Float64
end

# 存在冲突控制但在错误试次下没有改变学习率的学习者
struct RLLearner_withCCC_no_error <: RLLearner
    α_v::Float64
    α_s::Float64

    α_v_CCC::Float64
    α_s_CCC::Float64

    CCC::Float64
    decay::Float64
end

#### Define the Functions

# 定义SR学习中的决策过程
function selection_value(
    options_vector::Array{Float64,1},
    true_selection::Tuple,
    debug=false,
)
    options_matrix = reshape(options_vector, 2, 2)'
    true_selection_idx = CartesianIndex(true_selection) + CartesianIndex(1, 1)

    if debug
        println(true_selection_idx)
    end

    return options_matrix[true_selection_idx]
end

# 定义抽象概念学习的决策过程
function selection_value(
    options_vector::Array{Float64,1},
    true_selection::Int,
    debug=false,)
    true_selection_idx = true_selection + 1

    if debug
        println(true_selection_idx)
    end

    return options_vector[true_selection_idx]
end

function get_action_para(env::ExpEnv, agent::RLLearner_basic, realsub::RealSub, idx::Int)
    if env.env_type[idx] == "v"
        α = agent.α_v
    elseif env.env_type[idx] == "s"
        α = agent.α_s
    end

    return(α)
end

function get_action_para(env::ExpEnv, agent::RLLearner_witherror, realsub::RealSub, idx::Int)
    if env.env_type[idx] == "v"
        if realsub.corrections[idx] == 1
            α = agent.α_v
        elseif realsub.corrections[idx] == 0
            α = agent.α_v_error
        end
    elseif env.env_type[idx] == "s"
        if realsub.corrections[idx] == 1
            α = agent.α_s
        elseif realsub.corrections[idx] == 0
            α = agent.α_s_error
        end
    end

    return(α)
end

function get_action_para(env::ExpEnv, agent::RLLearner_withCCC, realsub::RealSub, idx::Int, conflict)

    if env.env_type[idx] == "v"
        if realsub.corrections[idx] == 1 && conflict ≥ agent.CCC
            α = agent.α_v
        elseif realsub.corrections[idx] == 1 && conflict < agent.CCC
            α = agent.α_v_CCC
        elseif realsub.corrections[idx] == 0 && -conflict ≥ agent.CCC
            α = agent.α_v_error
        elseif realsub.corrections[idx] == 0 && -conflict < agent.CCC
            α = agent.α_v_CCC
        end
    elseif env.env_type[idx] == "s"
        if realsub.corrections[idx] == 1 && conflict ≥ agent.CCC
            α = agent.α_s
        elseif realsub.corrections[idx] == 1 && conflict < agent.CCC
            α = agent.α_s_CCC
        elseif realsub.corrections[idx] == 0 && -conflict ≥ agent.CCC
            α = agent.α_s_error
        elseif realsub.corrections[idx] == 0 && -conflict < agent.CCC
            α = agent.α_s_CCC
        end
    end

    return(α)
end

function get_action_para(env::ExpEnv, agent::RLLearner_withCCC_no_error, realsub::RealSub, idx::Int, conflict)

    if env.env_type[idx] == "v"
        if realsub.corrections[idx] == 1 && conflict ≥ agent.CCC
            α = agent.α_v
        elseif realsub.corrections[idx] == 1 && conflict < agent.CCC
            α = agent.α_v_CCC
        elseif realsub.corrections[idx] == 0 && -conflict ≥ agent.CCC
            α = agent.α_v
        elseif realsub.corrections[idx] == 0 && -conflict < agent.CCC
            α = agent.α_v_CCC
        end
    elseif env.env_type[idx] == "s"
        if realsub.corrections[idx] == 1 && conflict ≥ agent.CCC
            α = agent.α_s
        elseif realsub.corrections[idx] == 1 && conflict < agent.CCC
            α = agent.α_s_CCC
        elseif realsub.corrections[idx] == 0 && -conflict ≥ agent.CCC
            α = agent.α_s
        elseif realsub.corrections[idx] == 0 && -conflict < agent.CCC
            α = agent.α_s_CCC
        end
    end

    return(α)
end

##### 定义强化学习相关函数

# 学习具体SR联结的强化学习过程
function rl_learning_sr(
    env::ExpEnv,
    agent::RLLearner,
    realsub::RealSub;
    dodecay = true
)

    # Check the subtag
    if env.sub_tag ≠ realsub.sub_tag
        return println("The env and sub_real_data not come from the same one!")
    end

    # init learning parameters list
    total_trials_num, options_weight_matrix, p_selection_history =
        init_param(env, :sr)

    # Start learning
    conflict_list = []

    for idx = 1:total_trials_num

        if isa(agent, RLLearner_withCCC) | isa(agent, RLLearner_withCCC_no_error)
            conflict = calc_CCC(options_weight_matrix[idx,:], (env.stim_task_unrelated[idx],
                                                               env.stim_correct_action[idx]))
            α = get_action_para(env, agent, realsub, idx, conflict)
            push!(conflict_list, conflict)
        else
            α = get_action_para(env, agent, realsub, idx)
        end

        ## Decision
        p_selection_history[idx] = selection_value(
            options_weight_matrix[idx, :],
            (env.stim_task_unrelated[idx], realsub.response[idx]))

        ## Update
        # Please note the first row of the value matrix
        # represent the preparedness of the subject!
        options_weight_matrix[idx + 1, :] =
            update_options_weight_matrix(
                options_weight_matrix[idx, :],
                α,
                agent.decay,
                (env.stim_task_unrelated[idx], realsub.response[idx]),
                dodecay = dodecay
            )
    end

    options_weight_result = options_weight_matrix[2:end, :]
    prediction_error = env.stim_correct_action - p_selection_history
    return Dict(
        :options_weight_history => options_weight_result,
        :p_selection_history => p_selection_history,
        :prediction_error => prediction_error,
        :conflict_list => conflict_list
    )
end

# 学习抽象的 con/inc 概念的强化学习过程
function rl_learning_ab(env::ExpEnv, agent::RLLearner, realsub::RealSub)

    # Check the subtag
    if env.sub_tag ≠ realsub.sub_tag
        return println("The env and sub_real_data not come from the same one!")
    end

    # init learning parameters list
    total_trials_num, options_weight_matrix, p_selection_history, PE_history =
        init_param(env, :ab)

    # Start learning
    for idx = 1:total_trials_num

        if isa(agent, RLLearner_withCCC)
            conflict = calc_CCC(options_weight_matrix[idx,:], env.stim_action_congruency[idx])
            α = get_action_para(env, agent, realsub, idx, conflict)
        else
            α = get_action_para(env, agent, realsub, idx)
        end

        ## Decision
        p_selection_history[idx] = selection_value(options_weight_matrix[idx, :],
                                                   env.stim_action_congruency[idx])
        ## Update
        options_weight_matrix[idx + 1, :] = update_options_weight_matrix(options_weight_matrix[idx, :],
                                                                         α, env.stim_action_congruency[idx])

    end

    options_weight_result = options_weight_matrix[2:end, :]
    prediction_error = env.stim_action_congruency - p_selection_history
    return Dict(
        :options_weight_history => options_weight_result,
        :p_selection_history => p_selection_history,
        :prediction_error => prediction_error
    )
end

end # RLModels_no_SoftMax
end # module
