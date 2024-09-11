using Distributions
using LinearAlgebra
using Random
using SparseArrays
using Statistics
using Threads

function generate_J(C::Matrix{Float64}, L::Int64, q::Int64, r::Int64, β::Float64)
    J = spzeros(q, q, L, L)
    contacts = Vector{Tuple{Int64, Int64}}()
    neigh = [Vector{Int64}() for _ in 1:L]
    
    for k in 1:r
        i, j = rand(1:L, 2)
        while i == j || j in neigh[i]
            i, j = rand(1:L, 2)
        end
        push!(contacts, i < j ? (i, j) : (j, i))
        
        J[:, :, i, j] = C
        J[:, :, j, i] = C
        
        push!(neigh[i], j)
        push!(neigh[j], i)
    end

    return β * J, contacts
end

function calculate_H(J::SparseArray{Float64, 4}, x::Array{Int64, 1}, L::Int64, contacts::Vector{Tuple{Int64, Int64}})
    H = 0.0
    for (i, j) in contacts
        H += J[1 + x[i], 1 + x[j], i, j]
    end
    return H
end

function metropolis_hastings(J::SparseArray{Float64, 4}, x::Array{Int64, 1}, num_samples::Int64, L::Int64, alphabet::Array{Int64, 1}, contacts::Vector{Tuple{Int64, Int64}}, calH::Function)
    for i in 1:num_samples
        x_new = copy(x)
        idx = rand(1:L)
        x_new[idx] = rand(alphabet)

        α = min(1, exp(-calH(J, x_new, L, contacts) + calH(J, x, L, contacts)))

        if rand() < α
            x = x_new
        end
    end
    return x
end

function construct_tree(J::SparseArray{Float64, 4}, r::Int64, q::Int64, L::Int64, N::Int64, times::Array{Int64, 1}, sampling_method::Function, alphabet::Array{Int64, 1}, d::Int64, contacts::Vector{Tuple{Int64, Int64}}, calH::Function)
    tree = Dict{Int64, Dict{Symbol, Any}}()
    
    x = rand(alphabet, L)
    
    # Generate root
    tree[0] = Dict{Symbol, Any}()
    tree[0][:seq] = copy(sampling_method(J, x, times[1], L, alphabet, contacts, calH))
    tree[0][:level] = 1
    tree[0][:leaf] = 1 == N
    
    for i in 2:N
        @threads for j in 0:(d^(i - 1) - 1)
            node_num = sum(d^k for k in 0:(i - 2)) + j
            parent_num = node_num ÷ d

            tree[node_num] = Dict{Symbol, Any}()
            tree[node_num][:seq] = copy(sampling_method(J, tree[parent_num][:seq], times[i], L, alphabet, contacts, calH))
            tree[node_num][:level] = i
            tree[node_num][:leaf] = i == N
            tree[node_num][:parent] = parent_num

            # Store child references
            tree[parent_num]["child$(div(node_num - j, d) + 1)"] = node_num
        end
    end
    
    return tree
end

function run_simulation(params::Dict{Symbol, Any})
    q = params[:q]
    L = params[:L]
    r = params[:r]
    N = params[:N]
    d = params[:d]
    β = params[:β]
    alphabet = collect(0:(q-1))
    times = params[:times]

    Random.seed!(123)
    B = Matrix{Float64}(I, q, q)
    ν = 10
    C = rand(Wishart(ν, B))

    J, contacts = generate_J(C, L, q, r, β)
    
    # Initial sequence and sampling method (Metropolis Hastings)
    tree = construct_tree(J, r, q, L, N, times, metropolis_hastings, alphabet, d, contacts, calculate_H)
    
    return tree
end