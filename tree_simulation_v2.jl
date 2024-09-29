using Distributions
using LinearAlgebra
using Random
#using FastaIO
#using PlmDCA
#using PyPlot
using Statistics
using Base.Threads

function sample_C(L::Int64, q::Int64, μ::Float64, σ::Float64)
    C = zeros(q, q)
    for i in 1:q
        for j in 1:q
            C[i, j] = rand(Normal(μ, σ))
        end
    end
    return C
end

function generate_J(C::Matrix{Float64}, L::Int64, q::Int64, r::Int64, β::Float64)
    J = zeros(q, q, L, L)
    contacts= Vector{Tuple{Int64,Int64}}()
    neigh = [Vector{Int64}() for _ in 1:L]
    for k in 1:r
        i, j = rand(1:L, 2)
        while i == j || j in neigh[i] 
            i, j = rand(1:L, 2)
        end
    
        push!(contacts, i<j ? (i,j) : (j,i))
        
        J[:,:,i,j] = copy(C)
        J[:,:,j,i] = copy(C)
        
        push!(neigh[i],j)
        push!(neigh[j],i)

    end

    return β * J, contacts
end

function calculate_H(J::Array{Float64,4}, x::Array{Int64,1}, L::Int64, contacts::Vector{Tuple{Int64, Int64}}) # calculo del hamiltoniano teniendo en cuenta solo los vecinos
    H = 0.0
    for (i,j) in contacts
        H += J[1+x[i],1+x[j],i,j]
    end
    return H
end

function gibbs_sampling(J::Array{Float64,4}, x::Array{Int64,1}, num_samples::Int64, L::Int64, alphabet::Array{Int64,1},contacts::Vector{Tuple{Int64, Int64}}, calH::Function)
    x_c = copy(x)
    for _ in 1:num_samples
        for idx in 1:L
            probabilities = [begin 
                                x_c[idx] = q
                                exp(-calH(J, x_c, L, contacts))
                             end for q in alphabet]
            probabilities /= sum(probabilities)
            x_c[idx] = rand(Categorical(probabilities))-1
        end
    end
    return x_c
end

function metropolis_hastings(J::Array{Float64,4}, x::Array{Int64,1}, num_samples::Int64, L::Int64, alphabet::Array{Int64,1},contacts::Vector{Tuple{Int64, Int64}}, calH::Function)
    
    for i in 1:num_samples
        x_new = copy(x)
        idx = rand(1:L) 
        x_new[idx] = rand(alphabet)

        α = min(1,  exp(-calH(J,x_new,L,contacts)+calH(J,x,L,contacts)))

        if rand() < α
            x = x_new 
        end
        
    end
    return x
end

function construct_tree(J::Array{Float64,4}, r::Int64, q::Int64, L::Int64, k::Int64, times::Array{Int64, 1}, sampling_method::Function, alphabet::Array{Int64,1}, c::Int64, contacts::Vector{Tuple{Int64, Int64}}, calH::Function)

    tree = Dict{Int64, Dict{Symbol, Any}}()
    
    x = rand(alphabet, L)
    
    # generate root 
    tree[0] = Dict{Symbol, Any}()
    tree[0][:seq] = copy(sampling_method(J, x, times[1], L, alphabet,contacts, calH))
    tree[0][:level] = 0
    tree[0][:leaf] = 1 == k+1
    tree[0][:children] = []
    
    for i in 2:(k+1)
        @threads for j in 0:(c^(i-1)-1)
            node_num = sum(c^n for n in 0:(i-2)) + j
            parent_num = (node_num - 1) ÷ c

            tree[node_num] = Dict{String, Any}()
            tree[node_num][:seq] = copy(sampling_method(J, tree[parent_num][:seq], times[i], L, alphabet,contacts, calH))
            tree[node_num][:level] = (i-1)
            tree[node_num][:leaf] = i == (k+1)
            tree[node_num][:parent] = parent_num
            tree[node_num][:children] = []

            push!(tree[parent_num][:children],node_num)
        end
    end
    
    return tree
end

function fill_leaves(tree::Dict{Int64, Dict{Symbol, Any}})
    leaves = Dict{Int64, Dict{Symbol, Any}}()

    for (node_num, node_data) in tree
        # Verificar si el nodo es una hoja
        if node_data[:leaf] == true
            # Crear un subdiccionario para almacenar :seq y :parent
            leaf_data = Dict{Symbol, Any}()
            leaf_data[:seq] = copy(node_data[:seq])
            leaf_data[:parent] = copy(node_data[:parent])
            
            # Agregarlo al diccionario leaves con el ID del nodo como clave
            leaves[node_num] = leaf_data
        end
    end
    return leaves
end

function one_hot_encoding(seq::Vector{Int64}, q::Int64, L::Int64)
    b = [zeros(Float64, q) for _ in 1:L]  # Crear un arreglo de q*L

    for i in 1:L
        state = seq[i]
        b[i][state + 1] = 1  # Sumar 1 a state porque el alfabeto original va entre {0,...,q-1}
    end

    flat_b = vcat(b...)  # Aplanar el vector de vectores en un solo vector

    return flat_b
end

function convert_to_continuous!(leaves::Dict{Int64, Dict{Symbol, Any}}, q::Int64, L::Int64)
    # Paso 1: Primero, generar todas las secuencias con ruido
    for (leaf_id, leaf_info) in leaves
        seq = leaf_info[:seq]  # Obtener la secuencia original
        bin_seq = one_hot_encoding(seq, q, L)  # Convertir a arreglo one-hot

        # Sumar ruido gaussiano a cada posición
        for i in 1:length(bin_seq)
            bin_seq[i] += rand(Normal(1/q, 0.01))  
        end

        # Guardar la secuencia con ruido en 'leaves'
        leaf_info[:binary_seq] = bin_seq
    end

    #= Paso 2: Calcular la media global de todas las secuencias
    all_sequences = vcat([leaves[id][:binary_seq] for id in keys(leaves)]...)
    global_mean = mean(all_sequences)

    # Paso 3: Restar la media global a cada secuencia
    for (leaf_id, leaf_info) in leaves
        leaf_info[:binary_seq] .-= global_mean  # Centrar cada secuencia restando la media global
        leaf_info[:global_mean] = global_mean  # Guardar la media global en el diccionario
    end=#

    # Paso 2: Calcular la media de cada columna
    all_sequences = vcat([leaves[id][:binary_seq] for id in keys(leaves)]...)
    column_means = mean(all_sequences, dims=1)  # Calcular la media de cada columna

    # Paso 3: Restar la media de cada columna a cada secuencia
    for (leaf_id, leaf_info) in leaves
        leaf_info[:binary_seq] .-= column_means  # Centrar cada secuencia restando la media de cada columna
        leaf_info[:data_mean] = column_means  # Guardar la media de cada columna en el diccionario
    end

    return leaves
end

function correlation_matrix(leaves::Dict{Int64, Dict{Symbol, Any}})
    sequences = [leaves[id][:binary_seq] for id in keys(leaves)]
    Cemp = cor(hcat(sequences...)')  
    return Cemp
end

function run_simulation(params::Dict{Symbol, Any})
    q = params[:q] #size of the alphabet 
    L = params[:L] # size of the sequences
    r = params[:r] # connectivity of the Potts model
    k = params[:k] # number of levels in the tree k+1
    c = params[:c] # branching per node
    β = params[:β]
    alphabet = collect(0:(q-1))
    times = params[:times] # debe haber k+1 tiempos

    μ = get(params, :μ, 0.8) # valor por defecto 0.8
    σ = get(params, :σ, 0.2) # valor por defecto 0.2

    C = sample_C(L, q, μ, σ)

    J, contacts = generate_J(C, L, q, r, β)
    
    # Initial sequence and sampling method (Metropolis Hastings)
    tree = construct_tree(J, r, q, L, k, times, metropolis_hastings, alphabet, c, contacts, calculate_H)
    leaves = fill_leaves(tree)
    convert_to_continuous!(leaves,q,L)
    C = correlation_matrix(leaves)

    max_node = maximum(keys(leaves))
    data_mean = leaves[max_node][:data_mean]

    return tree, leaves, C, data_mean
end