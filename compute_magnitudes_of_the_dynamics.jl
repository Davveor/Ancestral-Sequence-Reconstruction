using LinearAlgebra

# Define la función para llenar el diccionario de tiempos
function fill_times(t::Vector{Int64}, adj_list::Dict{Int64,Dict{Symbol,Any}})
    times = Dict{Tuple{Int64, Int64}, Int64}()

    # Función auxiliar para encontrar la ruta hacia la raíz de un nodo
    function path_to_root(node_id)
        path = []
        current_node = node_id
        while current_node != nothing
            push!(path, current_node)
            if haskey(adj_list[current_node], :parent)
                current_node = adj_list[current_node][:parent]
            else
                current_node = nothing  # Si no tiene padre, asumimos que es la raíz
            end
        end
        reverse!(path)  # La ruta va desde la raíz hasta el nodo
        return path
    end

    # Función auxiliar para calcular el tiempo hasta la raíz para un nodo dado
    function time_to_root(node_id)
        level = length(path_to_root(node_id)) - 1
        return sum(t[1:level])
    end

    # Encuentra el ancestro común más cercano entre dos nodos
    function common_ancestor(node_i, node_j)
        path_i = path_to_root(node_i)
        path_j = path_to_root(node_j)
        
        min_len = min(length(path_i), length(path_j))
        ancestor = nothing
        for k in 1:min_len
            if path_i[k] == path_j[k]
                ancestor = path_i[k]
            else
                break
            end
        end
        return ancestor
    end

    # Llena el diccionario de tiempos entre todos los pares de nodos
    node_ids = keys(adj_list)
    for i in node_ids
        for j in node_ids
            if i != j
                ancestor = common_ancestor(i, j)
                time_ancestor = time_to_root(ancestor)
                time_i = time_to_root(i)
                time_j = time_to_root(j)
                times[(i, j)] = (time_i + time_j) - 2 * time_ancestor
            else
                times[(i, j)] = 0  # El tiempo entre un nodo y sí mismo es 0
            end
        end
    end

    return times
end

function fill_lambda(C::Array{Float64,2}, gamma::Float64, times::Dict{Tuple{Int64, Int64}, Int64})
    # Crear el diccionario para almacenar las matrices Lambda_{ij}.
    lambda = Dict{Tuple{Int64, Int64}, Array{Float64,2}}()
    inv_C = inv(C)
    # Iterar sobre cada par de índices (i, j) en el diccionario de tiempos.
    for (ij, time) in times
        # Calcular Lambda_{ij} = exp(-γ * inv(C) * Δt_{ij})
        lambda[ij] = exp(-gamma * inv_C * time)
    end
    
    return lambda
end

function fill_sigma(C::Array{Float64,2}, lambda::Dict{Tuple{Int64, Int64}, Array{Float64,2}})
    # Crear el diccionario para almacenar las matrices Sigma_{ij}.
    sigma = Dict{Tuple{Int64, Int64}, Array{Float64,2}}()
    
    # Iterar sobre cada par de índices (i, j) en el diccionario de lambda.
    for (ij, Λ) in lambda
        # Calcular Sigma_{ij} = C - Λ_{ij} * C * Λ_{ij}
        sigma[ij] = C - Λ * C * Λ
    end
    
    return sigma
end

function fill_J(adj_list::Dict{Int64, Dict{Symbol, Any}}, 
    lambda::Dict{Tuple{Int64, Int64}, Array{Float64,2}}, 
    sigma::Dict{Tuple{Int64, Int64}, Array{Float64,2}}, 
    q::Int64, L::Int64, max_node::Int64)

    # Crear el diccionario para almacenar las matrices J_{ij}.
    J = Dict{Tuple{Int64, Int64}, Array{Float64,2}}()

    # Tamaño de las matrices (q*L) x (q*L)
    matrix_size = q * L

    # Llenar todas las posibles tuplas con matrices de ceros.
    for i in 0:max_node
        for j in 0:max_node
            J[(i, j)] = zeros(matrix_size, matrix_size)
        end
    end

    # Iterar sobre cada nodo en la lista de adyacencia.
    for (node, info) in adj_list
    # Iterar sobre cada hijo del nodo actual (es decir, sus vecinos en el árbol).
        for child in info[:children]
            # Calcular J_{ij} = Λ_{ij} * Σ_{ij}^{-1}
            J[(node, child)] = lambda[(node, child)] * inv(sigma[(node, child)])
            # Asegurar que J_{ji} = J_{ij}
            J[(child, node)] = copy(J[(node, child)])
        end
    end

    return J
end

function fill_H(adj_list::Dict{Int64, Dict{Symbol, Any}}, 
        C::Array{Float64,2}, 
        lambda::Dict{Tuple{Int64, Int64}, Array{Float64,2}}, 
        sigma::Dict{Tuple{Int64, Int64}, Array{Float64,2}})

    H = Dict{Int64, Array{Float64,2}}()

    for (i, info) in adj_list
        if i == 0
            # Caso 1: Nodo raíz (i = 0)
            sum_term = zeros(Float64, size(C))  # Inicializar la suma
            for j in info[:children]
                Λ_squared = lambda[(i, j)]^2
                sum_term += Λ_squared * inv(I - Λ_squared)
            end
            H[i] = -0.5 * inv(C) * (I + sum_term)

        elseif info[:leaf]
            # Caso 2: Nodo hoja
            H[i] = -0.5 * inv(sigma[(info[:parent], i)])

        else
            # Caso 3: Nodo interno
            parent = info[:parent]
            Λ_parent_squared = lambda[(parent, i)]^2
            sum_term = inv(I - Λ_parent_squared)
            for j in info[:children]
                Λ_squared = lambda[(i, j)]^2
                sum_term += Λ_squared * inv(I - Λ_squared)
            end

            H[i] = -0.5 * inv(C) * sum_term

        end
    end

    return H
end

function fill_mu(adj_list::Dict{Int64, Dict{Symbol, Any}},
        lambda::Dict{Tuple{Int64, Int64}, Array{Float64,2}},
        sigma::Dict{Tuple{Int64, Int64}, Array{Float64,2}},
        datos::Dict{Int64, Array{Float64,1}},
        C::Array{Float64,2})  # Pasamos C como argumento
    mu = Dict{Int64, Array{Float64,1}}()

    for (i, info) in adj_list
        sum_term = zeros(Float64, size(C, 1))  # Inicializamos sum_term usando C

        if !isempty(info[:children])
            for j in info[:children]
                if haskey(datos, j)
                    Λ = lambda[(i, j)]
                    Σ_inv = inv(sigma[(i, j)])
                    x_j = datos[j]

                    sum_term += Λ * (Σ_inv * x_j)
                end
            end
        end

        if sum(sum_term) != 0
            mu[i] = sum_term

        else
            mu[i] = zeros(Float64, size(C, 1))
        end
    end

    return mu
end

function compute_magnitudes(tree::Dict{Int64, Dict{Symbol, Any}},
        leaves::Dict{Int64, Dict{Symbol, Any}},
        C::Array{Float64,2},
        gamma::Float64,
        t::Array{Int64,1},
        q::Int64,
        L::Int64
    )

    datos = Dict{Int64, Array{Float64,1}}()
    for (node, info) in leaves
        datos[node] = info[:binary_seq]
    end 

    max_node = maximum(keys(tree))

    times = fill_times(t,tree)
    Lambda = fill_lambda(C, gamma, times)
    Sigma = fill_sigma(C, Lambda)
    J = fill_J(tree, Lambda, Sigma , q, L, max_node)
    H = fill_H(tree, C, Lambda, Sigma)
    mu = fill_mu(tree, Lambda, Sigma, datos, C)

    return times, Lambda, Sigma, J, H, mu
end 
