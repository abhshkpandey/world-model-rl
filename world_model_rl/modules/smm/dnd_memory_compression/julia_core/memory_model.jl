# memory_model.jl - Core Julia implementation (DND + Autoencoder)

using Random, LinearAlgebra, Flux, Statistics, NearestNeighbors

struct DifferentiableNeuralDictionary
    memory_size::Int
    k::Int
    keys::Matrix{Float64}
    values::Matrix{Float64}
    tree::Union{KDTree, Nothing}
end

function DifferentiableNeuralDictionary(memory_size=1000, k=5, feature_dim=128)
    return DifferentiableNeuralDictionary(memory_size, k, zeros(feature_dim, memory_size), zeros(feature_dim, memory_size), nothing)
end

mutable struct MemoryBuffer
    dnd::DifferentiableNeuralDictionary
    buffer_size::Int
    current_size::Int

    function MemoryBuffer(memory_size=1000, k=5, feature_dim=128)
        new(DifferentiableNeuralDictionary(memory_size, k, feature_dim), memory_size, 0)
    end
end

function update_tree!(buffer::MemoryBuffer)
    if buffer.current_size > 0
        buffer.dnd.tree = KDTree(buffer.dnd.keys[:, 1:buffer.current_size])
    end
end

function add_memory!(buffer::MemoryBuffer, key::Vector{Float64}, value::Vector{Float64})
    if buffer.current_size < buffer.buffer_size
        buffer.current_size += 1
    else
        buffer.dnd.keys[:, 1:buffer.buffer_size-1] = buffer.dnd.keys[:, 2:buffer.buffer_size]
        buffer.dnd.values[:, 1:buffer.buffer_size-1] = buffer.dnd.values[:, 2:buffer.buffer_size]
    end
    buffer.dnd.keys[:, buffer.current_size] = key
    buffer.dnd.values[:, buffer.current_size] = value
    update_tree!(buffer)
end

function retrieve_memory(buffer::MemoryBuffer, query::Vector{Float64})
    if buffer.current_size == 0 || buffer.dnd.tree === nothing
        return nothing
    end
    indices, distances = knn(buffer.dnd.tree, query, buffer.dnd.k)
    similarity_weights = exp.(-distances) / sum(exp.(-distances))
    retrieved_values = sum(buffer.dnd.values[:, indices] .* similarity_weights', dims=2)
    return retrieved_values
end

# Initialize global Memory Buffer instance
buffer = MemoryBuffer()
