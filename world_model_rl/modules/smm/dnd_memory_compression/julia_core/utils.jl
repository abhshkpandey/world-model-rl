# utils.jl - Utility Functions for DND + Memory Compression

using LinearAlgebra, Statistics, Distances

"""
    normalize_vector(vec::Vector{Float64}; epsilon::Float64=1e-8) -> Vector{Float64}

Normalizes a vector to unit norm with numerical stability.
Avoids unnecessary computation if the vector is already close to zero.
"""
function normalize_vector(vec::Vector{Float64}; epsilon::Float64=1e-8)
    vec_norm = norm(vec)
    return vec_norm < epsilon ? vec : vec / (vec_norm + epsilon)
end

"""
    cosine_similarity(vec1::Vector{Float64}, vec2::Vector{Float64}) -> Float64

Computes cosine similarity between two vectors using Distances.jl for better numerical stability.
"""
function cosine_similarity(vec1::Vector{Float64}, vec2::Vector{Float64})
    return 1 - cosine_dist(vec1, vec2)
end

"""
    moving_average(data::Vector{Float64}, window_size::Int) -> Vector{Float64}

Computes a simple moving average for a given data vector using an optimized rolling window approach.
"""
function moving_average(data::Vector{Float64}, window_size::Int)
    if length(data) < window_size
        return [mean(data[1:i]) for i in 1:length(data)]
    end
    
    avg_values = zeros(length(data))
    current_sum = sum(data[1:window_size])
    avg_values[window_size] = current_sum / window_size
    
    for i in (window_size+1):length(data)
        current_sum += data[i] - data[i - window_size]
        avg_values[i] = current_sum / window_size
    end
    
    return avg_values
end

"""
    euclidean_distance(vec1::Vector{Float64}, vec2::Vector{Float64}) -> Float64

Computes the Euclidean distance between two vectors.
"""
function euclidean_distance(vec1::Vector{Float64}, vec2::Vector{Float64})
    return norm(vec1 - vec2)
end

# Example Usage
demo_vec1 = rand(128)
demo_vec2 = rand(128)

println("Cosine Similarity: ", cosine_similarity(demo_vec1, demo_vec2))
println("Euclidean Distance: ", euclidean_distance(demo_vec1, demo_vec2))
