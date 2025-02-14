# contrastive_loss.jl - Contrastive Learning Loss Implementation

using LinearAlgebra, Flux, Distances

function contrastive_loss(positive_pairs::Tuple{Vector{Float64}, Vector{Float64}}, 
                          negative_pairs::Tuple{Vector{Float64}, Vector{Float64}}, 
                          margin::Float64=1.0, epsilon::Float64=1e-8)
    pos_vec1 = positive_pairs[1] / (norm(positive_pairs[1]) + epsilon)
    pos_vec2 = positive_pairs[2] / (norm(positive_pairs[2]) + epsilon)
    pos_sim = 1 - cosine_dist(pos_vec1, pos_vec2)
    
    neg_vec1 = negative_pairs[1] / (norm(negative_pairs[1]) + epsilon)
    neg_vec2 = negative_pairs[2] / (norm(negative_pairs[2]) + epsilon)
    neg_sim = 1 - cosine_dist(neg_vec1, neg_vec2)
    
    loss = log(1 + exp(margin - neg_sim)) + log(1 + exp(1 - pos_sim))
    return loss
end

# Example Usage
pos_pair = (rand(128), rand(128))
neg_pair = (rand(128), rand(128))
loss_value = contrastive_loss(pos_pair, neg_pair)
println("Contrastive Loss: ", loss_value)
