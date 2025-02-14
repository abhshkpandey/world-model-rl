import torch

def run_ablation_study(model, test_data, remove_module):
    """
    Runs an ablation study by removing a specific module and measuring the impact.
    :param model: The RL model being tested.
    :param test_data: The dataset used for evaluation.
    :param remove_module: The name of the module to be ablated.
    """
    original_performance = evaluate_model(model, test_data)
    
    # Remove the specified module
    if hasattr(model, remove_module):
        setattr(model, remove_module, None)
    
    ablated_performance = evaluate_model(model, test_data)
    
    print(f"Ablation Study - Removed {remove_module}:")
    print(f"Original Performance: {original_performance}")
    print(f"Ablated Performance: {ablated_performance}")
    
    return original_performance, ablated_performance

def evaluate_model(model, test_data):
    """
    Evaluates the model's performance on test data.
    """
    with torch.no_grad():
        total_reward = 0
        for state, action, reward, next_state, done in test_data:
            predicted_action = model(state).argmax().item()
            total_reward += reward if predicted_action == action else 0
        return total_reward / len(test_data)  # Returns average reward
