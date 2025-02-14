import torch

def benchmark_model(model, test_data):
    """
    Runs benchmark tests on the given model using test data.
    :param model: The RL model to benchmark.
    :param test_data: The dataset used for evaluation.
    """
    performance = evaluate_model(model, test_data)
    print(f"Benchmark Performance: {performance}")
    return performance

def evaluate_model(model, test_data):
    """
    Evaluates model performance using cumulative rewards.
    """
    with torch.no_grad():
        total_reward = 0
        for state, action, reward, next_state, done in test_data:
            predicted_action = model(state).argmax().item()
            total_reward += reward if predicted_action == action else 0
        return total_reward / len(test_data)  # Returns average reward
