import torch

def test_model(model, test_data):
    """
    Tests the model's performance on a given dataset.
    :param model: The trained RL model.
    :param test_data: The dataset for testing.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for state, action, _, _, _ in test_data:
            predicted_action = model(state).argmax().item()
            correct += (predicted_action == action)
            total += 1
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy
