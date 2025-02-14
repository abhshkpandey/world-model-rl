import torch

def run_inference(model, state):
    """
    Runs inference on a given state using the trained model.
    :param model: The trained RL model.
    :param state: The input state for inference.
    """
    model.eval()
    with torch.no_grad():
        action_probs = model(state)
        selected_action = torch.argmax(action_probs, dim=-1).item()
    print(f"Predicted Action: {selected_action}")
    return selected_action
