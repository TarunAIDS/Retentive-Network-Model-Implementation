import torch
import torch.nn as nn
import models.retnet
import os
import json

# Check if hyperparameters.json exists
if os.path.exists("hyperparameters.json"):
    with open("hyperparameters.json", "r") as f:
        hyperparameters = json.load(f)
else:
    # Define hyperparameters dictionary
    hyperparameters = {
        "layers": 24,
        "hidden_dim": 2048,
        "ffn_size": 4096,
        "heads": 16,
        "double_v_dim": True,
        "learning_rate": 0.001,
        "num_epochs": 10
    }

if __name__ == "__main__":
    # Verify model size for hyperparameters in paper
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract hyperparameters
    layers = hyperparameters["layers"]
    hidden_dim = hyperparameters["hidden_dim"]
    ffn_size = hyperparameters["ffn_size"]
    heads = hyperparameters["heads"]
    double_v_dim = hyperparameters["double_v_dim"]
    learning_rate = hyperparameters["learning_rate"]
    num_epochs = hyperparameters["num_epochs"]

    # Create RetNet model
    retnet_model = retnet.RetNet(layers, hidden_dim, ffn_size, heads, double_v_dim=double_v_dim).to(device)
    print("1.3B model:", sum(p.numel() for p in retnet_model.parameters() if p.requires_grad))
