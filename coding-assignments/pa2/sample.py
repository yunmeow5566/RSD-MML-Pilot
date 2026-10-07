from pathlib import Path

import torch
import torch.nn as nn


class SampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 5)
        self.activation = nn.ReLU()
        self.output = nn.Linear(5, 2)

    def forward(self, x):
        x = self.activation(self.hidden(x))
        return self.output(x)


def main():
    # Ensure the same initial weights for the sample model.
    torch.manual_seed(407)

    # Load provided training and testing dataset
    data_dir = Path(__file__).resolve().parent / "generated_data"
    train_data = torch.load(data_dir / "train_data.pt", weights_only=True)
    test_data = torch.load(data_dir / "test_data.pt", weights_only=True)
    train_X, train_Y = train_data["X"], train_data["Y"]
    test_X, test_Y = test_data["X"], test_data["Y"]

    print("Train input shape:", train_X.shape)
    print("Train target shape:", train_Y.shape)
    print("Test input shape:", test_X.shape)
    print("Test target shape:", test_Y.shape)

    # Set up model, loss function and optimizer; feel free to play with different methods.
    model = SampleModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Perform training
    for step in range(200):
        optimizer.zero_grad()
        predictions = model(train_X)
        loss = criterion(predictions, train_Y)
        loss.backward()
        optimizer.step()

        if step % 40 == 0:
            print(f"step {step:4d}: train loss = {loss.item():.6f}")

    # Save the model you just trained. Please include this in your assignment
    model_path = data_dir / "sample_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved trained model to: {model_path}")

    # Load the model you just trained.
    loaded_model = SampleModel()
    loaded_model.load_state_dict(torch.load(model_path, weights_only=True))
    loaded_model.eval()

    # Perform testing
    with torch.no_grad():
        test_loss = criterion(loaded_model(test_X), test_Y)

    # Goal: make this test loss as small as possible! do not train with testing dataset. We have a blind set!
    print("\nFinal test loss:", test_loss.item())


    print("\nSample predictions:")
    with torch.no_grad():
        sample_predictions = loaded_model(test_X[:3])
        print(sample_predictions)
    print("\nSample targets:")
    print(test_Y[:3])


if __name__ == "__main__":
    main()
