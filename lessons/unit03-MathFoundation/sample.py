import torch


def main():
    # Initialize random seed.
    torch.manual_seed(407)

    # X has 20 data points, with three features/observations in each row.
    X = torch.randn(20, 3)
    print("X shape:", X.shape)
    # The data was generated from w = 2x - 3y + 0.5z + 0.7.
    true_coefficients = torch.tensor([[2.0], [-3.0], [0.5]])
    true_bias = torch.tensor([0.7])
    Y = X @ true_coefficients + true_bias
    print("Y shape:", Y.shape)

    # Initial guess on coefficients
    coefficients = torch.zeros((3, 1), requires_grad=True)
    bias = torch.zeros((1,), requires_grad=True)
    learning_rate = 0.001

    for step in range(200):
        # perform w = ax + by + cz + bias
        predictions = X @ coefficients + bias
        error = predictions - Y
        loss = (error ** 2).mean()
       
        # Built-in magic, compute gradient
        loss.backward()
        with torch.no_grad():
            coefficients -= learning_rate * coefficients.grad
            bias -= learning_rate * bias.grad
        # clean up gradient before the next iteration
        coefficients.grad.zero_()
        bias.grad.zero_()

        if step % 100 == 0:
            print(f"step {step:3d}: loss = {loss.item():.6f}")

    print("\nLearned coefficients [a, b, c]:")
    # detach to create a new tensor but not tracking the gradient.
    print(coefficients.detach().reshape(-1))
    print("\nLearned bias:")
    print(bias.detach())

    print("\nExpected coefficients [a, b, c]:")
    print(true_coefficients.reshape(-1))
    print("\nExpected bias:")
    print(true_bias)


if __name__ == "__main__":
    main()