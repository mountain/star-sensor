import torch


def adjusted_mse(theta_hat, theta):
    mask = (abs(theta_hat - theta) > 180)
    theta_diff = theta_hat - theta
    theta_diff[mask] = 360 - abs(theta_hat[mask] - theta[mask])
    return sum(theta_diff ** 2) / len(theta_hat)


if __name__ == '__main__':
    x = ([229.4433], [259.0966], [235.3389], [222.0915], [80.1688], [194.6028],
         [350.4031], [138.9966], [333.5420], [89.3357], [313.5604], [288.6507],
         [65.6733], [173.1181], [244.5383], [261.0913])

    y = ([29.4433], [59.0966], [135.3389], [182.0915], [180.4211], [294.6028],
         [50.4031], [338.9966], [300.5420], [289.3357], [13.5604], [88.6507],
         [165.6733], [273.1181], [344.5383], [161.0913])
    x = torch.tensor(x)
    y = torch.tensor(y)
    print(adjusted_mse(x, y))
