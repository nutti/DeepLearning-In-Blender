import torch
import torchvision
import torch.nn.functional as F

DATA_ROOT = "./mnist-data"
TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 32
LOG_INTERVAL = 10
NUMBER_OF_EPOCHS = 10
BASE_LEARNING_RATE = 0.1


def main():
    torch.manual_seed(1)

    # Create device.
    if torch.cuda.is_available():
        device_type = "cpu"
    else:
        device_type = "cpu"
    device = torch.device(device_type)

    # Build model.
    model = torch.nn.Sequential(
        torch.nn.Linear(28*28, 256),
        torch.nn.Tanh(),

        torch.nn.Linear(256, 10),
        torch.nn.Softmax()
    )
    model.to(device)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Load dataset.
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(DATA_ROOT, train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                   ])
        ),
        batch_size=TRAIN_BATCH_SIZE,
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(DATA_ROOT, train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                   ])
        ),
        batch_size=TEST_BATCH_SIZE,
    )


    # Train loop.
    for epoch in range(NUMBER_OF_EPOCHS):
        # Train.
        print("Start train.")
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.reshape(data.shape[0], -1)
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            loss = F.nll_loss(torch.log(output), target)

            loss.backward()
            optimizer.step()

            #if batch_idx % LOG_INTERVAL == 0:
            print("Batch: {}, Loss: {}".format(batch_idx, loss.item()))

        # Evaluate.
        print("Start eval.")
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.reshape(data.shape[0], -1)
                data = data.to(device)
                target = target.to(device)
                output = model(data)

                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print("Average loss: {}, Accuracy: {}".format(test_loss, correct / len(test_loader.dataset)))


    torch.save(model.state_dict(), "mnist.pth")


if __name__ == "__main__":
    main()
