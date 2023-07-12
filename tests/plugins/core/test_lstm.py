from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from dc_check.plugins.core.datahandler import DataHandler
from dc_check.plugins.core.models.simple_mlp import SimpleMLP
from dc_check.utils.constants import DEVICE


def test_lstm():
    X, y = load_iris(return_X_y=True, as_frame=True)
    df = X.copy(deep=True)
    df["target"] = y
    std_scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    X_train = std_scaler.fit_transform(X_train)
    X_test = std_scaler.transform(X_test)

    datahander = DataHandler(X_train, y_train, batch_size=32)
    dataloader = datahander.dataloader

    model = SimpleMLP()
    # creating our optimizer and loss function object
    learning_rate = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Move model to DEVICE
    model.to(DEVICE)

    # Set model to training mode
    optimizer.lr = learning_rate
    losses = []
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for batch_idx, data in enumerate(dataloader):
            inputs, true_label, indices = data

            inputs = inputs.to(DEVICE)
            true_label = true_label.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, true_label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)
    assert losses[-1] < losses[0]
