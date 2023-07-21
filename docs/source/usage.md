# Usage

## 🚀 Installation

The library can be installed from PyPI using
```bash
$ pip install datagnosis
```
or from source, using
```bash
$ pip install .
```
Other library extensions:
 * Install the library with unit-testing support
```bash
 pip install datagnosis[testing]
```

## 💥 Sample Usage

```
# Load iris dataset from sklearn and create DataHandler object
from sklearn.datasets import load_iris
from datagnosis.plugins.core.datahandler import DataHandler
X, y = load_iris(return_X_y=True, as_frame=True)
datahander = DataHandler(X, y, batch_size=32)

# Create model an parameters
from datagnosis.plugins.core.models.simple_mlp import SimpleMLP
import torch

model = SimpleMLP()

# creating our optimizer and loss function object
learning_rate = 0.01
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


# Get a plugin and fit it
hcm = Plugins().get(
    "vog",
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    lr=learning_rate,
    epochs=10,
    num_classes=3,
    logging_interval=1,
)
hcm.fit(
    datahandler=datahander,
    use_caches_if_exist=True,
)

# Plot the resulting scores
hcm.plot_scores(axis=1, plot_type="scatter")
```
