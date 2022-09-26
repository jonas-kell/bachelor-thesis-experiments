### Adapt constants in `main.py`:

```python
constants = PathAndFolderConstants(
    ...
)
```

### Prepare Data

```cmd
python3 main.py prep
```

### Show a stored tensor

Lets you display a tensor that is stored on file as an image.
Will apply inverse of the ImageNet normalization beforehand.

```cmd
python3 main.py show "<<path/to/tensor>>"
```

### Show a stored image

Lets you display an image that is stored on file as an image.

```cmd
python3 main.py show_image "<<path/to/image>>"
```

### Evaluate a stored model

Lets you evaluate a model on custom images to check on its behavior.
The images that you want to evaluate on, need to be placed in the appropriate folder (there is a constant for it).

The model file needs to be available in the `path`. There is no clean way for this as of now. Just look into `evaluating/evaluate_model.py`.

```cmd
python3 main.py eval "<<path/to/stored/model>>"
```

Example model paths for my system:

```
"/media/jonas/69B577D0C4C25263/MLData/tensorboard/DINO-CLASSIFIER/adamw/model_41.pt"
```

### Default Training pass

```cmd
python3 main.py train
```

Additional parameters that modify the training behavior can be specified:

```cmd
python3 main.py train <<argument>>=<<value>>
```

At the moment the following parameters are available:

```
"<<argument>>=<<example-value>>"

"learning_rate=1e-3"
"momentum=0.9"
"dampening=0"
"weight_decay=0"
"epochs=80"
"batch_size=128"
"loss_fn_name=cross_entropy_loss"
"optimizer_name=sgd"
```

A training pass can be continued at the point of each epoch:

```cmd
python3 main.py train continue=<<path/to/tensorboard/log/folder/of/run>>
```
