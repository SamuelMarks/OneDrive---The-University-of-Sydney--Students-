from trax import layers as tl


def get_model(num_classes):
    return tl.Serial(
        tl.Flatten(),
        tl.Dense(512),
        tl.Relu(),
        tl.Dense(512),
        tl.Relu(),
        tl.Dense(num_classes),
        tl.LogSoftmax(),
    )
