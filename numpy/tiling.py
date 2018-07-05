#!/usr/bin/env python3

import numpy as np

# version 1
x = np.linspace(-1.0, 1.0, 10)
x = np.expand_dims(x, 0)
x = np.tile(x, [10, 1])
x = np.transpose(x, [1, 0])
print(x)

# version 2
x = np.linspace(-1.0, 1.0, 10)
x = np.reshape(x, [10, 1])
x = np.tile(x, [1, 10])
print(x)
