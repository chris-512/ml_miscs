#!/usr/bin/env python3

import numpy as np
x = np.array([1., -1., -2., 3])
x[x < 0] += 20
print(x)
