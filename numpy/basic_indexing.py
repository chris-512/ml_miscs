#!/usr/bin/env python3

import numpy as np

'''Integer array indexing

Integer array indexing allows selection of arbitrary items in the array
based on their N-dimensional index. Each integer array represents
a numbe of indexes into that dimension.

Example:

From each row, a specific element should be selected. The row index is just [0, 1, 2]
and the column index specifies the element to choose for the corresponding row, here [0, 1, 0].
Using both together the task can be solved using advanced indexing:

'''
x = np.array([[1, 2], [3, 4], [5, 6]])
print(x[[0, 1, 2], [0, 1, 0]])

'''
To achieve a behaviour similar to the basic slicing above, broadcasting can be used.

The function ix_ can help with this broadcasting. This is best understood with an example.

For example:

From a 4x3 array the corner elements should be selected using advanced indexing.
Thus all elements for which the column is one of [0, 2] and the row is one of [0, 3]
need to be selected. To use advanced indexing one needs to select all elements explicitly.

Using the method explained previously one could write:

'''
item = np.arange(12)
x = np.array(item)
x = np.reshape(x, [4, 3])
rows = np.array([[0, 0],
                 [3, 3]], dtype=np.intp)
cols = np.array([[0, 2],
                 [0, 2]], dtype=np.intp)
print(x[rows, cols])

