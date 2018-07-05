import numpy as np

x = np.array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8],
           [9, 10, 11]])
rows = np.array([[0, 0], [3, 3]], dtype=np.intp)
cols = np.array([[0, 2], [0, 2]], dtype=np.intp)
# value in same axis becomes the pair coordinates = (0, 0), (0, 2), (3, 0), (3, 2)
print(x[rows, cols])

'''Advanced indexing

However, since the indexing arrays above just repeat themselves, broadcasting can be used
(compare operations such as rows[:, np.newaxis] + columns)
to simplify this: row[:, np.newaxis] + columns

'''

rows = np.array([0, 3], dtype=np.intp)
cols = np.array([0, 2], dtype=np.intp)
# adding new axis to rows
# rows[:, np.newaxis]
print(x[rows[:, np.newaxis], cols])

# This broadcasting can also be achieved using the function ix_:
print(x[np.ix_(rows, cols)])
# Note that without the np.ix_ call, only the diagonal elements would be selected,
# as was used in the previous example.
# This difference is the most important thing to remember about indexing with multiple advanced indexes.
