#!/usr/bin/env python3

import numpy as np

x = np.array([
    ('Rex', 9, 81.0),
    ('Fido', 3, 27.0)
    ],
    dtype=[
        ('name', 'U10'),
        ('age', 'i4'),
        ('weight', 'f4')
    ])
print(x)
print(x[1])
print(x['age'])
x['age'] = 5
print(x)

'''
Structured arrays are designed for low-level manipulation of structured data,
for example, for interpreting binary blobs.
Structured datatypes are designed to mimic 'structs' in the C langauge,
making them also useful of interfacing with C code.

For these purposes, numpy supports specialized features such as subarrays
and nested datatypes, and allows manual control over the memory layout
of the strcuture.

For simple manipulation of tabular data other pydata projects, such
as pandas, xarray, or DataArray, provide higher-level interfaces
that may be more suitable. These projects may also give better
performance for tabular data analysis because the C-struct-like
memory layout of structured arrays can lead to poor cache behavior.
'''

'''Structured Datatypes


'''


'''Indexing Structured Arrays

Accessing individual fields

Individual fields of a structured array may be accessed
and modified by indexing the array with the field name. 
'''
x = np.array([(1, 2), (3, 4)], dtype=[('foo', 'i8'), ('bar', 'f4')])
print(x['foo'])
x['foo'] = 100
print(x)

'''
The resulting array is a view into the original array.
It shares the same memory locations and writing to the view
will modify the original array.
'''
y = x['bar']
y[:] = 10
print(x)
'''
This view has the same dtype and itemsize as the indexed field,
so it is typically a non-structured array, except in the case of
nested structures.
'''
print(y.dtype, y.shape, y.strides)

'''
Accessing Multiple Fields

One can index and assign to a structured array with a multi-field
index, where the index is a list of field names.
'''

a = np.zeros(3, dtype=[('a', 'i4'), ('b', 'i4'), ('c', 'f4')])
print(a[['a', 'c']])

'''
Assigning to an array with a multi-field index will behave the same
in NumPy 1.14 and NumPy 1.15. In both versions, the assignment
will modify the original array:
'''
a[['a', 'c']] = (2, 3)
print(a)

'''
This obeys the structured array assignment rules described above.
For example, this means that one can swap the values of two fields
using appropriate multi-field indexes:
'''
a[['a', 'c']] = a[['c', 'a']]

'''Indexing with an Integer to get a Structured Scalar

Indexing a single element of a structured array (with an integer index)
returns a structured scalar:

'''
x = np.array([(1, 2., 3.)], dtype='i,f,f')
scalar = x[0]
print(scalar)
print(type(scalar))

'''
Unlike other numpy scalars, structured scalars are mutable and
act like views into the original array, such that modifying the scalar
will modify the original array. Structured arrays also support
access and assignment by field name:
'''
x = np.array([(1, 2), (3, 4)], dtype=[('foo', 'i8'), ('bar', 'f4')])
s = x[0]
s['bar'] = 100 # accessible and assignable by field name
print(x)

'''
Similarly to tuples, structured scalars can also be indexed with an integer:
'''
scalar = np.array([1, 2., 3], dtype='i,f,f')[0]
print(scalar[0])
scalar[1] = 4
'''
Thuse, tuples might be thought of as the native Python equivalent to
numpy's structured types, much like native python integers are
the equivalent to numpy's integer types.
Structured scalars may be converted to a tuple by calling ndarray.item
'''
print(scalar.item(), type(scalar.item()))






