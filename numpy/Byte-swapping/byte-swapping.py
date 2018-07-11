'''Introduction to byte ordering and ndarrays

The ndarray is an object that provide a python array
interface to data in memory.

It often happens that the memory that you want to view with an array
is not of the same byte ordering as the computer on which you are
running Python.

For example, I might be working on a computer with an little-endian
CPU - such as an Intel Pentium, but I have loaded some data
from a file written by a computer that is big-endian. Let's say
I have loaded 4 bytes from a file written by Sun (big-endian)
computer. I know that these 4 bytes represent two 16-bit integers.
On a big-endian machine, a two-byte integer is stored with the
Most Significant Byte (MSB) first, and then the Least Significant
Byte (LSB). Thus the bytes are, in memory order:

1. MSB integer 1
2. LSB integer 1
3. MSB integer 2
4. LSB integer 2

Let's say the two integers were in fact 1 and 770. Because 770 = 256 * 3 + 2,
the 4 bytes in memory would contain respectively: 0, 1, 3, 2

'''
big_end_str = chr(0) + chr(1) + chr(3) + chr(2)
print(big_end_str)

'''
We might want to use an ndarray to access these integers.
In that case, we can create an array around this memory,
and tell numpy that there are two integers, and that they are
16 bit and big-endian.
'''
import numpy as np
big_end_str = np.ndarray(shape=(2,), dtype='>i2', buffer=big_end_str)
print(big_end_str[0])
print(big_end_str[1])

'''
Note the array dtype above of i>2. The > means 'big-endian' (< is little-endian)
and i2 means 'signed 2-byte integer'. For example, if our data
represented a single unsigned 4-byte little-endian integer, the dtype
string would be <u4.
'''
little_end_u4 = np.ndarray(shape=(1,), dtype='<u4', buffer=big_end_str)
little_end_u4[0]= 1 * 256 ** 1 + 3 * 256 ** 2 + 2 * 256 ** 3

'''
Returning to our big_end_str - in this case our underlying data
is big-endian (data endianess) we've set the dtype to match
(the dtype is also big-endian). However, sometimes you need to flip
these around.
'''
