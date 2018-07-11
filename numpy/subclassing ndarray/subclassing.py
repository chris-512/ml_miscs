#!/usr/bin/env python3

import numpy as np

'''
Subclasssing ndarray is relatively simple, but it has some
complications compard to other Python objects. On this page,
we explain the machinery that allos you to subclass ndarray,
and the implications for implementing a subclass.
'''

'''View casting

View casting is the standard ndarray mechanism by which you take
an ndarray of any subclass, and return a view of the array as
another (specified) subclass:

'''

# create a completely useless ndarray subclass
class C(np.ndarray): pass
# create a standard ndarray
arr = np.zeros((3, ))
# take a view of it, as our useless subclass
c_arr = arr.view(C)
type(c_arr)

'''Creating new from template

New instances of an ndarray subclass can also come about by a very
similar mechanism of View casting, when numpy finds it needs 
to create a new instance from a template instance. The most obvious
place this has to happen is when you are taking slices of subclassed
arrays. For example:
'''
v = c_arr[1:]

'''Implications for subclassing

If we subclass ndarray, we need to deal not only with explicit
construction of our array type, but also View casting or Creating
new from template. NumPy has the machinery to do this,
and this machinery that makes subclassing slightly non-standard.

There are two aspects to the machinery that ndarray uses to support
views and new-from-template subclasses.

The first is the use of ndarray.__new__ method for the main work
of object initialization, rather than the more usual
__init__ method. The second is the use of __array_finalize__ method
to allow subclasses to clean up after the creation of views 
and new instances from templates.

A brief Python primer on __new__ and __init__

__new__ is a standard Python method and, if present, is called
before __init__ when we create a class instance. 

'''
class C(object):
    def __new__(cls, *args):
        print('Cls in __new__:', cls)
        print('Args in __new__:', args)
        return object.__new__(cls, *args)

    def __init__(self, *args):
        print('type(self) in __init__:', type(self))
        print('Args in __init__:', args)

c = C('hello')
print(c)

'''
After python calls __new__, it usually (see below)
calls our __init__ method, with the output of __new__ as the
first argument (now a class instance), and the passed arguments
following.

As you can see, the object can be initialized in the __new__ method
or the __init__ method, or both, and in fact ndarray does 
not have an __init__ method, because all the initialization is done
in the __new__ method.

Why use __new__ rather than just the usual __init__?
Because in some cases, as for ndarray, we want to be able to return
an object of some other class. Consider the following:
'''


class D(C):
    def __new__(cls, *args):
        print('D cls is:', cls)
        print('D args in __new__', args)
        return C.__new__(C, *args)

    def __init__(self, *args):
        # we never get here
        print('In D __init__')

obj = D('hello')

'''
The definition of C is the same as before, but for D, 
the __new__ method returns an instance of class C rather than D. 
Note that the __init__ method of D does not get called. 
In general, when the __new__ method returns an object of class 
other than the class in which it is defined, 
the __init__ method of that class is not called.

'''

'''The role of __array_finalize__

'''
import numpy as np

class C(np.ndarray):
    def __new__(cls, *args, **kwargs):
        print('In __new__ with class %s' % cls)
        return super(C, cls).__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        # in practice you probably will not need or want an __init__
        # method for your subclass
        print('In __init__ with class %s' % self.__class__)

    def __array_finalize__(self, obj):
        print('In array_finalize:')
        print('   self type is %s' % type(self))
        print('   obj type is %s' % type(obj))


# Explicit constructor
c = C((10, ))
print(c)
# View casting
a = np.arange(10)
cast_a = a.view(C)
# Slicing (example of new-from-template)
cv = c[:1]

