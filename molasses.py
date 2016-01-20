# This IO file works in the IDX file format

import tensorflow as tf
import gzip
import numpy
import functools as ft

def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  buf = bytestream.read(4)
  return numpy.frombuffer(buf, dtype=dt)[0]

# dtype doesn't actually do anything yet
def tensor_to_idx(var, filename, dtype='float'):
  array_to_idx(var.eval(), filename)

def idx_to_tensor(filename):
  return tf.constant(idx_to_array(filename))

def array_to_idx(var, filename, dtype='f4'):
  data = numpy.reshape(var,[-1]).astype(dtype)
  dt = numpy.dtype('u4').newbyteorder('>')
  shape = numpy.array(var.shape, dtype=dt)
  with gzip.open(filename, 'wb') as f:
    f.write(bytes([0, 0, 0xd, len(shape)]))
    f.write(shape.tobytes())
    f.write(data.tobytes())

def idx_to_array(filename):
  with gzip.open(filename, 'rb') as bytestream:
    magic = _read32(bytestream)
    # magic & 0xff == number of dimensions
    shape = []
    # dt will need to be changed eventually
    dt = numpy.dtype('f4')
    for i in range(magic & 0xff):
      shape.append(_read32(bytestream))
    buf = bytestream.read(dt.itemsize * ft.reduce(lambda x,y: x*y, shape))
    data = numpy.frombuffer(buf, dtype=dt)
    array = data.tolist()
    array = numpy.reshape(array, shape).astype(dt)
    return array
