# This IO file works in the IDX file format

import tensorflow as tf
import gzip
import numpy
import functools as ft


'''
#sess = tf.InteractiveSession()
#sess = tf.get_default_session()
if sess == None:
  sess = tf.Session()
'''


# dtype doesn't actually do anything yet
def tensor_to_idx(var, filename, dtype='float'):
  data = numpy.array(tf.reshape(var,[-1]).eval())
  shape_list = list((map(lambda d: d.value, var.get_shape().dims)))
  dt = numpy.dtype('u4').newbyteorder('>')
  shape = numpy.array(shape_list, dtype=dt)
  with gzip.open(filename, 'wb') as f:
    f.write(bytes([0, 0, 3, len(shape)]))
    f.write(shape.tobytes())
    f.write(data.tobytes())

def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  buf = bytestream.read(4)
  return numpy.frombuffer(buf, dtype=dt)[0]

def idx_to_tensor(filename):
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
    tensor = tf.constant(data.tolist())
    tensor = tf.reshape(tensor, shape)
    return tensor
        
print("Imported Molasses")

'''
a = tf.constant([[1.0,2.0],[3.0,4.0]])
tensor_to_idx(a, 'a.gz')
b = idx_to_tensor('a.gz')
print(a.eval())
print(b.eval())
'''
