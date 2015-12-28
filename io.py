# This IO file works in the IDX file format

# import array as arr
import tensorflow as tf
import gzip
import numpy

sess = tf.InteractiveSession()


# dtype doesn't actually do anything yet
def tensor_to_idx(var, filename, dtype='float'):
  data = numpy.array(tf.reshape(var,[-1]).eval())
  shape_list = list((map(lambda d: d.value, var.get_shape().dims)))
  dt = numpy.dtype('u4').newbyteorder('>')
  shape = numpy.array(shape_list, dtype=dt)
  with gzip.open(filename, 'wb') as f:
    f.write(bytes([0, 0, len(shape), 3]))
    shape.tofile(f)
    data.tofile(f)

'''
def idx_to_tensor(filename):
  
  tensor = tf.constant(flat_array)
  return 
'''
        

a = tf.constant([[1.0,2.0],[3.0,4.0]])
tensor_to_idx(a, 'a.gz')
