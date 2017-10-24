import tensorflow as tf # Import library

T, F = 1., 0 # Assign values to true (1) and false (0)

bias = 1.0

# Rank 2 tensor (matrix) with shape [4,3]
training_input = [
    [T, T, bias],
    [T, F, bias],
    [F, T, bias],
    [F, F, bias],
]

# Rank 2 tensor (matrix) with shape [4,1]
# Only value at [1,1] is T because AND gate requires two T values
training_output = [
    [T],
    [F],
    [F],
    [F],
]

# Assign varible to w. Variables differ from placeholders in that they can be 'fed'
# in. tf.random_normal creates a random tensor with a given shape of given type.
# So in this case, w is a matrix with three rows and one column
w = tf.Variable(tf.random_normal([3,1]), float32)

# step(x) = { 1 if x > 0; -1 otherwise }
def step(x):
    is_greater = tf.greater(x, 0) # tf.greater returns tensor of type boolean
    as_float = tf.to_float(is_greater) # returns tensor with same shape as x
    doubled = tf.multiply(as_float, 2) 
