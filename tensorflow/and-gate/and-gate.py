import tensorflow as tf # Import library

# Get path of the current dir, then use it to create paths:
CURRENT_DIR = os.path.dirname(__file__)
file_path = os.path.join(CURRENT_DIR, 'test.txt')

# Then work using the absolute paths:
f = open(file_path,'w')
f.write('testing the script')

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
# These are going to be the weighted connections between each input value (remember
# that there are four). We need 3 rows if we want to perform matrix multiplication
# in line 40.
w = tf.Variable(tf.random_normal([3,1]), float32)

# step(x) = { 1 if x > 0; -1 otherwise }
def step(x):
    is_greater = tf.greater(x, 0) # tf.greater returns tensor of type boolean
    as_float = tf.to_float(is_greater) # returns tensor with same shape as x
    doubled = tf.multiply(as_float, 2)
    return tf.subtract(doubled - 1)

# Matrix multiplication
output = step(tf.matmul(training_input, w))
# Calculate difference between output and training input
error = tf.subtract(training_output, output)
# Calculate linear regression; overall, how far off was the NN output?
mse = tf.reduce_mean(tf.squared(error))

# Tells us how much we should adjust the weights
delta = tf.matmul(training_input, error, transpose_a=True)
# New weights
train = tf.assign(w, tf.add(w, delta))

# Start session and initialize vars
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Assume error is 1; target error is 0
err, target = 1, 0
# How many times do we iterate or train the network
epoch, max_epochs = 0, 10

# While we still have an error and we haven't passed 10 epochs, keep on training
while err > target and epoch < max_epochs:
    epoch += 1
    err, _ = sess.run([mse, train])
