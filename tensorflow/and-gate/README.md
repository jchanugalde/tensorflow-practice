# TensorFlow | Logical AND Gate
---
#### Author @ Ankur Kumar | [Medium Article](https://medium.com/towards-data-science/tensorflow-for-absolute-beginners-28c1544fb0d6)

#### Date: 23 October 2017

#### Notes
* Practice using TensorFlow and Python.
* This is code for a logical AND gate from Kumar's tutorial. I've included inline comments that emphasize terminology and function.
* Below are some challenges I encountered and their solutions. Kumar provides a thorough explanation, but I wanted to go through the code before reading it to get a sense of what I find unintuitive.
* Considering that I'm learning Python along the way, some of the challenges are really just questions about syntax. I've included them for personal reference.

#### Challenges & Solutions
* Syntax:
    * Variables: Declaration and instantiation are simultaneous in Python.  
      * Floats: To declare a float, include a decimal (e.g. 1. or 1.0).
      * I assume the precision ML requires means we'll typically be using floats.
* Kumar assigned 1. and -1. to T and F respectively. I understand that these values are arbitrary. As such, I used 1 and 0 (in keeping with discrete mathematics convention). I'm unsure if this should affect what the bias is (in part because I'm not yet sure what exactly the bias does). That is, should I also alter the bias value or is the bias value also arbitrary?
* tf.greater returns a tensor of type bool.
  * (1) What shape is a bool? Scalar? Vector?
    * print(sess.run(is_greater)) prints 'True' (for tf.greater(1,0))
    * print(sess.run(tf.shape(is_greater))) prints '[]' (i.e. scalar)
  * (2) There can only be two bool values (T & F). How do you turn a bool into a float? Am I the one who defines them (e.g. 1 and 0)? Or is there a default?
    ```
    >>> true = tf.greater(1,0)
    >>> false = tf.greater(0,1)
    >>> true_float = tf.to_float(true)
    >>> false_float = tf.to_float(false)
    >>> print(sess.run(true_float))
    1.0
    >>> print(sess.run(false_float))
    ```
