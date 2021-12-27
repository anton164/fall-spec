import random
import math


def randNorm(mean, stddev):
    # Return a real number from a normal (Gaussian) distribution with given
    # mean and standard deviation by polar form of Box-Muller transformation
    x = 2.0 * random.randint(0, RAND_MAX) - 1.0
    y = 2.0 * random.randint(0, RAND_MAX) - 1.0
    r = x * x + y * y
    while r >= 1.0 or r == 0.0:
        x = 2.0 * random.randint(0, RAND_MAX) - 1.0
        y = 2.0 * random.randint(0, RAND_MAX) - 1.0
        r = x * x + y * y
    s = math.sqrt(-2.0 * log(r) / r)
    return mean + x * s * stddev
