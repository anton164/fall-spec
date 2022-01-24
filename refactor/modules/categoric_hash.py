import numpy as np
import random

RAND_MAX = 32767


class Categorichash:
    def __init__(self, r, b):
        self.num_rows = r
        self.num_buckets = b
        self.hash_a = np.zeros(self.num_rows)
        self.hash_b = np.zeros(self.num_rows)
        a_func = lambda e: random.randint(0, RAND_MAX) % (self.num_buckets - 1) + 1
        b_func = lambda e: random.randint(0, RAND_MAX) % self.num_buckets
        vfunc_a = np.vectorize(a_func)
        vfunc_b = np.vectorize(b_func)
        self.hash_a = vfunc_a(self.hash_a)
        self.hash_b = vfunc_b(self.hash_b)
        self.count = np.zeros((self.num_rows, self.num_buckets))

    def hash(self, a, i):
        resid = (a * self.hash_a[i] + self.hash_b[i]) % self.num_buckets
        return resid + (self.num_buckets if resid < 0 else 0)

    def insert(self, cur_int, weight):
        for i in range(self.num_rows):
            bucket = hash(cur_int, i)
            self.count[i][bucket] += weight

    def get_count(self, cur_int):
        min_count = float("inf")
        for i in range(self.num_rows):
            bucket = hash(cur_int, i)
            min_count = min(min_count, self.count[i][bucket])
        return min_count

    def clear(self):
        self.count = np.zeros((self.num_rows, self.num_buckets))

    def lower(self, factor):
        self.count = self.count * factor

    def get_bucket(self, cur_int):
        min_count = float("inf")
        for i in range(self.num_rows):
            bucket = hash(cur_int, i)
            if self.count[i][bucket] < min_count:
                bucket_to_return = bucket
        return bucket_to_return
