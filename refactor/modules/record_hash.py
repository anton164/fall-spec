import math
import numpy as np
import random
from helpers import randNorm

RAND_MAX = 32767


class Recordhash:
    def __init__(self, r, b, dim1, dim2):
        self.num_rows = r
        self.num_buckets = b
        self.dimension1 = dim1
        self.dimension2 = dim2
        self.num_recordhash = np.zeros(self.num_rows)
        self.cat_recordhash = np.zeros(self.num_rows)
        self.count = np.zeros((self.num_rows, self.num_buckets))
        for i in range(self.num_rows):
            log_bucket = math.ceil(math.log2(self.num_buckets))
            self.num_recordhash[i] = np.zeros(log_bucket)
            for j in range(log_bucket):
                self.num_recordhash[i][j] = np.zeros(self.dimension1)
                for k in range(self.dimension1):
                    self.num_recordhash[i][j][k] = randNorm(1, 1)

        for i in range(self.num_rows):
            self.cat_recordhash[i] = np.zeros(self.dimension2)
            for k in range(self.dimension2 - 1):
                self.cat_recordhash[i][k] = (
                    random.randint(0, RAND_MAX) % (self.num_buckets - 1) + 1
                )
            if self.dimension2:
                self.cat_recordhash[i][self.dimension2 - 1] = (
                    random.randint(0, RAND_MAX) % self.num_buckets
                )

    def numerichash(self, cur_numeric, i):
        sum = 0
        bitcounter = 0
        log_bucket = math.ceil(math.log2(self.num_buckets))
        b = np.zeros(30)

        for iter in range(log_bucket):
            sum = 0
            for k in range(self.dimension):
                sum = sum + self.num_recordhash[i][iter][k] * cur_numeric[k]

            if sum < 0:
                b[bitcounter] = 0
            else:
                b[bitcounter] = 1
            bitcounter += 1
        return int("".join([str(int(item)) for item in b]), 2)

    def categhash(self, cur_categ, i):
        counter = 0
        resid = 0
        for k in range(self.dimension2):
            resid = (
                resid + self.cat_recordhash[i][counter] * cur_categ[counter]
            ) % self.num_buckets
            counter += 1
        return resid + (self.num_buckets if resid < 0 else 0)

    def insert(self, cur_numeric, cur_categ, weight):
        for i in range(self.num_rows):
            bucket1 = self.numerichash(cur_numeric, i)
            bucket2 = self.categhash(cur_categ, i)
            bucket = (bucket1 + bucket2) % self.num_buckets
            self.count[i][bucket] += weight

    def get_count(self, cur_numeric, cur_categ):
        min_count = float("inf")
        for i in range(self.num_rows):
            bucket1 = self.numerichash(cur_numeric, i)
            bucket2 = self.categhash(cur_categ, i)
            bucket = (bucket1 + bucket2) % self.num_buckets
            min_count = min(min_count, self.count[i][bucket])
        return min_count

    def lower(self, factor):
        self.count = self.count * factor

    def clear(self):
        self.count = np.zeros((self.num_rows, self.num_buckets))
