import math
import numpy as np
import random

RAND_MAX = 32767

class Recordhash:
    
    def __init__(self, r, b, dim1, dim2):
        self.num_rows = r
        self.num_buckets = b
        self.dimension1 = dim1
        self.dimension2 = dim2
        self.num_recordhash = np.zeros(self.num_rows)
        for i in range(self.num_rows):
            log_bucket = ceil(log2(self.num_buckets));
            self.num_recordhash[i] = np.zeros(log_bucket)
            for j in range(log_bucket):
                self.num_recordhash[i][j]= np.zeros(self.dimension1);
                for k in range(self.dimension1):
                    self.num_recordhash[i][j][k] = randNorm(1, 1);

        cat_recordhash.resize(num_rows);
        for (int i = 0; i < num_rows; i++) {
            cat_recordhash[i].resize(dimension2);
            for (int k = 0; k < dimension2 - 1; k++) {
                cat_recordhash[i][k] = (rand() % (num_buckets - 1) + 1);
            }
            if (dimension2)
                cat_recordhash[i][dimension2 - 1] = (rand() % num_buckets);
        }

        this->clear();