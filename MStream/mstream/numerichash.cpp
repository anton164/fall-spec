#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

#include "numerichash.hpp"
#include <cmath>
#include <iostream>

using namespace std;

Numerichash::Numerichash(int r, int b) {
    num_rows = r;
    num_buckets = b;
    this->clear();
}

int Numerichash::hash(double cur_node, int hacked_lsh, double max, double min) {
    int bucket;
    if (hacked_lsh == 1) {
        bucket = floor(cur_node/((max-min)/num_buckets));
        cout << "bucket reeturned for " << cur_node << "\n";
        cout << bucket << "\n";
    } else {
        cur_node = cur_node * (num_buckets - 1);
        bucket = floor(cur_node);
        if(bucket < 0)
            bucket = (bucket%num_buckets + num_buckets)%num_buckets;
    }
    return bucket;
}

void Numerichash::insert(double cur_node, double weight, int hacked_lsh, double max, double min) {
    int bucket;
    bucket = hash(cur_node, hacked_lsh, max, min);
    count[0][bucket] += weight;
}

double Numerichash::get_count(double cur_node, int hacked_lsh, double max, double min) {
    int bucket;
    bucket = hash(cur_node, hacked_lsh, max, min);
    return count[0][bucket];
}

void Numerichash::clear() {
    count = vector<vector<double> >(num_rows, vector<double>(num_buckets, 0.0));
}

void Numerichash::lower(double factor) {
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_buckets; j++) {
            count[i][j] = count[i][j] * factor;
        }
    }
}