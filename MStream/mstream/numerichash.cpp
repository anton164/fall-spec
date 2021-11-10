#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

#include "numerichash.hpp"
#include <cmath>
#include <iostream>

using namespace std;
double EPSILONs= 0.001;

bool are_sames(double a, double b) {
    return fabs(a - b) < EPSILONs;
}

Numerichash::Numerichash(int r, int b) {
    num_rows = r;
    num_buckets = b;
    this->clear();
}

int Numerichash::hash(double cur_node) {
    int bucket;
    /*cout << "poop" << endl;
    cout << cur_node << endl;
    cout << "-------" << endl;
    if (are_sames(cur_node, 7.37379)) {
        cout << "cur_node after floor:" << cur_node << endl;
    }*/
    double tmp_cur_node = cur_node;
    cur_node = cur_node * (num_buckets - 1);
    /*if (are_sames(tmp_cur_node, 7.37379)) {
        cout << "cur_node after floor:" << cur_node << endl;
    }*/
    bucket = floor(cur_node);
    if (are_sames(tmp_cur_node, 7.37379)) {
        cout << "bucket:" << bucket << endl;
    }
    if(bucket < 0) 
        bucket = (bucket%num_buckets + num_buckets)%num_buckets;
    return bucket;
}

void Numerichash::insert(double cur_node, double weight) {
    int bucket;
    bucket = hash(cur_node);
    count[0][bucket] += weight;
}

double Numerichash::get_count(double cur_node) {
    int bucket;
    bucket = hash(cur_node);
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