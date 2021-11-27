#ifndef numerichash_hpp
#define numerichash_hpp

#include <cstdio>
#include <vector>

class Numerichash {
public:
    Numerichash(int r, int b);

    void insert(double cur_node, double weight, int hacked_lsh, double max, double min);

    double get_count(double cur_node, int hacked_lsh, double max, double min);

    void clear();

    void lower(double factor);

    int hash(double cur_node, int hacked_lsh, double max, double min);

private:
    int num_rows;
    int num_buckets;
    std::vector<std::vector<double> > count;

};

#endif /* numerichash_hpp */
