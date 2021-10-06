#ifndef anom_hpp
#define anom_hpp

#include <cstdio>
#include <vector>

using namespace std;

vector<double> *
mstream(vector<vector<double> > &numeric, vector<vector<long> > &categ, vector<int> &times, vector<int> &ignore, int num_rows,
        int num_buckets, double factor, int dimension1, int dimension2);

#endif /* anom_hpp */
