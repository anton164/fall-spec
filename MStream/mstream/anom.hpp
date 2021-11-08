#ifndef anom_hpp
#define anom_hpp

#include <cstdio>
#include <vector>

using namespace std;

vector<double> *
mstream(vector<vector<double> > &numeric, vector<vector<long> > &categ, vector<int> &times, vector<int> &ignore, int num_rows,
        int num_buckets, double factor, int smoothing_factor, int dimension1, int dimension2, 
        vector<string> &scores_decomposed, vector<string> &scores_decomposed_p, string token_buckets_filename);

#endif /* anom_hpp */
