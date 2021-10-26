#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

#include <vector>
#include <iostream>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include "anom.hpp"
#include "numerichash.hpp"
#include "recordhash.hpp"
#include "categhash.hpp"

double counts_to_anom(double tot, double cur, int cur_t) {
    double cur_mean = tot / cur_t;
	double sqerr = pow(MAX(0, cur - cur_mean), 2);
    return sqerr / cur_mean + sqerr / (cur_mean * MAX(1, cur_t - 1));
}

vector<double> *mstream(vector<vector<double> > &numeric, vector<vector<long> > &categ, vector<int> &times, vector<int> &ignore, int num_rows,
                        int num_buckets, double factor, int dimension1, int dimension2, vector<string> &scores_decomposed, vector<string> &scores_decomposed_p) {

    int length = times.size(), cur_t = 1;

    Recordhash cur_count(num_rows, num_buckets, dimension1, dimension2);
    Recordhash total_count(num_rows, num_buckets, dimension1, dimension2);

    auto *anom_score = new vector<double>(length);

    vector<Numerichash> numeric_score(dimension1, Numerichash(num_rows, num_buckets));
    vector<Numerichash> numeric_total(dimension1, Numerichash(num_rows, num_buckets));
    vector<Categhash> categ_score(dimension2, Categhash(num_rows, num_buckets));
    vector<Categhash> categ_total(dimension2, Categhash(num_rows, num_buckets));

    vector<double> cur_numeric(0);
    vector<double> max_numeric(0);
    vector<double> min_numeric(0);
    if (dimension1) {
        max_numeric.resize(dimension1, numeric_limits<double>::min());
        min_numeric.resize(dimension1, numeric_limits<double>::max());
    }
    vector<long> cur_categ(0);

    for (int i = 0; i < length; i++) {
        std::vector< double > decomposed_scores;
        if (i == 0 || times[i] > cur_t) {
            //cout << factor;
            cur_count.lower(factor);
            for (int j = 0; j < dimension1; j++) {
                numeric_score[j].lower(factor);
            }
            for (int j = 0; j < dimension2; j++) {
                categ_score[j].lower(factor);
            }
            cur_t = times[i];
        }

        if (dimension1)
            cur_numeric.swap(numeric[i]);
        if (dimension2)
            cur_categ.swap(categ[i]);

        double sum = 0.0, t, cur_score;
        for (int node_iter = 0; node_iter < dimension1; node_iter++) {
            if (cur_numeric[node_iter] != 0) {
                cur_numeric[node_iter] = log10(1 + cur_numeric[node_iter]);
                if (!i) {
                    max_numeric[node_iter] = cur_numeric[node_iter];
                    min_numeric[node_iter] = cur_numeric[node_iter];
                    cur_numeric[node_iter] = 0;
                } else {
                    min_numeric[node_iter] = MIN(min_numeric[node_iter], cur_numeric[node_iter]);
                    max_numeric[node_iter] = MAX(max_numeric[node_iter], cur_numeric[node_iter]);
                    if (max_numeric[node_iter] == min_numeric[node_iter]) cur_numeric[node_iter] = 0;
                    else cur_numeric[node_iter] = (cur_numeric[node_iter] - min_numeric[node_iter]) /
                                    (max_numeric[node_iter] - min_numeric[node_iter]);
                }
                numeric_score[node_iter].insert(cur_numeric[node_iter], 1);
                numeric_total[node_iter].insert(cur_numeric[node_iter], 1);
                t = counts_to_anom(numeric_total[node_iter].get_count(cur_numeric[node_iter]),
                                numeric_score[node_iter].get_count(cur_numeric[node_iter]), cur_t);
            }
            else t = 0;
            decomposed_scores.push_back(t);
            sum = sum+t;
        }
        cur_count.insert(cur_numeric, cur_categ, 1);
        total_count.insert(cur_numeric, cur_categ, 1);

        for (int node_iter = 0; node_iter < dimension2; node_iter++) {
            categ_score[node_iter].insert(cur_categ[node_iter], 1);
            categ_total[node_iter].insert(cur_categ[node_iter], 1);
            if (cur_categ[node_iter] == 0) {
                t = 0;
            } else {
                t = counts_to_anom(categ_total[node_iter].get_count(cur_categ[node_iter]),
                                   categ_score[node_iter].get_count(cur_categ[node_iter]), cur_t);                
            }
            decomposed_scores.push_back(t);
            sum = sum+t;
        }
        if (ignore[i] == 0) {
            cur_score = counts_to_anom(total_count.get_count(cur_numeric, cur_categ),
                                       cur_count.get_count(cur_numeric, cur_categ), cur_t);
        } else {
            cur_score = 0;
        }
        sum = sum + cur_score;
        std::ostringstream stream;
        std::ostringstream stream_2;
        stream << log(1 + cur_score) << ',';
        if (cur_score == 0) {
            stream_2 << cur_score << ',';
        } else {
            stream_2 << cur_score/sum << ',';
        }
        
        for (int i = 0; i < decomposed_scores.size(); i++) {
            stream << std::to_string(log(1 + decomposed_scores[i]));
            if (sum == 0) {
                stream_2 << std::to_string(0);
            } else {
                stream_2 << std::to_string(decomposed_scores[i]/sum);
            }
            if (i != decomposed_scores.size() - 1) {
                stream << ',';
                stream_2 << ',';
            }
        }
        std::string string_decomposed_scores = stream.str();
        std::string string_decomposed_p_scores = stream_2.str();
        (*anom_score)[i] = log(1 + sum);
        (scores_decomposed)[i] = string_decomposed_scores;
        (scores_decomposed_p)[i] = string_decomposed_p_scores;
    }
    return anom_score;
}
