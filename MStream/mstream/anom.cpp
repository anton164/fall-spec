#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

#include <vector>
#include <iostream>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include "anom.hpp"
#include "numerichash.hpp"
#include "recordhash.hpp"
#include "categhash.hpp"

double EPSILON = 0.001;

double counts_to_anom(double tot, double cur, int cur_t, int smoothing_factor) {
    // tot = total counts, no decay
    // cur = total counts with decay
    // cur_t = time elapsed 
    // smoothing factor
    double cur_mean = MAX(tot / cur_t, smoothing_factor);
	double sqerr = pow(MAX(0, cur - cur_mean), 2);
    return sqerr / cur_mean + sqerr / (cur_mean * MAX(1, cur_t - 1));
}

bool are_same(double a, double b) {
    return fabs(a - b) < EPSILON;
}

void save_token_buckets(string token_buckets_filename, vector<vector<double>> words_to_bucket, vector<vector<double>> scores_to_bucket) {
    //write here
    std::ostringstream stream;
    stream << '[';
    for (int i = 0; i < words_to_bucket.size(); i++) {
        stream << '[';
        for (int j = 0; j < words_to_bucket.at(i).size(); j++) {
            stream << "(" << words_to_bucket.at(i).at(j) << "," << scores_to_bucket.at(i).at(j) << ")";
            if (j + 1 != words_to_bucket.at(i).size()) {
                stream << ',';
            }
        }
        stream << ']';
        if (i + 1 != words_to_bucket.size()) {
            stream << ',';
        }
    }
    stream << ']' << endl;
    std::string string_words_to_bucket = stream.str();
    std::ofstream outfile;
    outfile.open(token_buckets_filename, std::ios_base::app);
    outfile << string_words_to_bucket;
}

void delete_file(string filename) {
    try {
        if (std::filesystem::remove(filename)) {
            std::cout << "file " << filename << " deleted.\n";
        }
        else {
            std::cout << "file " << filename << " not found.\n";
        }
    }
    catch(const std::filesystem::filesystem_error& err) {
        std::cout << "filesystem error: " << err.what() << '\n';
    }
}

vector<double> find_max_numeric(int dimension1, vector<vector<double>> &numeric) {
    vector<double> max_numeric(dimension1, -1*numeric_limits<int>::max());
    for (int i = 0; i < numeric.size(); i++) {
        for (int j = 0; j < numeric.at(i).size(); j++) {
            if (max_numeric.at(j) < numeric.at(i).at(j)) {
                max_numeric.at(j) = numeric.at(i).at(j);
            }
        }   
    }
    return max_numeric;
}

vector<double> find_min_numeric(int dimension1, vector<vector<double>> &numeric) {
    vector<double> min_numeric(dimension1, numeric_limits<int>::max());
    for (int i = 0; i < numeric.size(); i++) {
        for (int j = 0; j < numeric.at(i).size(); j++) {
            if (min_numeric.at(j) > numeric.at(i).at(j)) {
                min_numeric.at(j) = numeric.at(i).at(j);
            }
        }   
    }
    return min_numeric;
}

void SplitString(string s, vector<string> &v, char sep){
	string temp = "";
	for(int i=0;i<s.length();++i){
		if(s[i]==sep){
			v.push_back(temp);
			temp = "";
		}
		else{
			temp.push_back(s[i]);
		}		
	}
	v.push_back(temp);
}

vector<double> *mstream(vector<vector<double> > &numeric, vector<vector<long> > &categ, vector<int> &times, vector<int> &ignore, int num_rows,
                        int num_buckets, double factor, int smoothing_factor, int dimension1, int dimension2, 
                        vector<string> &scores_decomposed, vector<string> &scores_decomposed_p, string token_buckets_filename,
                        int abs_min_max, string columns_filename, int min_count) {
    int length = times.size(), cur_t = 1;
    // get column names
    ifstream infile(columns_filename);
    string sLine;
    if (infile.good()) {
        getline(infile, sLine);
    }
    vector<string> columns;
    char sep = ',';
	SplitString(sLine, columns, sep);
    delete_file(token_buckets_filename);
    for (int k = 1; k < columns.size(); k++) {
        vector<string> tmp_file_name;
        char sep = '.';
        SplitString(token_buckets_filename, tmp_file_name, sep);
        delete_file(tmp_file_name.at(0)+"_"+columns.at(k)+".txt");
    }
    Recordhash cur_count(num_rows, num_buckets, dimension1, dimension2);
    Recordhash total_count(num_rows, num_buckets, dimension1, dimension2);
    
    auto *anom_score = new vector<double>(length);
    vector<Numerichash> numeric_score(dimension1, Numerichash(num_rows, num_buckets));
    vector<Numerichash> numeric_total(dimension1, Numerichash(num_rows, num_buckets));
    vector<Categhash> categ_score(dimension2, Categhash(num_rows, num_buckets));
    vector<Categhash> categ_total(dimension2, Categhash(num_rows, num_buckets));
    vector<vector<vector<double>>> words_to_bucket(dimension1+dimension2, vector<vector<double>>(num_buckets, vector<double>(0, 0)));
    vector<vector<vector<double>>> scores_to_bucket(dimension1+dimension2, vector<vector<double>>(num_buckets, vector<double>(0, 0)));
    
    vector<double> abs_max_numeric = find_max_numeric(dimension1, numeric);
    vector<double> abs_min_numeric = find_min_numeric(dimension1, numeric);

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
            cur_count.lower(factor);
            for (int j = 0; j < dimension1; j++) {
                numeric_score[j].lower(factor);
            }
            for (int j = 0; j < dimension2; j++) {
                categ_score[j].lower(factor);
            }

            for (int i = 0; i < words_to_bucket.size(); i++) {
                vector<string> tmp_file_name;
                char sep = '.';
                SplitString(token_buckets_filename, tmp_file_name, sep);
                save_token_buckets(tmp_file_name.at(0)+"_"+columns.at(i+1)+".txt", words_to_bucket.at(i), scores_to_bucket.at(i));
            }
            vector<vector<vector<double>>> tmp_1(dimension1+dimension2, vector<vector<double>>(num_buckets, vector<double>(0, 0)));
            vector<vector<vector<double>>> tmp_2(dimension1+dimension2, vector<vector<double>>(num_buckets, vector<double>(0, 0)));
            words_to_bucket = tmp_1;
            scores_to_bucket = tmp_2;
            cur_t = times[i];
        }

        if (dimension1)
            cur_numeric.swap(numeric[i]);
        if (dimension2)
            cur_categ.swap(categ[i]);

        double sum = 0.0, t, cur_score;
        for (int node_iter = 0; node_iter < dimension1; node_iter++) {
            if (cur_numeric[node_iter] != 0) {
                double tmp_original_numeric = cur_numeric[node_iter];
                cur_numeric[node_iter] = log10(1 + cur_numeric[node_iter]);
                if (abs_min_max != 1) {
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
                } else {
                    if (abs_max_numeric[node_iter] == abs_min_numeric[node_iter]) cur_numeric[node_iter] = 0;
                    else cur_numeric[node_iter] = (cur_numeric[node_iter] - abs_min_numeric[node_iter]) /
                                    (abs_max_numeric[node_iter] - abs_min_numeric[node_iter]);
                }
                
                int bucket_index = numeric_score[node_iter].hash(cur_numeric[node_iter]);
                /*if (are_same(tmp_original_numeric, 1.82161)) {
                    cout << bucket_index << endl;
                }*/
                words_to_bucket.at(node_iter)[bucket_index].push_back(tmp_original_numeric);
                numeric_score[node_iter].insert(cur_numeric[node_iter], 1);
                numeric_total[node_iter].insert(cur_numeric[node_iter], 1);
                t = counts_to_anom(numeric_total[node_iter].get_count(cur_numeric[node_iter]),
                                numeric_score[node_iter].get_count(cur_numeric[node_iter]), cur_t,
                                smoothing_factor);
                scores_to_bucket.at(node_iter)[bucket_index].push_back(t);
            }
            else t = 0;
            decomposed_scores.push_back(t);
            sum = sum+t;
        }
        cur_count.insert(cur_numeric, cur_categ, 1);
        total_count.insert(cur_numeric, cur_categ, 1);

        for (int node_iter = 0; node_iter < dimension2; node_iter++) {
            int tmp_original_categoric = cur_categ[node_iter];
            categ_score[node_iter].insert(cur_categ[node_iter], 1);
            categ_total[node_iter].insert(cur_categ[node_iter], 1);
            if (cur_categ[node_iter] == 0 || categ_score[node_iter].get_count(cur_categ[node_iter]) <= min_count) {
                t = 0;
            } else {
                t = counts_to_anom(categ_total[node_iter].get_count(cur_categ[node_iter]),
                                   categ_score[node_iter].get_count(cur_categ[node_iter]), cur_t,
                                   smoothing_factor);                
            }
            int bucket_index = categ_score[node_iter].get_bucket(tmp_original_categoric);
            words_to_bucket.at(dimension1+node_iter)[bucket_index].push_back(tmp_original_categoric);
            decomposed_scores.push_back(t);
            scores_to_bucket.at(dimension1+node_iter)[bucket_index].push_back(t);
            sum = sum+t;
        }
        if (ignore[i] == 0) {
            cur_score = counts_to_anom(total_count.get_count(cur_numeric, cur_categ),
                                       cur_count.get_count(cur_numeric, cur_categ), cur_t,
                                       smoothing_factor);
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
