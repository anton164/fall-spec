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

void save_min_max_per_features(string min_max_filename, vector<double> max_numeric_score, vector<double> min_numeric_score) {
    vector<string> tmp_file_name;
    char sep = '_';
    SplitString(min_max_filename, tmp_file_name, sep);
    std::ostringstream stream_max;
    std::ostringstream stream_min;
    for (int i = 0; i < max_numeric_score.size(); i++) {
        stream_max << max_numeric_score.at(i);
        stream_min << min_numeric_score.at(i);
        if (i + 1 != max_numeric_score.size()) {
            stream_max << ',';
            stream_min << ',';
        }
    }
    std::string string_max = stream_max.str();
    std::string string_min = stream_min.str();
    std::ofstream outfile_max;
    std::ofstream outfile_min;
    delete_file(tmp_file_name.at(0) + "_max.txt");
    delete_file(tmp_file_name.at(0) + "_min.txt");
    outfile_max.open(tmp_file_name.at(0) + "_max.txt", std::ios_base::app);
    outfile_min.open(tmp_file_name.at(0) + "_min.txt", std::ios_base::app);
    outfile_max << string_max;
    outfile_min << string_min;
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

vector<double> *mstream(vector<vector<double> > &numeric, vector<vector<long> > &categ, vector<int> &times, vector<int> &ignore, int num_rows,
                        int num_buckets, double factor, int smoothing_factor, int dimension1, int dimension2, 
                        vector<string> &scores_decomposed, vector<string> &scores_decomposed_p, string token_buckets_filename,
                        int abs_min_max, string columns_filename, int min_count, int hacked_lsh) { 
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
    vector<double> max_numeric_score(0);
    vector<double> min_numeric_score(0);

    // check if using hacked lsh, if yes, load max and min values
    if (hacked_lsh == 1) {
        vector<string> tmp_file_name;
        char sep = '_';
        SplitString(token_buckets_filename, tmp_file_name, sep);
        //load max min values
        ifstream infile;
        infile.open(tmp_file_name.at(0) + "_max.txt");
        string sLine;
        while (!infile.eof()){
            infile >> sLine;
        }
        infile.close();
        vector<string> tmp_max_string;
        sep = ',';
        SplitString(sLine.data(), tmp_max_string, sep);
        for (int i = 0; i < tmp_max_string.size(); i++) {
            max_numeric_score.push_back(atof(tmp_max_string.at(i).c_str()));
        }
        //load min values
        infile.open(tmp_file_name.at(0) + "_min.txt");
        while (!infile.eof()){
            infile >> sLine;
        }
        infile.close();
        vector<string> tmp_min_string;
        SplitString(sLine.data(), tmp_min_string, sep);
        for (int i = 0; i < tmp_min_string.size(); i++) {
            min_numeric_score.push_back(atof(tmp_min_string.at(i).c_str()));
        }
        //test
        for (int i = 0; i < min_numeric_score.size(); i++) {
            cout << min_numeric_score.at(i) << " min value \n";
        }
        for (int i = 0; i < max_numeric_score.size(); i++) {
            cout << max_numeric_score.at(i) << " max value \n";
        }
    }

    if (dimension1) {
        max_numeric.resize(dimension1, numeric_limits<double>::min());
        min_numeric.resize(dimension1, numeric_limits<double>::max());
        max_numeric_score.resize(dimension1, numeric_limits<double>::min());
        min_numeric_score.resize(dimension1, numeric_limits<double>::max());
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
                
                int bucket_index = numeric_score[node_iter].hash(cur_numeric[node_iter], hacked_lsh, max_numeric_score[node_iter], min_numeric_score[node_iter]);
                /*if (are_same(tmp_original_numeric, 1.82161)) {
                    cout << bucket_index << endl;
                }*/
                words_to_bucket.at(node_iter)[bucket_index].push_back(tmp_original_numeric);
                numeric_score[node_iter].insert(cur_numeric[node_iter], 1, hacked_lsh, max_numeric_score[node_iter], min_numeric_score[node_iter]);
                numeric_total[node_iter].insert(cur_numeric[node_iter], 1, hacked_lsh, max_numeric_score[node_iter], min_numeric_score[node_iter]);
                t = counts_to_anom(numeric_total[node_iter].get_count(cur_numeric[node_iter], hacked_lsh, max_numeric_score[node_iter], min_numeric_score[node_iter]),
                                numeric_score[node_iter].get_count(cur_numeric[node_iter], hacked_lsh, max_numeric_score[node_iter], min_numeric_score[node_iter]), cur_t,
                                smoothing_factor);
                scores_to_bucket.at(node_iter)[bucket_index].push_back(t);
                min_numeric_score[node_iter] = MIN(min_numeric_score[node_iter], t);
                max_numeric_score[node_iter] = MAX(max_numeric_score[node_iter], t);
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
    //Save max + min values into files for each numeric value
    if (hacked_lsh == 0) {
        save_min_max_per_features(token_buckets_filename, max_numeric_score, min_numeric_score);
    }
    return anom_score;
}
