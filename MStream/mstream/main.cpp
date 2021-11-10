#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include "anom.hpp"
#include "argparse.hpp"

using namespace std;

void load_data(vector<vector<double> > &numeric, vector<vector<long> > &categorical, vector<int> &times, vector<int> &ignore,
               const string &numeric_filename, const string &categ_filename, const string &time_filename, const string &ignore_filename) {
    int l = 0;
    string s, line;
    if (!numeric_filename.empty()) {
        ifstream numericFile(numeric_filename);
        while (numericFile) {
            l++;
            if (!getline(numericFile, s))
                break;
            if (s[0] != '#') {
                istringstream ss(s);
                vector<double> record;
                while (ss) {

                    if (!getline(ss, line, ',')) {
                        break;
                    }
                    try {
                        record.push_back(stod(line));
                    }
                    catch (const std::invalid_argument &e) {
                        cout << "NaN found in file " << numeric_filename << " line " << l
                             << endl;
                        e.what();
                    }
                }
                numeric.push_back(record);
            }
        }
        if (!numericFile.eof()) {
            cerr << "Could not read file " << numeric_filename << "\n";
            __throw_invalid_argument("File not found.");
        }
    }
    if (!categ_filename.empty()) {
        ifstream categFile(categ_filename);
        l = 0;
        while (categFile) {
            l++;
            if (!getline(categFile, s))
                break;
            if (s[0] != '#') {
                istringstream ss(s);
                vector<long> record;
                while (ss) {
                    if (!getline(ss, line, ','))
                        break;
                    try {
                        record.push_back(stol(line));
                    }
                    catch (const std::invalid_argument &e) {
                        cout << "NaN found in file " << categ_filename << " line " << l
                             << endl;
                        e.what();
                    }
                }
                categorical.push_back(record);
            }
        }
        if (!categFile.eof()) {
            cerr << "Could not read file " << categ_filename << "\n";
            __throw_invalid_argument("File not found.");
        }
    }
    ifstream timeFile(time_filename);
    l = 0;
    while (timeFile) {
        l++;
        if (!getline(timeFile, s))
            break;
        if (s[0] != '#') {
            istringstream ss(s);
            while (ss) {
                if (!getline(ss, line, ','))
                    break;
                try {
                    times.push_back(stoi(line));
                }
                catch (const std::invalid_argument &e) {
                    cout << "NaN found in file " << time_filename << " line " << l
                         << endl;
                    e.what();
                }
            }
        }
    }
    if (!timeFile.eof()) {
        cerr << "Could not read file " << time_filename << "\n";
        __throw_invalid_argument("File not found.");
    }
    ifstream ignoreFile(ignore_filename);
    l = 0;
    while (ignoreFile) {
        l++;
        if (!getline(ignoreFile, s))
            break;
        if (s[0] != '#') {
            istringstream ss(s);
            while (ss) {
                if (!getline(ss, line, ','))
                    break;
                try {
                    ignore.push_back(stoi(line));
                }
                catch (const std::invalid_argument &e) {
                    cout << "NaN found in file " << ignore_filename << " line " << l
                         << endl;
                    e.what();
                }
            }
        }
    }
    if (!ignoreFile.eof()) {
        cerr << "Could not read file " << ignore_filename << "\n";
        __throw_invalid_argument("File not found.");
    }
}

int main(int argc, const char *argv[]) {
    argparse::ArgumentParser program("mstream");
    program.add_argument("-n", "--numerical")
            .default_value(string(""))
            .help("Numerical Data File");
    program.add_argument("-c", "--categorical")
            .default_value(string(""))
            .help("Categorical Data File");
    program.add_argument("-t", "--times")
            .required()
            .help("Timestamp Data File");
    program.add_argument("-i", "--ignore")
            .required()
            .help("Ignore file is required");
    program.add_argument("-d", "--decompose")
            .required()
            .help("Decomposed scores file is required");
    program.add_argument("-dp", "--decompose")
            .required()
            .help("Decomposed percentages scores file is required");
    program.add_argument("-tb", "--token_buckets")
            .required()
            .help("Token buckets filename file is required");
    program.add_argument("-r", "--rows")
            .default_value(2)
            .action([](const std::string &value) { return std::stoi(value); })
            .help("Number of rows. Default is 2");
    program.add_argument("-b", "--buckets")
            .default_value(1024)
            .action([](const std::string &value) { return std::stoi(value); })
            .help("Number of buckets. Default is 1024");
    program.add_argument("-a", "--alpha")
            .default_value(0.8)
            .action([](const std::string &value) { return std::stod(value); })
            .help("Alpha: Temporal Decay Factor. Default is 0.8");
    program.add_argument("-absminmax", "--absminmax")
            .default_value(0)
            .action([](const std::string &value) { return std::stoi(value); })
            .help("Abs min max, should the max and min value should be searched in stream fashion or a priori. Default is 1");
    program.add_argument("-beta", "--beta")
            .default_value(0)
            .action([](const std::string &value) { return std::stoi(value); })
            .help("Beta: Anomaly Scoring Smoothing Term. Default is 0");
    program.add_argument("-o", "--output").default_value(string("scores.txt")).help(
            "Output File. Default is scores.txt");
    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err) {
        std::cout << err.what() << std::endl;
        program.print_help();
        exit(1);
    }

    string numeric_filename = program.get<string>("-n");
    string categ_filename = program.get<string>("-c");
    string times_filename = program.get<string>("-t");
    string output_filename = program.get<string>("-o");
    string ignore_filename = program.get<string>("-i");
    string decomposed_scores_filename = program.get<string>("-d");
    string decomposed_scores_p_filename = program.get<string>("-dp");
    string token_buckets_filename = program.get<string>("-tb");
    int abs_min_max = program.get<int>("-absminmax");
    int rows = program.get<int>("-r");
    int buckets = program.get<int>("-b");
    int beta = program.get<int>("-beta");
    auto alpha = program.get<double>("-a");

    if (rows < 1) {
        cerr << "Number of numerichash functions should be positive.\n";
        exit(1);
    }

    if (beta < 0) {
        cerr << "Smoothing term can't be negative.\n";
        exit(1);
    }

    if (buckets < 2) {
        cerr << "Number of buckets should be at least 2\n";
        exit(1);
    }

    if (alpha <= 0 || alpha >= 1) {
        cerr << "Alpha: Temporal Decay Factor must be between 0 and 1.\n";
        exit(1);
    }

    if (numeric_filename.empty() && categ_filename.empty()) {
        cerr << "Please give at least one of numeric or categorical data file\n";
        exit(1);
    }

    vector<vector<double> > numeric;
    vector<vector<long> > categ;
    vector<int> times;
    vector<int> ignore;
    int dimension1 = 0, dimension2 = 0;
    load_data(numeric, categ, times, ignore, numeric_filename, categ_filename, times_filename, ignore_filename);
    if (!numeric.empty())
        dimension1 = numeric[0].size();
    if (!categ.empty())
        dimension2 = categ[0].size();
    if ((dimension1 && times.size() != numeric.size()) || (dimension2 && times.size() && ignore.size() != categ.size())) {
        cout << (dimension2 && times.size() != categ.size());
        exit(1);
    }

    clock_t start_time2 = clock();
    int length = times.size();
    //vector<double> scores_decomposed = new vector<double>(length);
    std::vector<string> scores_decomposed(length);
    std::vector<string> scores_decomposed_p(length);
    vector<double> *scores2 = mstream(numeric, categ, times, ignore, rows, buckets, alpha, beta, dimension1, dimension2, scores_decomposed, scores_decomposed_p, token_buckets_filename, abs_min_max);
    cout << "@ " << ((double) (clock() - start_time2)) / CLOCKS_PER_SEC << endl;

    FILE *output_file = fopen(output_filename.c_str(), "w");
    FILE *decomposed_score_file = fopen(decomposed_scores_filename.c_str(), "w");
    FILE *decomposed_score_p_file = fopen(decomposed_scores_p_filename.c_str(), "w");
    for (double i : *scores2) {
        fprintf(output_file, "%f\n", i);
    }
    for (string i : scores_decomposed) {
        fprintf(decomposed_score_file, "%s\n", i.c_str());
    }
    for (string i : scores_decomposed_p) {
        fprintf(decomposed_score_p_file, "%s\n", i.c_str());
    }
    return 0;
}
