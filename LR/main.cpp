#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <climits>


#include <cfloat>
#include <cstring>

#ifdef OFFLINE
#include <chrono>
#endif

using namespace std;


const int features_num = 1000;

struct Data {
    vector<double> features;
    int label;
    Data(vector<double> f, int l) : features(f), label(l)
    {}
};


class LR {
public:
    void train();

    void predict();

    LR(const string & train_file, const string & test_file, const string & predict_file);

private:
    vector<Data> train_data_;
    vector<Data> test_data_;
    vector<int> predict_vec;

    vector<double> weight_;
    vector<double> min_error_weight_;
    double min_error_rate_;
    int min_error_iter_no_;

    string weightParamFile = "modelweight.txt";

private:
    vector<Data> LoadData(const string & filename, bool last_label_exist ,int read_line_num);
    void LoadTrainData();

    void LoadTestData();

    void initParam();

    double wxbCalc(const Data &data, const vector<double> weight);
    double sigmoidCalc(const double wxb);
    double lossCal(const vector<double> & weight);

    double gradientSlope(const vector<Data> & dataSet, int index, const vector<double> & sigmoidVec);
    void StorePredict();

private:
    string train_file_;
    string test_file_;
    string predict_file_;
    const int read_line_num = 4000;
private:

    const double wtInitV = 1.0;

    const double rate_start = 0.9;
    const double decay = 0.05;
    const double rate_min = 0.02;

    const int maxIterTimes = 200;
    const double predictTrueThresh = 0.5;
    const int train_show_step = 1;
};


void LR::train() {

    LoadTrainData();

#ifdef OFFLINE
    chrono::steady_clock::time_point start_time = chrono::steady_clock::now();
#endif

    double sigmoidVal;
    double wxbVal;

    vector<double> sigmoidVec(train_data_.size(), 0.0);

    for (int i = 0; i < maxIterTimes; i++) {

        for (int j = 0; j < train_data_.size(); j++) {
            wxbVal = wxbCalc(train_data_[j], weight_);
            sigmoidVal = sigmoidCalc(wxbVal);
            sigmoidVec[j] = sigmoidVal;
        }

        double error_rate = 0.0;
        for (int j = 0; j < train_data_.size(); j++) {
            error_rate += pow(train_data_[j].label - sigmoidVec[j], 2);
        }

        if (error_rate < min_error_rate_) {
            min_error_rate_ = error_rate;
            min_error_weight_ = weight_;
            min_error_iter_no_ = i;
        }


        double rate = rate_start * 1.0 / (1.0 + decay * i);
        rate = max(rate, rate_min);

        for (int j = 0; j < weight_.size(); j++) {
            weight_[j] += rate * gradientSlope(train_data_, j, sigmoidVec);
        }

#ifdef OFFLINE
        if (i % train_show_step == 0) {
            cout << "iter: " << i << " error_rate: " << error_rate << " rate: " << rate;
//                cout << ". updated weight value is : ";
//                for (int j = 0; j < param.wtSet.size(); j++) {
//                    cout << param.wtSet[j] << "  ";
//                }
            cout << endl;
        }
#endif
    }


#ifdef OFFLINE
    chrono::steady_clock::time_point end_time = chrono::steady_clock::now();
    printf("模型训练时间（ms）: %f \n", chrono::duration<double, std::milli>(end_time - start_time).count());
#endif

}

void LR::predict() {

    LoadTestData();

#ifdef OFFLINE
    chrono::steady_clock::time_point start_time = chrono::steady_clock::now();
#endif

    double sigVal;
    int predictVal;

    for (int j = 0; j < test_data_.size(); j++) {
        sigVal = sigmoidCalc(wxbCalc(test_data_[j], min_error_weight_));
        predictVal = sigVal >= predictTrueThresh ? 1 : 0;
        predict_vec.push_back(predictVal);
    }

#ifdef OFFLINE
    chrono::steady_clock::time_point end_time = chrono::steady_clock::now();
    printf("模型预测时间（ms）: %f \n", chrono::duration<double, std::milli>(end_time - start_time).count());
#endif

    StorePredict();


#ifdef OFFLINE
    cout << "min_error_iter_no: " << min_error_iter_no_ << " min_error_rate: " << min_error_rate_ << endl;
#endif

}

LR::LR(const string & train_file, const string & test_file, const string & predict_file):
        train_file_(train_file),
        test_file_(test_file),
        predict_file_(predict_file),
        min_error_rate_(DBL_MAX),
        min_error_iter_no_(-1) {
    initParam();
}


void LR::LoadTrainData() {
#ifdef OFFLINE
    chrono::steady_clock::time_point start_time = chrono::steady_clock::now();
#endif
    train_data_ = LoadData(train_file_, true, read_line_num);
#ifdef OFFLINE
    chrono::steady_clock::time_point end_time = chrono::steady_clock::now();
    printf("训练集读取时间（ms）: %f \n", chrono::duration<double, std::milli>(end_time - start_time).count());
#endif
}

void LR::LoadTestData() {
#ifdef OFFLINE
    chrono::steady_clock::time_point start_time = chrono::steady_clock::now();
#endif
    test_data_ = LoadData(test_file_, false, -1);
#ifdef OFFLINE
    chrono::steady_clock::time_point end_time = chrono::steady_clock::now();
    printf("测试集读取时间（ms）: %f \n", chrono::duration<double, std::milli>(end_time - start_time).count());
#endif
}

void LR::initParam()
{
    weight_ = vector<double>(features_num, wtInitV);
}

double LR::wxbCalc(const Data &data, const vector<double> weight) {
    double mulSum = 0.0L;
    double wtv, feav;
    for (int i = 0; i < weight.size(); i++) {
        wtv = weight[i];
        feav = data.features[i];
        mulSum += wtv * feav;
    }

    return mulSum;
}
double LR::sigmoidCalc(const double wxb) {
    double expv = exp(-1 * wxb);
    double expvInv = 1 / (1 + expv);
    return expvInv;
}
double LR::lossCal(const vector<double> & weight) {
    double lossV = 0.0L;
    int i;

    for (i = 0; i < train_data_.size(); i++) {
        lossV -= train_data_[i].label * log(sigmoidCalc(wxbCalc(train_data_[i], weight)));
        lossV -= (1 - train_data_[i].label) * log(1 - sigmoidCalc(wxbCalc(train_data_[i], weight)));
    }
    lossV /= train_data_.size();
    return lossV;
}

double LR::gradientSlope(const vector<Data> & dataSet, int index, const vector<double> & sigmoidVec) {
    double gsV = 0.0L;
    int i;
    double sigv, label;
    for (i = 0; i < dataSet.size(); i++) {
        sigv = sigmoidVec[i];
        label = dataSet[i].label;
        gsV += (label - sigv) * (dataSet[i].features[index]);
    }

    gsV = gsV / dataSet.size();
    return gsV;
}

vector<Data> LR::LoadData(const string & filename, bool last_label_exist ,int read_line_num)
{

    FILE * fp = NULL;
    char * line, * record;
    char buffer[20000];


    if ((fp = fopen(filename.c_str(), "rb")) == NULL) {
        printf("file [%s] doesnnot exist \n", filename.c_str());
        exit(1);
    }

    vector<Data> data_set;
    int cnt = 0;
    while ((line = fgets(buffer, sizeof(buffer), fp)) != NULL) {

        if (last_label_exist && cnt >= read_line_num) break;
        cnt++;

        vector<double> feature(features_num + (last_label_exist? 1: 0), 0.0);

        int f_cnt = 0;
        record = strtok(line, ",");
        while (record != NULL) {
//            printf("%s ", record);
            feature[f_cnt++] = atof(record);
            record = strtok(NULL, ","); // 每个特征值
        }

        if (last_label_exist) {
            int ftf = (int) feature.back();
            feature.pop_back();
            data_set.push_back(Data(feature, ftf));
        } else {
            data_set.push_back(Data(feature, 0));
        }
    }
    fclose(fp);
    fp = NULL;


    return data_set;
}


void LR::StorePredict()
{
    string line;
    int i = 0;

    ofstream fout(predict_file_.c_str());
    if (!fout.is_open()) {
        printf("打开预测结果文件失败");
        exit(1);
    }
    for (i = 0; i < predict_vec.size(); i++) {
        fout << predict_vec[i] << endl;
    }
    fout.close();
}

bool loadAnswerData(string awFile, vector<int> & awVec) {
    ifstream infile(awFile.c_str());
    if (!infile) {
        cout << "打开答案文件失败" << endl;
        exit(0);
    }

    while (infile) {
        string line;
        int aw;
        getline(infile, line);
        if (line.size() > 0) {
            stringstream sin(line);
            sin >> aw;
            awVec.push_back(aw);
        }
    }

    infile.close();
    return true;
}



void Test (string answerFile, string predictFile) {
    vector<int> answerVec;
    vector<int> predictVec;
    int correctCount;
    double accurate;

    cout << "ready to load answer data" << endl;
    loadAnswerData(answerFile, answerVec);
    loadAnswerData(predictFile, predictVec);

    cout << "test data set size is " << predictVec.size() << endl;
    correctCount = 0;
    for (int j = 0; j < predictVec.size(); j++) {
        if (j < answerVec.size()) {
            if (answerVec[j] == predictVec[j]) {
                correctCount++;
            }
        } else {
            cout << "answer size less than the real predicted value" << endl;
        }
    }

    accurate = ((double)correctCount) / answerVec.size();
    cout << "the prediction accuracy is " << accurate << endl;
}

int main(int argc, char *argv[])
{

#ifdef OFFLINE
    chrono::steady_clock::time_point start_time = chrono::steady_clock::now();
#endif

#ifdef OFFLINE // 线下测试用的数据路径
    string train_file = "../data/train_data.txt";
    string test_file = "../data/test_data.txt";
    string predict_file = "../data/result.txt";
    string answer_file = "../data/answer.txt";
#else // 提交到线上，官方要求的数据路径
    string train_file = "/data/train_data.txt";
    string test_file = "/data/test_data.txt";
    string predict_file = "/projects/student/result.txt";
    string answer_file = "/projects/student/answer.txt";
#endif

    LR logist(train_file, test_file, predict_file);

#ifdef OFFLINE
    chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
#endif

    logist.train();

    logist.predict();


#ifdef OFFLINE
    chrono::steady_clock::time_point end_time = chrono::steady_clock::now();
    printf("总耗时（ms）: %f \n", chrono::duration<double, std::milli>(end_time - start_time).count());
#endif

#ifdef TEST
    Test(answer_file, predict_file);
#endif

    return 0;
}
