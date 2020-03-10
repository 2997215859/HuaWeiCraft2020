#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <climits>
#include <zconf.h>

using namespace std;

///////////////////////////////
///    LR Begin
///////////////////////////////

struct Data {
    vector<double> features;
    int label;
    Data(vector<double> f, int l) : features(f), label(l)
    {}
};
struct Param {
    vector<double> wtSet;
};


class LR {
public:
    void train();
    void predict(const vector<Data> & test_data_set);
    int loadModel();
    int storeModel();
    vector<int>  GetPredictVec();
    LR(const vector<Data> & train_data_set);

private:
    vector<Data> trainDataSet;
    vector<Data> testDataSet;
    vector<int> predictVec;
    Param param;
    string trainFile;
    string testFile;
    string predictOutFile;
    string weightParamFile = "modelweight.txt";

private:
    bool loadTrainData();
    bool loadTestData();
    int storePredict(vector<int> &predict);
    void initParam();
    double wxbCalc(const Data &data);
    double sigmoidCalc(const double wxb);
    double lossCal();
    double gradientSlope(const vector<Data> &dataSet, int index, const vector<double> &sigmoidVec);

private:
    int featuresNum;
    const double wtInitV = 1.0;
    const double stepSize = 0.1;
    const int maxIterTimes = 300;
    const double predictTrueThresh = 0.5;
    const int train_show_step = 10;
};

LR::LR(const vector<Data> & train_data_set): trainDataSet(train_data_set) {
    featuresNum = trainDataSet[0].features.size();
    initParam();
}

inline vector<int> LR::GetPredictVec() {
    return predictVec;
}


void LR::initParam()
{
    param.wtSet.clear();
    for (int i = 0; i < featuresNum; i++) {
        param.wtSet.push_back(wtInitV);
    }
}


double LR::wxbCalc(const Data & data)
{
    double mulSum = 0.0L;
    int i;
    double wtv, feav;
    for (i = 0; i < param.wtSet.size(); i++) {
        wtv = param.wtSet[i];
        feav = data.features[i];
        mulSum += wtv * feav;
    }

    return mulSum;
}

inline double LR::sigmoidCalc(const double wxb)
{
    double expv = exp(-1 * wxb);
    double expvInv = 1 / (1 + expv);
    return expvInv;
}


double LR::lossCal()
{
    double lossV = 0.0L;
    int i;

    for (i = 0; i < trainDataSet.size(); i++) {
        lossV -= trainDataSet[i].label * log(sigmoidCalc(wxbCalc(trainDataSet[i])));
        lossV -= (1 - trainDataSet[i].label) * log(1 - sigmoidCalc(wxbCalc(trainDataSet[i])));
    }
    lossV /= trainDataSet.size();
    return lossV;
}


double LR::gradientSlope(const vector<Data> &dataSet, int index, const vector<double> &sigmoidVec)
{
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

void LR::train() {

    double sigmoidVal;
    double wxbVal;
    int i, j;

    for (i = 0; i < maxIterTimes; i++) {
        vector<double> sigmoidVec;

        for (j = 0; j < trainDataSet.size(); j++) {
            wxbVal = wxbCalc(trainDataSet[j]);
            sigmoidVal = sigmoidCalc(wxbVal);
            sigmoidVec.push_back(sigmoidVal);
        }

        for (j = 0; j < param.wtSet.size(); j++) {
            param.wtSet[j] += stepSize * gradientSlope(trainDataSet, j, sigmoidVec);
        }

        if (i % train_show_step == 0) {
            cout << "iter " << i << ". updated weight value is : ";
            for (j = 0; j < param.wtSet.size(); j++) {
                cout << param.wtSet[j] << "  ";
            }
            cout << endl;
        }
    }
}

void LR::predict(const vector<Data> & test_train_data)
{

    testDataSet = test_train_data;

    double sigVal;
    int predictVal;

    for (int j = 0; j < test_train_data.size(); j++) {
        sigVal = sigmoidCalc(wxbCalc(test_train_data[j]));
        predictVal = sigVal >= predictTrueThresh ? 1 : 0;
        predictVec.push_back(predictVal);
    }

}

int LR::storeModel()
{
    string line;
    int i;

    ofstream fout(weightParamFile.c_str());
    if (!fout.is_open()) {
        cout << "打开模型参数文件失败" << endl;
    }
    if (param.wtSet.size() < featuresNum) {
        cout << "wtSet size is " << param.wtSet.size() << endl;
    }
    for (i = 0; i < featuresNum; i++) {
        fout << param.wtSet[i] << " ";
    }
    fout.close();
    return 0;
}

///////////////////////////////
///    LR end
///////////////////////////////

bool loadAnswerData(string awFile, vector<int> & awVec)
{
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

vector<Data> LoadTrainData(string train_file)
{

    ifstream infile(train_file.c_str());
    string line;

    if (!infile) {
        cout << "打开训练文件失败" << endl;
        exit(0);
    }

    vector<Data> train_data_set;
    while (infile) {
        getline(infile, line);
        if (line.size() > 0) {
            stringstream sin(line);
            char ch;
            double dataV;
            int i = 0;
            vector<double> feature;

            while (sin) {
                char c = sin.peek();
                if (int(c) != -1) {
                    sin >> dataV;
                    feature.push_back(dataV);
                    sin >> ch;
                    i++;
                } else {
                    printf("训练文件数据格式不正确，出错行为[%d]行", train_data_set.size() + 1);
                    exit(1);
                }
            }
            int ftf = (int) feature.back();
            feature.pop_back();
            train_data_set.push_back(Data(feature, ftf));
        }
    }
    infile.close();
    return train_data_set;
}


vector<Data> LoadTestData(string test_file)
{
    ifstream infile(test_file.c_str());
    string lineTitle;

    if (!infile) {
        cout << "打开测试文件失败" << endl;
        exit(0);
    }

    vector<Data> test_data_set;
    while (infile) {
        vector<double> feature;
        string line;
        getline(infile, line);
        if (line.size() > 0) {
            stringstream sin(line);
            double dataV;
            int i = 0;
            char ch;
            while (sin) {
                char c = sin.peek();
                if (int(c) != -1) {
                    sin >> dataV;
                    feature.push_back(dataV);
                    sin >> ch;
                    i++;
                } else {
                    cout << "测试文件数据格式不正确" << endl;
                    exit(-1);
                }
            }
            test_data_set.push_back(Data(feature, 0));
        }
    }

    infile.close();
    return test_data_set;
}

int StorePredict(const vector<int> & predict, string predict_out_file)
{
    string line;
    int i = 0;

    ofstream fout(predict_out_file.c_str());
    if (!fout.is_open()) {
        printf("打开预测结果文件失败");
        exit(1);
    }
    for (i = 0; i < predict.size(); i++) {
        fout << predict[i] << endl;
    }
    fout.close();
    return 0;
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


#ifdef OFFLINE // 线下测试用的数据路径
    string trainFile = "../data/train_data.txt";
    string testFile = "../data/test_data.txt";
    string predictFile = "../data/result.txt";
    string answerFile = "../data/answer.txt";
#else // 提交到线上，官方要求的数据路径
    string trainFile = "/data/train_data.txt";
    string testFile = "/data/test_data.txt";
    string predictFile = "/projects/student/result.txt";
    string answerFile = "/projects/student/answer.txt";
#endif

    vector<Data> train_data_set = LoadTrainData(trainFile);
    vector<Data> test_data_set = LoadTestData(testFile);

    //////////////////////////////
    ///  LR Begin
    //////////////////////////////
    LR logist(train_data_set);

    cout << "ready to train model" << endl;
    logist.train();

    cout << "let's have a prediction test" << endl;
    logist.predict(test_data_set);

    //////////////////////////////
    /// SVR Begin
    //////////////////////////////

    StorePredict(logist.GetPredictVec(), predictFile);

#ifdef TEST
    Test(answerFile, predictFile);
#endif

    return 0;
}
