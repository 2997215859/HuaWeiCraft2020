#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <climits>


#include <cfloat>
#include <cstring>
#include <future>
#include <thread>

#ifndef NO_NEON
#include <arm_neon.h>
#endif


#include <sys/mman.h>

#ifdef ZCONF
#include <zconf.h>
#endif

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

using namespace std;

int features_num = 1000;

const int read_line_num = 1500;

static int buf_len_train_file;

static char * buf;

vector<int> train_labels;

vector<vector<float>> train_features;


inline int getc() {
    if (buf_len_train_file < 0) return EOF;
    buf_len_train_file--;
    return *buf++;
}

inline float GetOneFloatData() {

    int f = 1;
    char ch = getc();

    if (ch == char(EOF)) return -2;

    if (ch == '-') {f = -1; ch = getc();}


    float ret = ch - '0';
    ret *= 1000;

    getc(); // 点号

    ch = getc();

    ret += (ch - '0') * 100;
    ch = getc();

    ret += (ch - '0') * 10;

    ch = getc();

    ret += (ch - '0');
    return ret * f / 1000.0;
}





inline float GetOneFloatData (char* &  buf) {
    int f = 1;
    char ch = *buf++;

//    if (ch == char(EOF)) return -2;
    if (ch == '\0') return -2;

    if (ch == '-') {f = -1; ch = *buf++;}


    float ret = ch - '0';
    ret *= 1000;

    buf++; // 点号

    ch = *buf++;

    ret += (ch - '0') * 100;
    ch = *buf++;

    ret += (ch - '0') * 10;

    ch = *buf++;

    ret += (ch - '0');

    return ret * f / 1000.0;
}

void SplitTestFunc (vector<vector<float>> & data_set, char * buf, int line_start, int line_num)
{

    int cnt = 0;

    while (cnt < line_num) {

        int f_cnt = 0;
        int eof_flag = 0;

        while (f_cnt < features_num) {
            float f = GetOneFloatData(buf);
            if (f == -2) {eof_flag = 1; break;}
            data_set[line_start + cnt][f_cnt++] = f;
            buf++;
        }
        if (eof_flag) break;
        cnt++;
//        if (*buf == '\0') break;
    }
}

void LoadTrainData (const string & filename) {
#ifdef TEST
    clock_t start_time = clock();
#endif

    int fd = open(filename.c_str(), O_RDONLY);


    buf_len_train_file = 10 * 1024 * 1204;
    buf = (char *) mmap(NULL, buf_len_train_file, PROT_READ, MAP_PRIVATE, fd, 0);


    int cnt = 0;


    vector<float> feature(features_num, 0.0);

    while (cnt < read_line_num) {

        int f_cnt = 0;
        int eof_flag = 0;
        int neg_flag = 0;
        while (f_cnt < features_num) {
            float f = GetOneFloatData();

            if (f == -2) {eof_flag = 1; break;}
            feature[f_cnt++] = f;
            getc();
        }

        if (eof_flag) break;

        int ftf =  getc() - '0';
        getc(); // 获取回车键

        if (neg_flag) continue;

        train_features.emplace_back(feature);
        train_labels.emplace_back(ftf);

        cnt++;


    }

    close(fd);

#ifdef TEST
    clock_t end_time = clock();
    printf("训练集读取时间（s）: %f \n", (double) (end_time - start_time) / CLOCKS_PER_SEC);
#endif

}



vector<vector<float>> LoadTestDataMultiThread (const string & filename) {

    int fd = open(filename.c_str(), O_RDONLY);

    char * buf1 = (char *) mmap(NULL, 120000000, PROT_READ, MAP_PRIVATE, fd, 0);


    vector<vector<float>> data_set(20000, vector<float>(1000));

    int thread_num = 8;
    int cnt_per_thread = 20000 / thread_num;
    vector<thread> thread_vec;
    for (int i = 0; i < thread_num; i++) {
        thread_vec.emplace_back(SplitTestFunc, ref(data_set), buf1 + cnt_per_thread * 6000 * i, i * cnt_per_thread, cnt_per_thread);
    }

    for (int i = 0; i < thread_vec.size(); i++) {
        thread_vec[i].join();
    }


    close(fd);


    return data_set;

}


void StorePredict(const vector<char> & predict_vec, const string & predict_file)
{

    const int DATA_LEN = 2 * predict_vec.size();
    char * pData = new char[DATA_LEN];
    int j = 0;
    for (int i = 0; i < predict_vec.size(); i++) {
        pData[j++] = predict_vec[i];
        pData[j++] = '\n';
    }
    int fd = open(predict_file.c_str(), O_RDWR | O_CREAT, 0777);
    lseek(fd, DATA_LEN - 1, SEEK_SET);
    write(fd, "\0", 1);
    void* p = mmap(NULL, DATA_LEN, PROT_WRITE, MAP_SHARED, fd, 0);

    close(fd);
    memcpy(p, pData, DATA_LEN);

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
    float accurate;

    cout << "ready to load answer data" << endl;
    loadAnswerData(answerFile, answerVec);
    loadAnswerData(predictFile, predictVec);

    cout << "test data set size is " << predictVec.size() << endl;
    correctCount = 0;
    int alb0_cnt = 0;
    int alb1_cnt = 0;
    for (int j = 0; j < answerVec.size(); j++) {
        if (j < answerVec.size()) {
            if (answerVec[j] == 0) {
                alb0_cnt++;
            } else {
                alb1_cnt++;
            }
            if (answerVec[j] == predictVec[j]) {
                correctCount++;
            }
        } else {
            cout << "answer size less than the real predicted value" << endl;
        }
    }

    accurate = ((float)correctCount) / answerVec.size();
    cout << "the prediction accuracy is " << accurate << endl;
    cout << "answer 0 cnt: " << alb0_cnt << " ; 1 cnt: " << alb1_cnt << endl;
}


int main (int argc, char *argv[]) {
#ifdef TEST
    clock_t start_time = clock() ;
#endif


#ifdef TEST // 线下测试用的数据路径
    string train_file = "../data/train_data.txt";
    string test_file = "../data/test_data2.txt";
    string predict_file = "../data/result.txt";
    string answer_file = "../data/answer.txt";
#else // 提交到线上，官方要求的数据路径
    string train_file = "/data/train_data.txt";
    string test_file = "/data/test_data.txt";
    string predict_file = "/projects/student/result.txt";
    string answer_file = "/projects/student/answer.txt";
#endif


    LoadTrainData(train_file);

    vector<int> train_labels_cp = train_labels;

    vector<vector<float>> train_features_cp = train_features;

    vector<float> mean_features0(features_num, 0.0);
    int cnt0 = 0;
    vector<float> mean_features1(features_num, 0.0);
    int cnt1 = 0;

    for (int i = 0; i < train_features.size(); i++) {
        if (train_labels[i]) {
            for (int j = 0; j < features_num; j++) {
                mean_features1[j] += train_features[i][j];
            }
            cnt1++;
        } else {
            for (int j = 0; j < features_num; j++) {
                mean_features0[j] += train_features[i][j];
            }
            cnt0++;
        }
    }

    for (int j = 0; j < features_num; j++) {
        mean_features0[j] /= cnt0;
        mean_features1[j] /= cnt1;
    }

    vector<vector<float>> test_data = LoadTestDataMultiThread(test_file);

    vector<char> pred_vec(test_data.size());
    for (int i = 0; i < test_data.size(); i++) {
        float norm0 = 0.0;
        float norm1 = 0.0;
        for (int j = 0; j < features_num; j++) {
            norm0 += pow(mean_features0[j] - test_data[i][j], 2);
            norm1 += pow(mean_features1[j] - test_data[i][j], 2);
        }

        if (norm0 < norm1) {
            pred_vec[i] = '0';
        } else {
            pred_vec[i] = '1';
        }
    }

    StorePredict(pred_vec, predict_file);

#ifdef TEST
    clock_t end_time = clock();
    printf("总耗时（s）: %f \n", (double) (end_time - start_time) / CLOCKS_PER_SEC);
#endif

#ifdef TEST
    Test(answer_file, predict_file);
#endif
}