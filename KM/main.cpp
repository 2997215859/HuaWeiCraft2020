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

#define CacheLineSize 64
#define IB CacheLineSize / sizeof(float)
//#define IB 2

using namespace std;

int features_num = 1000;

const int read_line_num = 1500;

static int buf_len_train_file;

static char * buf;

vector<int> train_labels(read_line_num);

vector<vector<float>> train_features(read_line_num, vector<float>(features_num, 0.0));

vector<vector<float>> test_features(20000, vector<float>(1000));


float dot (const vector<float> & vec1, const vector<float> & vec2) {
    float sum = 0.0;
#ifdef NO_NEON

    for (int i = 0; i < vec1.size(); i++) {
        sum += vec1[i] * vec2[i];
    }
#else
    int len = vec1.size();
    const float * p_vec1 = &vec1[0];
    const float * p_vec2 = &vec2[0];


    float32x4_t sum_vec = vdupq_n_f32(0),left_vec,right_vec;
    for(int i = 0; i < len; i += 4)
    {
        left_vec = vld1q_f32(p_vec1 + i);
        right_vec = vld1q_f32(p_vec2 + i);
        sum_vec = vmlaq_f32(sum_vec, left_vec, right_vec);
    }

    float32x2_t r = vadd_f32(vget_high_f32(sum_vec), vget_low_f32(sum_vec));
    sum += vget_lane_f32(vpadd_f32(r,r),0);

#endif
    return sum;
}

void EuclideanDist (const vector<vector<float>> & a, const vector<float> & b, vector<float> & c) { // 不开根号
    int M = a.size();
    int N = a[0].size();

#ifdef NO_NEON

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            c[i] += (a[i][j] - b[j]) * (a[i][j] - b[j]);
        }
    }

#else

    int i = 0;
    for (i = 0; i + IB < M; i += IB) {
        float32x4_t temp[IB];
        for (int ii = 0; ii < IB; ii++) {
            temp[ii] = vdupq_n_f32(0.0f);
        }


        for (int k = 0; k < N; k += 4) {

            float32x4_t v_b = vld1q_f32(&b[k]);

            float32x4_t v_a[IB];
            for (int ii = 0; ii < IB; ii++) {
                v_a[ii] = vld1q_f32(&a[i + ii][k]);
                v_a[ii] = vsubq_f32(v_a[ii], v_b);
            }

            for (int ii = 0; ii < IB; ii++) {
                temp[ii] = vmlaq_f32(temp[ii], v_a[ii], v_a[ii]);
            }

        }


        for (int ii = 0; ii < IB; ii++) {
            float32x4_t temp_c = temp[ii];
            c[i + ii] = (temp_c[0] + temp_c[1]) + (temp_c[2] + temp_c[3]);
        }


    }


#endif

}

inline int getc() {
    if (buf_len_train_file < 0) return EOF;
    buf_len_train_file--;
    return *buf++;
}

inline float GetOneFloatData() {

    int f = 1;
    char ch = getc();

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

    while (cnt < read_line_num) {

        int f_cnt = 0;

        while (f_cnt < features_num) {
            float f = GetOneFloatData();
            train_features[cnt][f_cnt++] = f;
            getc();
        }

        train_labels[cnt] =  getc() - '0';
        getc(); // 获取回车键


        cnt++;


    }

    close(fd);

#ifdef TEST
    clock_t end_time = clock();
    printf("训练集读取时间（s）: %f \n", (double) (end_time - start_time) / CLOCKS_PER_SEC);
#endif

}



void LoadTestDataMultiThread (const string & filename) {

#ifdef TEST
    clock_t start_time = clock();
#endif

    int fd = open(filename.c_str(), O_RDONLY);

    char * buf1 = (char *) mmap(NULL, 120000000, PROT_READ, MAP_PRIVATE, fd, 0);




    int thread_num = 8;
    int cnt_per_thread = 20000 / thread_num;
    vector<thread> thread_vec;
    for (int i = 0; i < thread_num; i++) {
        thread_vec.emplace_back(SplitTestFunc, ref(test_features), buf1 + cnt_per_thread * 6000 * i, i * cnt_per_thread, cnt_per_thread);
    }

    for (int i = 0; i < thread_vec.size(); i++) {
        thread_vec[i].join();
    }


    close(fd);

#ifdef TEST
    clock_t end_time = clock();
    printf("测试集读取耗时（s）: %f \n", (double) (end_time - start_time) / CLOCKS_PER_SEC);
#endif
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

void Predict (const string & predict_file) {
#ifdef TEST
    clock_t start_time = clock();
#endif


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


#ifdef TEST
    clock_t cal_end_time = clock();
    printf("计算平均值阶段耗时（s）: %f \n", (double) (cal_end_time - start_time) / CLOCKS_PER_SEC);
#endif

    vector<char> pred_vec(test_features.size());

    vector<float> norm0(test_features.size());
    vector<float> norm1(test_features.size());
    EuclideanDist(test_features, mean_features0, norm0);
    EuclideanDist(test_features, mean_features1, norm1);

    for (int i = 0; i < test_features.size(); i++) {
        if (norm0[i] < norm1[i]) {
            pred_vec[i] = '0';
        } else {
            pred_vec[i] = '1';
        }
    }


//    for (int i = 0; i < test_features.size(); i++) {
//        float norm0 = dot(test_features[i], test_features[i]);
//        float norm1 = dot(test_features1[i], test_features1[i]);
//        if (norm0 < norm1) {
//            pred_vec[i] = '0';
//        } else {
//            pred_vec[i] = '1';
//        }
//    }

#ifdef TEST
    clock_t end_time = clock();
    printf("预测阶段耗时（s）: %f \n", (double) (end_time - cal_end_time) / CLOCKS_PER_SEC);
#endif
    StorePredict(pred_vec, predict_file);

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

    LoadTestDataMultiThread(test_file);

    Predict(predict_file);



#ifdef TEST
    clock_t end_time = clock();
    printf("总耗时（s）: %f \n", (double) (end_time - start_time) / CLOCKS_PER_SEC);
#endif

#ifdef TEST
    Test(answer_file, predict_file);
#endif
}