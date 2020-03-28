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
#include <unordered_map>

//#define CacheLineSize 64
//#define IB CacheLineSize / sizeof(float)
#define IB 4

using namespace std;



int features_num = 1000;


void StorePredict(const vector<int> & predict_vec, const string & predict_file)
{

    const int DATA_LEN = 2 * predict_vec.size();
    char * pData = new char[DATA_LEN];
    int j = 0;
    for (int i = 0; i < predict_vec.size(); i++) {
        pData[j++] = predict_vec[i] + '0';
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



void CalcTrainSum (char * buf, int start_line, int line_num, vector<float> & sum_label0, vector<float> & sum_label1, int & cnt_label0, int & cnt_label1) {

    buf += (start_line + line_num) * 6050; // 假设每行6050个字节，得到该块最后一行的中间某个位置
    while (*buf != '\n') buf++; // 跳到该块最后一个字符就，即该块最后一个换行符的位置


    // 开始倒着读取
    int cnt = 0;
    while (cnt < line_num) {

        buf--; // 跳过换行符，来到该行最后一个label的位置
        int label = *buf - '0';
        buf--; // 来到逗号的位置

        int f_cnt = features_num - 1;
        while (f_cnt >= 0) {

            buf -= 1; // 跳过逗号的位置

            float ret = (1000 * (*(buf - 4) - '0') + 100 * (*(buf - 2) - '0'));

            buf -= 5; // 跳过浮点数的部分，来到逗号或者符号位的位置。如果是该行第一个数，则会来到上一行的换行符位置
            if (*buf == '-') {ret = -ret;buf--;} // 负数需要取反，并多移动一位

            if (label == 0) {
                sum_label0[f_cnt] += ret;
            } else {
                sum_label1[f_cnt] += ret;
            }
            f_cnt--;

        }

        cnt++;
        if (label == 0) {
            cnt_label0++;
        } else {
            cnt_label1++;
        }
    }
}

vector<float> label0_means(features_num, 0.0);
vector<float> label1_means(features_num, 0.0);

void Train (const string & filename) {

#ifdef TEST
    clock_t start_time = clock();
#endif

    int fd = open(filename.c_str(), O_RDONLY);

    int buf_len = 10 * 1024 * 1204;
    char * buf = (char *) mmap(NULL, buf_len, PROT_READ, MAP_PRIVATE, fd, 0);

    close(fd);

    vector<float> sum_label0_part0(features_num, 0.0);
    vector<float> sum_label1_part0(features_num, 0.0);
    int cnt_label0_part0 = 0;
    int cnt_label1_part0 = 0;
//    CalcTrainSum(buf, 0, 400, sum_label0_part0, sum_label1_part0, cnt_label0_part0, cnt_label1_part0);
    thread t0(CalcTrainSum, buf, 0, 400, ref(sum_label0_part0), ref(sum_label1_part0), ref(cnt_label0_part0), ref(cnt_label1_part0));

    vector<float> sum_label0_part1(features_num, 0.0);
    vector<float> sum_label1_part1(features_num, 0.0);
    int cnt_label0_part1 = 0;
    int cnt_label1_part1 = 0;
//    CalcTrainSum(buf, 400, 400, sum_label0_part1, sum_label1_part1, cnt_label0_part1, cnt_label1_part1);
    thread t1(CalcTrainSum, buf, 400, 400, ref(sum_label0_part1), ref(sum_label1_part1), ref(cnt_label0_part1), ref(cnt_label1_part1));

    vector<float> sum_label0_part2(features_num, 0.0);
    vector<float> sum_label1_part2(features_num, 0.0);
    int cnt_label0_part2 = 0;
    int cnt_label1_part2 = 0;
//    CalcTrainSum(buf, 800, 400, sum_label0_part2, sum_label1_part2, cnt_label0_part2, cnt_label1_part2);
    thread t2(CalcTrainSum, buf, 800, 400, ref(sum_label0_part2), ref(sum_label1_part2), ref(cnt_label0_part2), ref(cnt_label1_part2));

    vector<float> sum_label0_part3(features_num, 0.0);
    vector<float> sum_label1_part3(features_num, 0.0);
    int cnt_label0_part3 = 0;
    int cnt_label1_part3 = 0;
//    CalcTrainSum(buf, 1200, 400, sum_label0_part3, sum_label1_part3, cnt_label0_part3, cnt_label1_part3);
    thread t3(CalcTrainSum, buf, 1200, 400, ref(sum_label0_part3), ref(sum_label1_part3), ref(cnt_label0_part3), ref(cnt_label1_part3));

    t0.join();
    t1.join();
    t2.join();
    t3.join();

    int cnt_label0 = cnt_label0_part0 + cnt_label0_part1 + cnt_label0_part2 + cnt_label0_part3;
    int cnt_label1 = cnt_label1_part0 + cnt_label1_part1 + cnt_label1_part2 + cnt_label1_part3;
    for (int i = 0; i < features_num; i++) {
        label0_means[i] =  (sum_label0_part0[i] + sum_label0_part1[i] + sum_label0_part2[i] + sum_label0_part3[i]) / static_cast<float>(cnt_label0);
        label1_means[i] = (sum_label1_part0[i] + sum_label1_part1[i] + sum_label1_part2[i] + sum_label1_part3[i]) / static_cast<float>(cnt_label1);
    }

#ifdef TEST
    clock_t end_time = clock();
    printf("测试集读取耗时（s）: %f \n", (double) (end_time - start_time) / CLOCKS_PER_SEC);
#endif

}



void JudgePart (char * buf, int start_line, int line_num, vector<int> & res) {

#ifdef NO_NEON
    int end_line = start_line + line_num;
    buf = buf + start_line * 6000;
    for (int i = start_line; i < end_line; i++) {
        float norm0 = 0.0;
        float norm1 = 0.0;
        for (int j = 0; j < features_num; j++) {
            float ret = 1000 * (*buf - '0') + 100 * (*(buf + 2) - '0');
            float t1 = ret - label0_means[j];
            float t2 = ret - label1_means[j];
            norm0 += t1 * t1;
            norm1 += t2 * t2;
            buf += 6;
        }
        res[i] = norm0 < norm1? 0: 1;
    }
#else

    int N = features_num;
    int i = 0;
    int end_line = start_line + line_num;


    for (i = start_line; i + IB < end_line; i += IB) {
        float32x4_t temp[IB];
        float32x4_t temp1[IB];
        for (int ii = 0; ii < IB; ii++) {
            temp[ii] = vdupq_n_f32(0.0f);
            temp1[ii] = vdupq_n_f32(0.0f);
        }


        for (int k = 0; k < N; k += 4) {

            float32x4_t v_b = vld1q_f32(&label0_means[k]);
            float32x4_t v_b1 = vld1q_f32(&label1_means[k]);

            float32x4_t v_a[IB];
            float32x4_t v_a1[IB];
            for (int ii = 0; ii < IB; ii++) {
                char * start_buf = buf + (i + ii) * 6000 + k * 6;

                float ret[4] = {0.0};
                ret[0] = 1000 * (*start_buf - '0') + 100 * (*(start_buf + 2) - '0');
                start_buf += 6;
                ret[1] = 1000 * (*start_buf - '0') + 100 * (*(start_buf + 2) - '0');
                start_buf += 6;
                ret[2] = 1000 * (*start_buf - '0') + 100 * (*(start_buf + 2) - '0');
                start_buf += 6;
                ret[3] = 1000 * (*start_buf - '0') + 100 * (*(start_buf + 2) - '0');

                v_a[ii] = vld1q_f32(ret);
                v_a[ii] = vsubq_f32(v_a[ii], v_b);

                v_a1[ii] = vld1q_f32(ret);
                v_a1[ii] = vsubq_f32(v_a1[ii], v_b1);

            }

            for (int ii = 0; ii < IB; ii++) {
                temp[ii] = vmlaq_f32(temp[ii], v_a[ii], v_a[ii]);
                temp1[ii] = vmlaq_f32(temp1[ii], v_a1[ii], v_a1[ii]);
            }

        }


        for (int ii = 0; ii < IB; ii++) {
            float32x4_t temp_c = temp[ii];
            float32x4_t temp_c1 = temp1[ii];
            float tmp0 = temp_c[0] + temp_c[1] + temp_c[2] + temp_c[3];
            float tmp1 = temp_c1[0] + temp_c1[1] + temp_c1[2] + temp_c1[3];
            res[i + ii] = (tmp0 < tmp1? 0: 1);

        }


    }



#endif
}


void Predict (const string & test_file, const string & predict_file) {
#ifdef TEST
    clock_t start_time = clock();
#endif

    int fd = open(test_file.c_str(), O_RDONLY);
    char * buf = (char *) mmap(NULL, 120000000, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    vector<int> res(20000, 0);
//    JudgePart(buf, 0, 5000, res);
//    JudgePart(buf, 5000, 5000, res);
//    JudgePart(buf, 10000, 5000, res);
//    JudgePart(buf, 15000, 5000, res);

    int thread_num = 4;
    int cnt_per_thread = 20000 / thread_num;
    vector<thread> thread_vec;
    for (int i = 0; i < thread_num; i++) {
        thread_vec.emplace_back(JudgePart, buf, i * cnt_per_thread, cnt_per_thread, ref(res));
    }

    for (int i = 0; i < thread_vec.size(); i++) {
        thread_vec[i].join();
    }

#ifdef TEST
    clock_t end_time = clock();
    printf("预测耗时（s）: %f \n", (double) (end_time - start_time) / CLOCKS_PER_SEC);
#endif

    StorePredict(res, predict_file);
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


    Train(train_file);

    Predict(test_file, predict_file);


#ifdef TEST
    clock_t end_time = clock();
    printf("总耗时（s）: %f \n", (double) (end_time - start_time) / CLOCKS_PER_SEC);
#endif

#ifdef TEST
    Test(answer_file, predict_file);
#endif
}