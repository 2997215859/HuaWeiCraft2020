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
#define IB 8

using namespace std;



int features_num = 1000;


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



void CalcTrainSum (char * buf, int start_line, int line_num, vector<int> & sum_label0, vector<int> & sum_label1, short & cnt_label0, short & cnt_label1) {

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

            int ret = 100 * (*(buf - 2) - '0');

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

vector<short> means_add(features_num, 0);
vector<short> means_sub(features_num, 0);
short C = 0;

void Train (const string & filename) {

#ifdef TEST
    clock_t start_time = clock();
#endif

    int fd = open(filename.c_str(), O_RDONLY);

    int buf_len = 10 * 1024 * 1204;
    char * buf = (char *) mmap(NULL, buf_len, PROT_READ, MAP_PRIVATE, fd, 0);

    close(fd);

    vector<int> sum_label0_part0(features_num, 0);
    vector<int> sum_label1_part0(features_num, 0);
    short cnt_label0_part0 = 0;
    short cnt_label1_part0 = 0;
//    CalcTrainSum(buf, 0, 400, sum_label0_part0, sum_label1_part0, cnt_label0_part0, cnt_label1_part0);
    thread t0(CalcTrainSum, buf, 0, 400, ref(sum_label0_part0), ref(sum_label1_part0), ref(cnt_label0_part0), ref(cnt_label1_part0));

    vector<int> sum_label0_part1(features_num, 0);
    vector<int> sum_label1_part1(features_num, 0);
    short cnt_label0_part1 = 0;
    short cnt_label1_part1 = 0;
//    CalcTrainSum(buf, 400, 400, sum_label0_part1, sum_label1_part1, cnt_label0_part1, cnt_label1_part1);
    thread t1(CalcTrainSum, buf, 400, 400, ref(sum_label0_part1), ref(sum_label1_part1), ref(cnt_label0_part1), ref(cnt_label1_part1));

    vector<int> sum_label0_part2(features_num, 0);
    vector<int> sum_label1_part2(features_num, 0);
    short cnt_label0_part2 = 0;
    short cnt_label1_part2 = 0;
//    CalcTrainSum(buf, 800, 400, sum_label0_part2, sum_label1_part2, cnt_label0_part2, cnt_label1_part2);
    thread t2(CalcTrainSum, buf, 800, 400, ref(sum_label0_part2), ref(sum_label1_part2), ref(cnt_label0_part2), ref(cnt_label1_part2));

    vector<int> sum_label0_part3(features_num, 0);
    vector<int> sum_label1_part3(features_num, 0);
    short cnt_label0_part3 = 0;
    short cnt_label1_part3 = 0;
//    CalcTrainSum(buf, 1200, 400, sum_label0_part3, sum_label1_part3, cnt_label0_part3, cnt_label1_part3);
    thread t3(CalcTrainSum, buf, 1200, 400, ref(sum_label0_part3), ref(sum_label1_part3), ref(cnt_label0_part3), ref(cnt_label1_part3));

    t0.join();
    t1.join();
    t2.join();
    t3.join();

    short cnt_label0 = cnt_label0_part0 + cnt_label0_part1 + cnt_label0_part2 + cnt_label0_part3;
    short cnt_label1 = cnt_label1_part0 + cnt_label1_part1 + cnt_label1_part2 + cnt_label1_part3;
    for (int i = 0; i < features_num; i++) {
        short a =  (sum_label0_part0[i] + sum_label0_part1[i] + sum_label0_part2[i] + sum_label0_part3[i]) / cnt_label0;
        short b = (sum_label1_part0[i] + sum_label1_part1[i] + sum_label1_part2[i] + sum_label1_part3[i]) / cnt_label1;
        means_add[i] = a + b;
        means_sub[i] = b - a;
        C = (a - b) * (a + b);
    }

    C = C / 200;

#ifdef TEST
    clock_t end_time = clock();
    printf("测试集读取耗时（s）: %f \n", (double) (end_time - start_time) / CLOCKS_PER_SEC);
#endif

}



void JudgePart (char * buf, int start_line, int line_num, vector<char> & res) {

#ifdef NO_NEON
    int end_line = start_line + line_num;
    buf = buf + start_line * 6000;
    for (int i = start_line; i < end_line; i++) {
        short norm_delta = 0;
        for (int j = 0; j < features_num; j++) {
            short ret =  (*(buf + 2) - '0');
            norm_delta += means_sub[j] * ret;
            buf += 6;
        }
        norm_delta = norm_delta + C;
        res[i] = norm_delta < 0? '0': '1';
    }
#else

    int N = features_num;
    int i = 0;
    int end_line = start_line + line_num - 50;


    for (i = start_line; i + IB < end_line; i += IB) {
        int16x8_t temp[IB];
        for (int ii = 0; ii < IB; ii++) {
            temp[ii] = vdupq_n_s16(0);
        }


        for (int k = 0; k < N; k += 8) {

            int16x8_t v_b = vld1q_s16(&means_sub[k]);

            int16x8_t v_a[IB];
            for (int ii = 0; ii < IB; ii++) {
                char * start_buf = buf + (i + ii) * 6000 + k * 6;

                short ret[8];
                ret[0] = (*(start_buf + 2) - '0');
                start_buf += 6;
                ret[1] = (*(start_buf + 2) - '0');
                start_buf += 6;
                ret[2] = (*(start_buf + 2) - '0');
                start_buf += 6;
                ret[3] = (*(start_buf + 2) - '0');
                start_buf += 6;
                ret[4] = (*(start_buf + 2) - '0');
                start_buf += 6;
                ret[5] = (*(start_buf + 2) - '0');
                start_buf += 6;
                ret[6] = (*(start_buf + 2) - '0');
                start_buf += 6;
                ret[7] = (*(start_buf + 2) - '0');

                v_a[ii] = vld1q_s16(ret);

                temp[ii] = vmlaq_s16(temp[ii], v_a[ii], v_b);

            }



        }


        for (int ii = 0; ii < IB; ii++) {
            int16x8_t temp_c = temp[ii];
            short tmp = temp_c[0] + temp_c[1] + temp_c[2] + temp_c[3] + temp_c[4] + temp_c[5] + temp_c[6] + temp_c[7];

//            int16x4_t part_sum4 = vadd_s16(vget_high_s16(temp_c),vget_low_s16(temp_c));// 两两相加
//            float tmp = part_sum4[0] + part_sum4[1] + part_sum4[2] + part_sum4[3];

            res[i + ii] = (tmp + C < 0)? '0': '1';

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
    madvise(buf, 120000000, MADV_SEQUENTIAL);
    close(fd);

    vector<char> res(20000, '1');
//    JudgePart(buf, 0, 5000, res);
//    JudgePart(buf, 5000, 5000, res);
//    JudgePart(buf, 10000, 5000, res);
//    JudgePart(buf, 15000, 5000, res);

    int thread_num = 8;
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