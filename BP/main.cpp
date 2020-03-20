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

#define IB 4


using namespace std;

struct Cache {
    vector<float> z1;
    vector<float> a1;
    vector<float> z2;
    vector<float> a2;
    Cache (const vector<float> & z1, const vector<float> & a1, const vector<float> & z2, const vector<float> & a2): z1(z1), a1(a1), z2(z2), a2(a2) {

    }
};

struct Grad {
    vector<float> dw1;
    float db1;
    float dw2;
    float db2;
    Grad (const vector<float> & dw1, const float & db1, const float & dw2, const float & db2): dw1(dw1), db1(db1), dw2(dw2), db2(db2) {

    }
};


inline int getc(FILE * fp, bool last_label_exist) {

    static char buf[1<<22], *p1=buf, *p2=buf;
    static char buf1[1<<22], *p3=buf1, *p4=buf1;

    if (last_label_exist) {
        return p1==p2&&(p2=(p1=buf)+fread(buf,1,1<<21, fp),p1==p2)?EOF:*p1++;
    } else {
        return p3==p4&&(p4=(p3=buf1)+fread(buf1,1,1<<21, fp),p3==p4)?EOF:*p3++;
    }

}



const int features_num = 1000;


//const int read_line_num = 1000;

const int read_line_num0 = 300;
const int read_line_num1 = 200;

inline float dot (const vector<float> & vec1, const vector<float> & vec2) {
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

struct Data {
    vector<float> features;
    int label;
    Data(vector<float> f, int l) : features(f), label(l)
    {}
};

vector<float> matrix_mul (const vector<Data> & a, const vector<float> & b) {

    int M = a.size();
    int K = a[0].features.size();
    int N = 1;

    vector<float> c(M, 0);
#ifdef NO_NEON

    for (int i = 0; i < M; i++) {
        c[i] = dot(a[i].features, b);
    }

#else

    int i = 0;
    for (i = 0; i + IB < M; i += IB) {
        float32x4_t temp[IB];
        for (int ii = 0; ii < IB; ii++) {
            temp[ii] = vdupq_n_f32(0.0f);
        }


        for (int k = 0; k < K; k += 4) {
            float32x4_t v_a[IB];
            for (int ii = 0; ii < IB; ii++) {
                v_a[ii] = vld1q_f32(&a[i + ii].features[k]);
            }
            float32x4_t v_b = vld1q_f32(&b[k]);

            for (int ii = 0; ii < IB; ii++) {
                temp[ii] = vmlaq_f32(temp[ii], v_a[ii], v_b);
            }

        }


        for (int ii = 0; ii < IB; ii++) {
            float32x4_t temp_c = temp[ii];
            c[i + ii] = (temp_c[0] + temp_c[1]) + (temp_c[2] + temp_c[3]);
        }


    }

    while (i < M) {
        c[i++] = dot(a[i].features, b);
    }


#endif

    return c;
}



inline float GetOneFloatData(FILE * fp, bool last_label_exist) {

    int f = 1;
    char ch = getc(fp, last_label_exist);

    if (ch == char(EOF)) return -2;

    if (ch == '-') {f = -1; ch = getc(fp, last_label_exist);}


    float ret = ch - '0';
    ret *= 1000;

    getc(fp, last_label_exist); // 点号

    ch = getc(fp, last_label_exist);

    ret += (ch - '0') * 100;
    ch = getc(fp, last_label_exist);

    ret += (ch - '0') * 10;

    ch = getc(fp, last_label_exist);

    ret += (ch - '0');
    return ret * f / 1000.0;
}

vector<float> means;
vector<float> vars;

void NormBySameRef(vector<Data> & data){ // 用之前的平均值和方差进行归一化
    int row_num = data.size();
    int col_num = data[0].features.size();

    for (int j = 0; j < col_num; j++) {
        for (int i = 0; i < row_num; i++) {
            data[i].features[j] = (data[i].features[j] - means[j]) / vars[j];
        }
    }
}

void Norm (vector<Data> & data) {
    int row_num = data.size();
    int col_num = data[0].features.size();
    for (int j = 0; j < col_num; j++) {

        float mean = 0.0;
        for (int i = 0; i < row_num; i++) {
            mean += data[i].features[j];
        }
        mean /= row_num;

        means.push_back(mean);

        float var = 0.0;
        for (int i = 0; i < row_num; i++) {
            var += pow((data[i].features[j] - mean), 2);
        }
        var /= row_num;

        vars.push_back(var);

        for (int i = 0; i < row_num; i++) {
            data[i].features[j] = (data[i].features[j] - mean) / var;
        }
    }
}

vector<Data> LoadData(const string & filename, bool last_label_exist)
{

    FILE * fp = NULL;
    char * line, * record;


    if ((fp = fopen(filename.c_str(), "rb")) == NULL) {
        printf("file [%s] doesnnot exist \n", filename.c_str());
        exit(1);
    }

    vector<Data> data_set;
    int label0_cnt = 0;
    int label1_cnt = 0;
    int cnt = 0;

    while (true) {

        vector<float> feature(features_num + (last_label_exist? 1: 0), 0.0);

        if (last_label_exist && cnt >= 1536) break;

        int f_cnt = 0;
        int eof_flag = 0;
        int neg_flag = 0;
        while (f_cnt < features_num) {
            float f = GetOneFloatData(fp, last_label_exist);
            if (f == -2) {eof_flag = 1; break;}
//            if (f < 0) {neg_flag = 1;}
            feature[f_cnt++] = f;
            getc(fp, last_label_exist);
        }

        if (eof_flag) break;

        if (last_label_exist) {
            feature[f_cnt] =  getc(fp, last_label_exist) - '0';
            getc(fp, last_label_exist); // 获取回车键
        }

        if (neg_flag) continue;


        if (last_label_exist) {
            int ftf = (int) feature.back();
            feature.pop_back();
            data_set.emplace_back(Data(feature, ftf));
            cnt++;

            if (ftf == 0) {
                label0_cnt++;
            } else {
                label1_cnt++;
            }

        } else {
            data_set.emplace_back(Data(feature, 0));
        }

    }

    fclose(fp);
    fp = NULL;

#ifdef TEST
    printf("0 数量: %d, 1 数量: %d \n", label0_cnt, label1_cnt);
#endif


    if (last_label_exist) {
        Norm(data_set);
    } else {
        NormBySameRef(data_set);
    }

//    memset(buf, 0, sizeof(buf) / sizeof(char));
//    p1 = buf;
//    p2 = buf;
    return data_set;

//    while ((line = fgets(buffer, sizeof(buffer), fp)) != NULL) {
//
//        if (last_label_exist && label0_cnt >= read_line_num0 && label1_cnt >= read_line_num1) break;
//
//        vector<float> feature(features_num + (last_label_exist? 1: 0), 0.0);
//
//        int f_cnt = 0;
//        record = strtok(line, ",");
//        int flag = 0;
//        while (record != NULL) {
////            printf("%s ", record);
//            feature[f_cnt++] = atof(record);
//            if (feature[f_cnt - 1] < 0) {
//                flag = 1;
//                break;
//            }
//            record = strtok(NULL, ","); // 每个特征值
//        }
//
//        if (flag) continue;
//
//
//        if (last_label_exist && (label0_cnt < read_line_num0 || label1_cnt < read_line_num1)) {
//            int ftf = (int) feature.back();
//            feature.pop_back();
//            data_set.push_back(Data(feature, ftf));
//
//            if (ftf == 0) {
//                label0_cnt++;
//            } else {
//                label1_cnt++;
//            }
//
//        } else {
//            data_set.emplace_back(Data(feature, 0));
//        }
////        if (last_label_exist) {
////            int ftf = (int) feature.back();
////            feature.pop_back();
////
////            if (ftf == 0 && label0_cnt < read_line_num0 ) {
////                data_set.push_back(Data(feature, ftf));
////                label0_cnt++;
////            } else if (ftf == 1 && label1_cnt < read_line_num1) {
////                data_set.push_back(Data(feature, ftf));
////                label1_cnt++;
////            }
//
////        } else {
////            data_set.push_back(Data(feature, 0));
////        }
//    }

}


class LR {
public:
    void train();

    void predict(const vector<Data> & train_data);

    LR(const string & train_file, const string & test_file, const string & predict_file);
    float fast_exp(float x);

private:
    vector<Data> train_data_;
    vector<Data> test_data_;
    vector<int> predict_vec;

    vector<float> weight1_;
    float weight2_;
    float b1;
    float b2;

    vector<float> min_error_weight_;
    float min_error_rate_;
    int min_error_iter_no_;

    string weightParamFile = "modelweight.txt";

private:
//    vector<Data> LoadData(const string & filename, bool last_label_exist);
    void LoadTrainData();

    void LoadTestData();

    void initParam();
    Cache Propagate (const vector<Data> & data, bool calculate_cost);
    Grad Backward (const Cache & cache, const vector<Data> & data, int lambd);
    void Optimize (int batch_size, float learning_rate, int lambd);

    float sigmoidCalc(const float wxb);
    float lossCal(const vector<float> & weight);

    float gradientSlope(const vector<Data> & dataSet, int index, const vector<float> & sigmoidVec);
    void StorePredict();

private:
    string train_file_;
    string test_file_;
    string predict_file_;

private:


    float rate_start = 1.2;
    const float decay = 1300;
    const float rate_min = 0.001;

    const int maxIterTimes = 6;
    const float predictTrueThresh = 0.5;
    const int train_show_step = 1;
};

inline float LR::fast_exp(float x) {
    union {uint32_t i;float f;} v;
    v.i=(1<<23)*(1.4426950409*x+126.93490512f);

    return v.f;
}

Grad LR::Backward (const Cache & cache, const vector<Data> & data, int lambd) {
    int len = cache.a2.size();

    vector<float> d_z2(len, 0.0);
    float d_w2 = 0;
    float d_b2 = 0;
    vector<float> d_z1(len, 0.0);

    for (int i = 0; i < len; i++) {
        d_z2[i] = cache.a2[i] - data[i].label;
        d_w2 += d_z2[i] * cache.a1[i];
        d_b2 += d_z2[i];
        d_z1[i] = weight2_ * d_z2[i] * (1 - cache.a1[i] * cache.a1[i]);
    }

    d_w2 += lambd * weight2_;
    d_w2 /= len;

    d_b2 /= len;

    vector<float> d_w1(features_num, 0.0);
    for (int i = 0; i < d_w1.size(); i++) {
        for (int j = 0; j < d_z1.size(); j++) {
            d_w1[i] += d_z1[j] * data[j].features[i];
        }
        d_w1[i] += lambd * weight1_[i];
        d_w1[i] /= len;
    }

    float d_b1 = 0.0;
    for (int i = 0; i < len; i++) {
        d_b1 += d_z1[i];
    }
    d_b1 /= len;

    Grad grad(d_w1, d_b1, d_w2, d_b2);

    return grad;

}

Cache LR::Propagate (const vector<Data> & data, bool calculate_cost) {
    int len = data.size();


    vector<float> z1 =  matrix_mul(data, weight1_);
    vector<float> a1(len, 0.0);
    vector<float> z2(len, 0.0);
    vector<float> a2(len, 0.0);


    for (int i = 0; i < len; i++) {
        z1[i] += b1;
        a1[i] = tanh(z1[i]);
        z2[i] = weight2_ * a1[i] + b2;
        a2[i] = 1 / (1 + exp(-z2[i]));
    }

    if (calculate_cost) {
        float cost = 0.0;
        for (int i = 0; i < data.size(); i++) {
            cost += data[i].label * log(a2[i]) + (1 - data[i].label) * log(1 - a2[i]);
        }
        cost = -1.0 * cost / len;
        Cache cache(z1, a1, z2, a2);

        return cache;
    } else {
        Cache cache({}, {}, {}, a2);
        return cache;

    }

}

void LR::Optimize (int batch_size, float learning_rate, int lambd) {
    vector<float> costs(maxIterTimes, 0.0);
    int batch_num = train_data_.size() / batch_size;
    int batch_num1 = batch_num - 1;

    for (int i = 0; i < maxIterTimes; i++) {
        for (int j = 0; j < batch_num; j++) {
            vector<Data> batch_samples;
            if (j == batch_num - 1) {
                batch_samples = vector<Data>(train_data_.begin() + j * batch_size, train_data_.end());
            } else {
                batch_samples = vector<Data>(train_data_.begin() + j * batch_size, train_data_.begin() + (j + 1) * batch_size);
            }

            Cache cache = Propagate(batch_samples, true);



            Grad grad = Backward(cache, batch_samples, lambd);
            

            for (int i = 0; i < grad.dw1.size(); i++) {
                weight1_[i] = weight1_[i] - learning_rate * grad.dw1[i];
            }
            b1 = b1 - learning_rate * grad.db1;
            weight2_ = weight2_ - learning_rate * grad.dw2;
            b2 = b2 - learning_rate * grad.db2;

        }
    }
}

void LR::train() {

    LoadTrainData();

#ifdef TEST
    clock_t start_time = clock();
#endif

    Optimize(512, 0.7, 90);


#ifdef TEST
    clock_t end_time = clock();
    printf("模型训练时间（s）: %f \n", (double) (end_time - start_time) / CLOCKS_PER_SEC);
#endif

}

void LR::predict(const vector<Data> & test_data) {


//    LoadTestData();

#ifdef TEST
    clock_t start_time = clock();
#endif

    Cache cache = Propagate(test_data, false);
    for (int i = 0; i < cache.a2.size(); i++) {
        int predictVal = cache.a2[i] >= predictTrueThresh ? 1 : 0;
        predict_vec.push_back(predictVal);
    }

#ifdef TEST



    clock_t end_time = clock();
    printf("模型预测时间（s）: %f \n", (double) (end_time - start_time) / CLOCKS_PER_SEC);
#endif

    StorePredict();


#ifdef TEST
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


inline void LR::LoadTrainData() {
#ifdef TEST
    clock_t start_time = clock();
#endif
    train_data_ = LoadData(train_file_, true);
#ifdef TEST
    clock_t end_time = clock();
    printf("训练集读取时间（s）: %f \n", (double) (end_time - start_time) / CLOCKS_PER_SEC);
#endif
}

inline void LR::LoadTestData() {
#ifdef TEST
    clock_t start_time = clock();
#endif
    test_data_ = LoadData(test_file_, false);
#ifdef TEST
    clock_t end_time = clock();
    printf("测试集读取时间（s）: %f \n", (double) (end_time - start_time) / CLOCKS_PER_SEC);
#endif
}

inline void LR::initParam()
{
    weight1_ = vector<float>({0.004170,0.007203,0.000001,0.003023,0.001468,0.000923,0.001863,0.003456,0.003968,0.005388,0.004192,0.006852,0.002045,0.008781,0.000274,0.006705,0.004173,0.005587,0.001404,0.001981,0.008007,0.009683,0.003134,0.006923,0.008764,0.008946,0.000850,0.000391,0.001698,0.008781,0.000983,0.004211,0.009579,0.005332,0.006919,0.003155,0.006865,0.008346,0.000183,0.007501,0.009889,0.007482,0.002804,0.007893,0.001032,0.004479,0.009086,0.002936,0.002878,0.001300,0.000194,0.006788,0.002116,0.002655,0.004916,0.000534,0.005741,0.001467,0.005893,0.006998,0.001023,0.004141,0.006944,0.004142,0.000500,0.005359,0.006638,0.005149,0.009446,0.005866,0.009034,0.001375,0.001393,0.008074,0.003977,0.001654,0.009275,0.003478,0.007508,0.007260,0.008833,0.006237,0.007509,0.003489,0.002699,0.008959,0.004281,0.009648,0.006634,0.006217,0.001147,0.009495,0.004499,0.005784,0.004081,0.002370,0.009034,0.005737,0.000029,0.006171,0.003266,0.005271,0.008859,0.003573,0.009085,0.006234,0.000158,0.009294,0.006909,0.009973,0.001723,0.001371,0.009326,0.006968,0.000660,0.007555,0.007539,0.009230,0.007115,0.001243,0.000199,0.000262,0.000283,0.002462,0.008600,0.005388,0.005528,0.008420,0.001242,0.002792,0.005858,0.009696,0.005610,0.000186,0.008006,0.002330,0.008071,0.003879,0.008635,0.007471,0.005562,0.001365,0.000599,0.001213,0.000446,0.001075,0.002257,0.007130,0.005597,0.000126,0.000720,0.009673,0.005681,0.002033,0.002523,0.007438,0.001954,0.005814,0.009700,0.008468,0.002398,0.004938,0.006200,0.008290,0.001568,0.000186,0.000700,0.004863,0.006063,0.005689,0.003174,0.009886,0.005797,0.003801,0.005509,0.007453,0.006692,0.002649,0.000663,0.003701,0.006297,0.002102,0.007528,0.000665,0.002603,0.008048,0.001934,0.006395,0.005247,0.009248,0.002633,0.000660,0.007351,0.007722,0.009078,0.009320,0.000140,0.002344,0.006168,0.009490,0.009502,0.005567,0.009156,0.006416,0.003900,0.004860,0.006043,0.005495,0.009262,0.009187,0.003949,0.009633,0.001740,0.001263,0.001351,0.005057,0.000215,0.009480,0.008271,0.000150,0.001762,0.003321,0.001310,0.008095,0.003447,0.009401,0.005820,0.008788,0.008447,0.009054,0.004599,0.005463,0.007986,0.002857,0.004903,0.005991,0.000155,0.005935,0.004337,0.008074,0.003152,0.008929,0.005779,0.001840,0.007879,0.006120,0.000539,0.004202,0.006791,0.009186,0.000004,0.009768,0.003766,0.009738,0.006047,0.008288,0.005747,0.006281,0.002856,0.005868,0.007500,0.008583,0.007551,0.006981,0.008645,0.003227,0.006708,0.004509,0.003821,0.004108,0.004015,0.003174,0.006219,0.004302,0.009738,0.006778,0.001986,0.004267,0.003433,0.007976,0.008800,0.009038,0.006627,0.002702,0.002524,0.008549,0.005277,0.008022,0.005725,0.007331,0.005190,0.007709,0.005689,0.004657,0.003427,0.000682,0.003779,0.000796,0.009828,0.001816,0.008119,0.008750,0.006884,0.005695,0.001610,0.004669,0.003452,0.002250,0.005925,0.003123,0.009163,0.009096,0.002571,0.001109,0.001930,0.004996,0.007286,0.002082,0.002480,0.008517,0.004158,0.006167,0.002337,0.001020,0.005159,0.004771,0.001527,0.006218,0.005440,0.006541,0.001445,0.007515,0.002220,0.005194,0.007853,0.000223,0.003244,0.008729,0.008447,0.005384,0.008666,0.009498,0.008264,0.008541,0.000987,0.006513,0.007035,0.006102,0.007996,0.000346,0.007702,0.007317,0.002597,0.002571,0.006323,0.003453,0.007966,0.004461,0.007827,0.009905,0.003002,0.001430,0.009013,0.005416,0.009747,0.006366,0.009939,0.005461,0.005264,0.001354,0.003557,0.000262,0.001604,0.007456,0.000304,0.003665,0.008623,0.006927,0.006909,0.001886,0.004419,0.005816,0.009898,0.002039,0.002477,0.002622,0.007502,0.004570,0.000569,0.005085,0.002120,0.007986,0.002973,0.000276,0.005934,0.008438,0.003810,0.007499,0.005111,0.005410,0.009594,0.008040,0.000323,0.007094,0.004650,0.009475,0.002214,0.002671,0.000815,0.004286,0.001090,0.006338,0.008030,0.006968,0.007662,0.003425,0.008459,0.004288,0.008240,0.006265,0.001434,0.000784,0.000183,0.000667,0.004586,0.001133,0.000278,0.007549,0.003949,0.007469,0.004524,0.004501,0.004781,0.004740,0.008032,0.004024,0.009047,0.000371,0.007739,0.001256,0.006185,0.000104,0.005386,0.000030,0.009512,0.009054,0.007960,0.009153,0.001456,0.001577,0.001876,0.006225,0.009058,0.009900,0.007111,0.007318,0.009093,0.004009,0.002499,0.001734,0.001195,0.008126,0.001468,0.002643,0.008191,0.003106,0.009824,0.002666,0.005337,0.003145,0.009108,0.003666,0.004336,0.005123,0.009389,0.000309,0.007169,0.008910,0.000273,0.005221,0.003260,0.008595,0.005585,0.006902,0.004529,0.006283,0.002901,0.000093,0.005768,0.003114,0.005173,0.009164,0.004265,0.002474,0.003713,0.009319,0.009369,0.008443,0.009202,0.002279,0.000875,0.002273,0.003144,0.001748,0.006071,0.004136,0.008164,0.001851,0.007019,0.002404,0.005742,0.003490,0.000570,0.002288,0.006641,0.004973,0.005190,0.001747,0.005707,0.009968,0.008168,0.005944,0.009760,0.009016,0.005956,0.000324,0.000936,0.000654,0.004517,0.003754,0.009754,0.001680,0.009728,0.007675,0.008242,0.006326,0.006687,0.004769,0.000131,0.003530,0.004921,0.007301,0.004686,0.004574,0.001377,0.000109,0.007583,0.003200,0.009844,0.002202,0.003387,0.005239,0.007549,0.004639,0.001248,0.003125,0.005045,0.006738,0.007701,0.001303,0.000229,0.005191,0.008100,0.000126,0.006725,0.006868,0.004492,0.009148,0.006444,0.000052,0.004844,0.008593,0.008304,0.006492,0.006737,0.005785,0.002741,0.005605,0.006717,0.003524,0.008558,0.001950,0.007473,0.002896,0.007738,0.004277,0.008077,0.003535,0.002137,0.007673,0.003086,0.007332,0.007445,0.002214,0.002141,0.001989,0.001425,0.003771,0.000266,0.001109,0.006746,0.007998,0.000805,0.002317,0.002076,0.009173,0.007113,0.005539,0.003045,0.008349,0.004353,0.009235,0.007061,0.004780,0.001262,0.009760,0.001598,0.002026,0.004312,0.004042,0.001468,0.007293,0.001887,0.006439,0.007543,0.002107,0.006010,0.007489,0.006382,0.005971,0.002955,0.007316,0.009453,0.004256,0.007822,0.000561,0.008353,0.001923,0.003951,0.003001,0.000801,0.009046,0.003702,0.005307,0.004941,0.001322,0.002065,0.000762,0.005079,0.002615,0.003571,0.001081,0.007876,0.001066,0.009857,0.001772,0.005724,0.000448,0.007871,0.001896,0.005279,0.007401,0.001499,0.005511,0.002166,0.007592,0.007229,0.001765,0.008620,0.000198,0.008602,0.005589,0.004032,0.007587,0.007169,0.009873,0.002781,0.000038,0.009339,0.008579,0.007289,0.005167,0.007070,0.007805,0.003749,0.007703,0.007506,0.006132,0.004019,0.006973,0.000031,0.007749,0.008964,0.002393,0.001208,0.002203,0.003021,0.008830,0.005432,0.002867,0.001384,0.002901,0.006139,0.003241,0.004574,0.004441,0.008281,0.004263,0.003457,0.006750,0.002215,0.004672,0.003148,0.006269,0.008774,0.004477,0.007845,0.004570,0.006562,0.001318,0.004330,0.009093,0.006055,0.007668,0.005047,0.004981,0.008429,0.000678,0.005733,0.009428,0.005179,0.001945,0.008479,0.002516,0.007007,0.005403,0.009488,0.006243,0.008380,0.000079,0.009893,0.000777,0.003221,0.009462,0.000089,0.008227,0.008612,0.004398,0.002557,0.008027,0.004779,0.001343,0.009278,0.008960,0.004915,0.008567,0.004186,0.006835,0.003980,0.005057,0.001896,0.009650,0.002942,0.001035,0.001443,0.000141,0.007159,0.005645,0.007946,0.005071,0.007918,0.006958,0.007778,0.004065,0.006478,0.001798,0.003218,0.001726,0.004086,0.002414,0.004069,0.009752,0.003203,0.009825,0.006363,0.003751,0.008575,0.006196,0.002520,0.007929,0.004329,0.003575,0.003303,0.006974,0.002687,0.008083,0.002953,0.005441,0.004879,0.008554,0.008884,0.001844,0.005853,0.008982,0.004461,0.009219,0.002790,0.006088,0.006825,0.002282,0.000138,0.004167,0.009385,0.003430,0.007797,0.001747,0.003420,0.001446,0.007168,0.006993,0.006885,0.002534,0.006924,0.002273,0.004246,0.003719,0.003553,0.000577,0.006316,0.007073,0.006136,0.006483,0.001699,0.001494,0.005142,0.008753,0.001840,0.004628,0.004289,0.004973,0.001615,0.003424,0.002619,0.008445,0.008003,0.004266,0.006070,0.001455,0.005096,0.002969,0.008597,0.006716,0.006335,0.001248,0.004706,0.009866,0.009483,0.006451,0.001517,0.006391,0.005657,0.004687,0.004280,0.005993,0.008500,0.007511,0.005794,0.009247,0.000647,0.009913,0.000530,0.001995,0.004228,0.001075,0.006237,0.000480,0.002846,0.000610,0.007035,0.006685,0.003786,0.001882,0.007470,0.003404,0.007953,0.004879,0.005257,0.000285,0.006442,0.003507,0.002292,0.004339,0.003825,0.004698,0.009795,0.003644,0.007744,0.005528,0.008891,0.003550,0.002455,0.009110,0.000435,0.009508,0.005564,0.003764,0.009951,0.000584,0.005167,0.000311,0.005712,0.001805,0.006310,0.009809,0.008749,0.004518,0.007085,0.007775,0.004948,0.005285,0.001508,0.003694,0.001422,0.007269,0.004770,0.004489,0.008860,0.005276,0.004091,0.002689,0.000720,0.004181,0.000258,0.002912,0.005035,0.009659,0.001094,0.006730,0.004999,0.007771,0.001436,0.000832,0.003992,0.007970,0.001917,0.007678,0.002903,0.002169,0.000167,0.003987,0.003811,0.006593,0.000709,0.001526,0.000166,0.001138,0.006518,0.004027,0.003210,0.005579,0.009935,0.008345,0.006996,0.009183,0.000397,0.000703,0.004740,0.003492,0.009373,0.004896,0.005396,0.008953,0.004466,0.008770,0.002536,0.002738,0.003284,0.005476,0.002201,0.006714,0.001428,0.000941,0.008702,0.002369,0.003860,0.005715,0.005258,0.000760,0.008741,0.009511,0.008125,0.002838,0.005278,0.003394,0.005547,0.009744,0.003117,0.006688,0.003260,0.007745});
    b1 = 0.0032581;
    weight2_ = 0.00889827;
    b2 = 0.00751708;

//    srand(1);
//    weight1_ = vector<float>(features_num, 0.0);
//    for (int i = 0; i < features_num; i++) {
//        weight1_[i] = 0.01 * rand() / RAND_MAX;
//    }
//    b1 = 0.01 * rand() / RAND_MAX;
//    weight2_ = 0.01 * rand() / RAND_MAX;
//    b2 = 0.01 * rand() / RAND_MAX;
}





inline float LR::sigmoidCalc(const float wxb) {
    return 1 / (1 + exp(-1 * wxb));
}
inline float LR::lossCal(const vector<float> & weight) {
    float lossV = 0.0L;
    int i;

    for (i = 0; i < train_data_.size(); i++) {
        lossV -= train_data_[i].label * log(sigmoidCalc(dot(train_data_[i].features, weight)));
        lossV -= (1 - train_data_[i].label) * log(1 - sigmoidCalc(dot(train_data_[i].features, weight)));
    }
    lossV /= train_data_.size();
    return lossV;
}

inline float LR::gradientSlope(const vector<Data> & dataSet, int index, const vector<float> & sigmoidVec) {
    float gsV = 0.0L;
    float sigv, label;
    for (int i = 0; i < dataSet.size(); i++) {
        sigv = sigmoidVec[i];
        label = dataSet[i].label;
        gsV += (label - sigv) * (dataSet[i].features[index]);
    }

    gsV = gsV / dataSet.size();
    return gsV;
}


inline void LR::StorePredict()
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

int main(int argc, char *argv[])
{

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


    packaged_task<vector<Data>(const string & , bool)> test_task(LoadData);
    future<vector<Data>> test_future = test_task.get_future();
    thread t1(move(test_task), test_file, false);


    LR logist(train_file, test_file, predict_file);


    logist.train();

    vector<Data> test_data = test_future.get();

    logist.predict(test_data);


#ifdef TEST
    clock_t end_time = clock();
    printf("总耗时（s）: %f \n", (double) (end_time - start_time) / CLOCKS_PER_SEC);
#endif

#ifdef TEST
    Test(answer_file, predict_file);
#endif

    t1.join();

    return 0;
}
