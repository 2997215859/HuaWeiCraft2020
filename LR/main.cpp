#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <climits>


#include <cfloat>
#include <cstring>

#ifndef NO_NEON
#include <arm_neon.h>
#endif

using namespace std;

const int features_num = 1000;


//const int read_line_num = 1000;

const int read_line_num0 = 226;
const int read_line_num1 = 450;


struct Data {
    vector<float> features;
    int label;
    Data(vector<float> f, int l) : features(f), label(l)
    {}
};


class LR {
public:
    void train();

    void predict();

    LR(const string & train_file, const string & test_file, const string & predict_file);
    float fast_exp(float x);

private:
    vector<Data> train_data_;
    vector<Data> test_data_;
    vector<int> predict_vec;

    vector<float> weight_;
    vector<float> min_error_weight_;
    float min_error_rate_;
    int min_error_iter_no_;

    string weightParamFile = "modelweight.txt";

private:
    vector<Data> LoadData(const string & filename, bool last_label_exist);
    void LoadTrainData();

    void LoadTestData();

    void initParam();

    float dot (const vector<float> & vec1, const vector<float> & vec2);

    float sigmoidCalc(const float wxb);
    float lossCal(const vector<float> & weight);

    float gradientSlope(const vector<Data> & dataSet, int index, const vector<float> & sigmoidVec);
    void StorePredict();

private:
    string train_file_;
    string test_file_;
    string predict_file_;

private:

    const float wtInitV = 0.1;

    float rate_start = 1.2;
    const float decay = 0.01;
    const float rate_min = 0.001;

    const int maxIterTimes = 50;
    const float predictTrueThresh = 0.5;
    const int train_show_step = 1;
};

inline float LR::fast_exp(float x) {
    union {uint32_t i;float f;} v;
    v.i=(1<<23)*(1.4426950409*x+126.93490512f);

    return v.f;
}


void LR::train() {

    LoadTrainData();

#ifdef TEST
    clock_t start_time = clock();
#endif

    float sigmoidVal;
    float wxbVal;

    vector<float> sigmoidVec(train_data_.size(), 0.0);

    for (int i = 0; i < maxIterTimes; i++) {

        for (int j = 0; j < train_data_.size(); j++) {
            wxbVal = dot(train_data_[j].features, weight_);
            sigmoidVal = sigmoidCalc(wxbVal);
            sigmoidVec[j] = sigmoidVal;
        }

        float error_rate = 0.0;
        for (int j = 0; j < train_data_.size(); j++) {
            error_rate += pow(train_data_[j].label - sigmoidVec[j], 2);
        }

        if (error_rate < min_error_rate_) {
            min_error_rate_ = error_rate;
            min_error_weight_ = weight_;
            min_error_iter_no_ = i;
        }


//        float rate = rate_start * 1.0 / (1.0 + decay * i);
//        rate = max(rate, rate_min);

        rate_start = rate_start * 1.0 / (1.0 + decay * i);
        float rate = max(rate_start, rate_min);

//        float rate = 0.50;

        for (int j = 0; j < weight_.size(); j++) {
            weight_[j] += rate * gradientSlope(train_data_, j, sigmoidVec);
        }

#ifdef TEST
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


#ifdef TEST
    clock_t end_time = clock();
    printf("模型训练时间（s）: %f \n", (double) (end_time - start_time) / CLOCKS_PER_SEC);
#endif

}

void LR::predict() {

    LoadTestData();

#ifdef TEST
    clock_t start_time = clock();
#endif

    float sigVal;
    int predictVal;

    for (int j = 0; j < test_data_.size(); j++) {
        sigVal = sigmoidCalc(dot(test_data_[j].features, min_error_weight_));
//        sigVal = sigmoidCalc(dot(test_data_[j].features, weight_));
        predictVal = sigVal >= predictTrueThresh ? 1 : 0;
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
    weight_ = vector<float>(features_num, wtInitV);
//    weight_ = vector<float>({-0.05619839195321735,-0.10774802287219157,-0.07349018287307663,-0.1777217643879414,-0.2501887537655504,-0.11121376301677374,0.2449273127011548,-0.03229639911345727,0.36913451559366633,0.1062877530017812,0.08359840517334238,-0.1508124393507631,0.005718638535695007,0.19457730997149847,-0.06538829121070838,0.2578696425062593,-0.11107330479958935,0.07974466524480658,0.23122238218235397,-0.2655621437427918,-0.000599533135228994,0.050905236317075386,-0.02021859476726999,-0.09294570968549984,0.47939842415266043,0.22614222291493397,-0.045363661068183514,0.3489834587467489,0.004524452743477347,-0.0758722843684814,-0.2868592158243379,-0.06840652801257921,0.18130172515608373,0.23197042095906292,0.060846630555839214,0.29879106223739565,0.1657917862225133,-0.06340084166027839,-0.10554315813789208,-0.025453633856428952,-0.15424695436035038,-0.0798330416472201,-0.34546866459331543,-0.19796083878635298,-0.24429253935207806,-0.04888508892592632,-0.07807507399932503,0.1945786031818561,0.20824983807869415,-0.059493898328424416,0.16977275451030813,0.06926397862000927,0.030264056309459456,0.03297984731568588,-0.02040502226331519,-0.09557498221593706,-0.19261208067178914,0.1344178994006956,0.15052179061914006,-0.25422727401532197,0.22783081816189943,-0.0708064292806404,-0.11949514308419817,-0.2881741359304193,-0.2601294709156988,0.2586415744849712,0.23422463245254338,-0.19189157417868768,-0.005825869320277163,0.11073711096038531,0.16263409898686607,-0.35197680839564705,0.24823188240629407,0.02395609929638623,0.3524273290195858,-0.05598549358357247,-0.1878866428878655,-0.0939457467646935,0.11874551310658277,-0.17421867243877723,0.008121474404464146,0.23709778076996618,-0.17949286136442022,-0.1380743003290605,-0.09377590211695268,0.09030837749249143,-0.15137367033617818,0.3861599636083361,0.19492003175152653,-0.09906975609112742,-0.10036260221639556,0.10449237862681075,0.21970455649036127,-0.1540336111921475,0.17489898729867367,-0.1384451300443867,0.22025603829966248,-0.2240668137622282,-0.33768436754682446,-0.18251957497708116,-0.011941768204330308,-0.07522510455451546,0.2084551585259222,-0.4602979735669754,-0.013587570974708453,0.18311388771192827,-0.07442393350252094,0.14143197949381467,0.25371265224631634,0.0832660059126378,0.2507597712051939,-0.17430862534982414,-0.023540939380708184,-0.019129846146320982,-0.4034572589816209,0.16210214666105316,-0.013288627431586167,0.03305491142923857,0.1713833758018361,0.40689527038310513,0.03256363524675269,-0.15468640488934982,-0.018746267964256456,0.024424006594861943,-0.22339897585082497,-0.02986096698544065,-0.13165135785294194,0.09289391032549882,-0.03720089770969352,0.1405336373617526,0.21060259898416078,0.386111063832131,-0.07041287717894831,-0.058679005323270854,-0.14416871826487737,-0.038149006975408434,0.28410651864951525,0.11364815315125552,-0.08230824968005115,0.028116214237836647,0.01102612739533567,0.353339110659234,-0.06272711082373461,0.035335282268677624,0.15895406151187874,-0.0396279537814209,-0.07734156798115682,-0.1348333693422074,-0.045378283298740225,-0.11211861844500977,0.06957920108231778,0.3142834040021694,-0.0016194710034950696,0.04198364405884125,-0.022589285910311445,-0.0415670910522019,0.05488941817084931,-0.07966677011524857,0.3299943548635655,0.007802056574527018,-0.10285686497129426,0.03153985585136338,0.02124793096302386,0.040910305605467455,0.0372623777638714,-0.1064997585428128,-0.1943402659266385,0.47623554492564035,-0.23373334539116733,0.016214264075498364,0.035547840932428604,-0.24382789415784453,-0.33484685288281635,0.006689418303338336,0.0013536145589678333,-0.16888680168636733,-0.024087595703376832,0.09394315797140138,-0.06395928767224196,0.047512978482442056,0.12621060614591453,-0.08153461694781036,-0.0461071589394543,-0.0794939857368134,-0.1613332127354354,-0.05589538777023454,-0.27692954992429114,-0.06719025009384572,-0.2938901415371085,-0.3526753638517742,0.07053418369001316,-0.25558095883617,0.06662555730196579,0.05421372248217045,-0.15433007696356266,-0.2984497650400876,0.36340946213626457,0.10495550780797713,0.2711907635692216,-0.16336689370320234,-0.033421014386447366,0.24369877015019184,-0.2170581864804365,0.013528236544602772,0.24145414789415792,0.04070739896400058,-0.23916007372861206,0.12585313284171487,-0.20376233036610633,-0.3692638871818251,0.0933476688312562,0.15904961975122334,0.20518564160907396,0.1613821493785204,-0.11329843561743383,0.0034944787781467992,0.05646972975171944,0.11939328097683385,0.2613622559571412,0.13840792215528502,-0.09863744104267255,0.06981270934430865,-0.28436263778577164,-0.13172593577984182,-0.15876044590285862,-0.08648377630364165,0.02293737960075707,-0.17006761630535902,0.15998783144534362,-0.11421156428044138,-0.35012769872948485,0.0013807769680872772,0.1180807680746894,0.03691898037133472,0.09790959282211473,-0.011645864389194162,-0.12202269105236063,0.013120230718926158,0.31675759793037156,-0.09293849792131094,-0.09408515495529354,-0.00017385033957081214,-0.03375718286673225,-0.1017355594022758,0.016038857518754716,0.22035155861362252,-0.05685031129777298,0.18844862714850846,-0.0784680450701642,-0.03424946295946676,0.004848696667086989,0.18923065281086,0.270482947784458,-0.2990133602001692,-0.178678629360313,-0.02697538365213651,-0.12707848227651736,-0.007401742394067323,-0.2453902011775002,-0.06071579156247558,-0.27584418342581885,0.14911944365437305,-0.2517871155727959,0.0924968059727195,0.3274815600757933,-0.15466677122421546,-0.07123975444531541,-0.33513437443959,0.11861543364138853,0.04217389846192917,-0.20051513972415078,0.22417052629656076,-0.13998577275362828,-0.13267764044960811,0.10484758413548252,-0.22350954989121571,0.000493209622609847,0.11792891137588563,0.23482902751152143,0.19447652760324563,0.08659886595290808,0.13251399873810993,0.15063893309428664,-0.08517496166871072,-0.01413179671831473,0.2973589296464596,0.0669017973922694,0.08470410807416964,0.19554140758576385,0.1323157261951116,-0.14102639185875818,0.03125541776198289,-0.016957079632854637,-0.19914142059971923,-0.08389976695392223,-0.08011024675167025,0.08130805671583549,0.39038886428524466,0.2501831288109871,0.11772846872997075,0.261237085418877,-0.2626814778876498,0.21791361114829813,-0.2553077466028256,-0.15066866484374597,0.16846989612662425,-0.050214384076508085,0.22783116203037485,-0.24682692533969286,-0.09881609053063797,-0.1669079938062538,0.17292319544928958,0.06035668049688222,0.12397825621119049,-0.09450181206615543,0.086552510032775,-0.06896207202949309,0.13007815049643356,0.30171535515537895,0.300750320072298,0.1349896753244068,0.3752819949673975,-0.19037840358617517,-0.011878003830252816,0.05071478296868111,-0.13682423395998042,0.43981323001249983,0.2993561228243734,-0.16659011281497685,-0.17901045465650628,-0.18965175283271873,-0.031869967862849784,-0.09875244146689466,-0.29458423274624795,-0.0776022880942865,-0.05779292843652382,0.3831485604868161,0.11849866092991473,0.35088370713799016,0.3695159621837966,0.1934357809054677,-0.15087660235465722,0.2065464880918999,0.014467158239652493,-0.031209199052240906,0.10853084500783494,0.12893106076999605,-0.2953250538028036,0.3386125426103744,0.17273582518369865,-0.05147607268921895,-0.12994777816286648,-0.17205490688006236,-0.11049572241893113,0.03547565643416248,0.21327560519389438,-0.09347585718114307,-0.13236366319178916,-0.014449676160250127,0.09996327757025697,0.013234896250313023,-0.1251757355328149,-0.1791969839426386,0.08552431264900548,-0.08752099871599031,-0.027764531875081808,-0.10528614978817791,0.15890461003010622,-0.4434224457799513,0.04997618775927713,0.03252117381868531,0.08301692528722253,-0.05132912528941316,-0.03966039071420713,0.24344562694669172,0.07670911267800035,-0.09388929171822213,0.043250412556226824,0.021748182435984198,0.21861998227139565,-0.21973005795846517,0.034333558222597566,-0.33793002323982624,-0.0070137877030881784,0.0962754989975146,-0.003256335178223324,-0.24475502126601284,0.25373163477098837,-0.4160949040879195,0.09609191067866041,-0.07965120977453363,0.3287081396221331,-0.395171324962614,0.12171130738060071,0.17952077247716955,0.3963462921502752,0.08094932435180358,-0.1899080224926303,0.2764966887700643,-0.24389505202292816,0.179535930746645,-0.09962537015635872,-0.06997787014703478,-0.3627594420250698,-0.19879501262567287,-0.09980057726457954,0.41467649689188757,0.10619846398545826,0.15111464612794484,-0.24951816139976624,-0.09997186224337867,0.01658265348561684,-0.03889977951979771,-0.02630271555274534,-0.3044541700070013,-0.07460819529753535,-0.030601076330263315,-0.12779651114853102,-0.044162536446726054,-0.16874929785925952,-0.047902854735465086,0.25537754941593394,0.24545559516729765,0.22619155405291905,-0.023513834000571603,-0.10177068706946776,-0.17103704817503795,0.033149020033955576,0.18791941337327148,0.2345099591051996,-0.03608925230549964,-0.027512713848134772,-0.050517373975596226,-0.20671517333342093,-0.15631954751548985,0.3717458087605606,0.059184338592731305,0.05198986650315407,0.15657070554025243,-0.16164120933511505,0.10104554743340848,-0.2680221283830014,0.19885312820535656,-0.019819095225265003,-0.03527814099562351,0.01751899838856589,0.19351037043578584,0.25325238928843447,-0.023440932919508828,0.22994736372879515,-0.28344977833023793,-0.3762943876511684,-0.02672490167983551,-0.14623863623006225,0.0981310743406132,0.09255155490215583,-0.2748220668563792,0.23577714029162233,-0.06678933243830043,-0.08007270861816616,0.04135190400311386,-0.2930928625156036,0.024283240701831597,-0.017212160998392526,-0.24278733160710605,-0.13613789688617595,-0.1946887670821859,0.23790528271898875,0.2986442627397116,-0.3165624062311969,0.04347786842314321,-0.1364629352205127,-0.2556827080718366,0.03538091589883961,-0.08105260670926531,-0.19569329155386886,0.07930423255480874,0.21446630138737258,-0.2569285595680385,-0.35635155305380295,-0.057229985314068026,-0.1174480733839631,0.06934680882569065,0.15899709084530395,-0.24319284467740582,-0.014783544767222878,0.02172400047080944,0.19489049039100212,-0.2364940463062398,0.18728050636840243,0.04838438366121185,-0.02739551447193246,-0.15758335027609557,0.07464762977403876,0.07870269730085186,0.12707224052694843,0.30509862121028697,-0.09211376329996096,-0.33621481524893354,-0.13919426639881843,0.060081845183167594,-0.011808329408304875,-0.1722934969126651,-0.1995994383218094,0.37135108126852195,-0.1278588032052681,0.28661093028018697,-0.19916863872950505,-0.12565413525901195,-0.019653404947983455,-0.23871855094661037,0.10722641330239085,0.10702866381681267,-0.07470387132312606,-0.12540046299215582,-0.290328964857891,-0.2309173160182505,-0.16475765151556657,0.1683913053723425,-0.12888596063365665,-0.07775207279389086,-0.07831934673685077,-0.03633977457137985,0.14163002319443121,-0.0939451836971387,0.017888389360353828,-0.36342149685031033,0.004917114012272419,-0.05095890006449578,-0.23082565757733028,-0.31180686168522465,0.04489670261903404,0.1205721909743683,-0.2866789896788085,-0.29141843638316134,-0.32379392759169445,-0.29116713833733143,0.2757793248450701,-0.1529024843762138,-0.2517107545825835,0.12040209060554655,-0.08007629714801595,0.2801568396385796,-0.14620486746148148,-0.27943691151405814,0.12051383656765127,0.14945345750036848,0.331772924268167,-0.049570920501732604,0.20136341628717092,0.10181251848037971,-0.24292574551990748,-0.06822920612141459,-0.1766699994915557,0.32029489307256587,-0.1677960614786452,-0.3130606300730275,-0.17592459065513313,-0.10774492020873566,-0.021246444150772474,0.2646278524782307,-0.2146218472884025,0.42057813480945144,-0.09569156535554305,-0.06492911695578577,0.23371886016867927,0.07331862231164338,-0.05819220668905441,-0.3148143420233251,0.01556241879805298,-0.263691727894171,-0.27970649714081547,0.2907693778003214,0.25423810891757487,-0.27337755288360055,0.4206036692565797,0.18933690350575305,0.16051124288456772,0.21934933504885473,-0.14742139841244892,0.11653013115237681,-0.002607078952117557,0.023364147462944622,-0.061734453718708136,0.2373576488194664,-0.03688349188415456,-0.0901998574612439,0.25828278115230974,-0.15538023884252733,-0.2288567190179226,0.04991757602650743,0.3318710985102722,-0.13005830583120823,-0.15651775495025466,0.11778655991663095,0.04723380287021689,0.3599131629135182,0.07354665450821903,0.11329211316304874,-0.20607238790575336,0.2262637304922579,0.023191489743870308,-0.0542609578308878,0.06330542498734229,-0.19760225191066205,-0.2019047447430098,-0.14409518201013127,0.11629335352579798,-0.10105825989908941,-0.15668335170321201,-0.11731846577446195,-0.31147457751154334,0.15276857305704405,0.2011796965061149,0.12288787012165703,0.214464951712179,-0.04077813063174473,0.19717844978646815,-0.09291662555886268,-0.1297305048201867,0.10859351569608924,-0.033886757783286904,-0.2575977654833038,-0.20586871990851732,-0.28250883991389786,0.2541278205287236,0.168463657948697,0.19707864078652354,-0.03691154600575117,0.07178810251213939,-0.015639537384475134,0.23557027351437168,0.45644398242277445,0.1349575733024555,0.28208434352169226,-0.10392986034793698,-0.33735213783603224,0.3008708322923258,0.12826730638674355,0.1720842505968893,-0.16635577169692622,0.21864226165303438,0.07732901083235483,0.2173660238521648,0.10789032551177595,0.021220133639738556,-0.2729456030196523,-0.1043716403248841,0.06293110385580625,-0.12999857076432006,-0.09519324866476234,-0.1815712707360653,0.2747826435178101,0.32780101248359894,-0.02029284586757417,0.25709305547683786,0.11735258545268258,0.12068037986832841,0.18937498095340122,-0.11417150880350707,-0.04348121718986126,0.22877573624287423,0.072735575510599,0.06567743092619621,0.2596838481705717,-0.2824967441927642,-0.1314715193984548,0.21511087910770865,-0.05817990637644286,0.1497374098941543,-0.18692183057923606,0.1268606583085329,0.1746722517377444,-0.18865750816323212,0.009605215798599145,-0.1809061286123325,0.021160387603495208,-0.04305592919307693,0.21038789990128043,0.08299035195050286,-0.1251062149378758,-0.257535503158305,-0.22817164050022548,-0.14640448526580724,0.14037981419252604,-0.0030451903602024613,0.09950984523096751,0.21338857480765622,0.01803401261704822,-0.05900816102736669,-0.08342863723290568,-0.06441987574476099,0.02888504174098755,0.016385684218198117,0.09300949108520845,0.19750726034637325,-0.07569948101181058,-0.12826978015999363,-0.025056069371404546,-0.050134857681207484,0.2707866059711283,-0.10342601816094549,0.006186062644683167,-0.03432958626501499,-0.014243206987137012,0.015234675946724417,0.06388524169076504,-0.05789736993464635,-0.07418583175105986,-0.05731225745507688,0.13940792964556076,-0.24819092598678885,-0.03510160854502315,-0.10167988040202731,-0.13465929764368695,0.11167058315559503,0.14935777417067,0.29959965173034875,-0.06549427953525776,-0.190887922262853,0.10837749701640903,-0.042529490624400486,0.2992251119823413,0.28374407593135814,-0.14174204405461294,-0.26255231507818766,-0.2192953977524269,0.20944602287778324,-0.21137128522917953,-0.2143530495495645,0.1974543245020181,0.225801267870539,0.012877393181294765,-0.0035380188200474713,-0.209417451243529,0.06562249315523758,-0.10010970760360466,0.10774362468333279,-0.16859602509729152,-0.1329147093588501,0.06647597549765019,-0.2766192097737541,-0.20102556999336044,0.11499547241504426,0.14815730403293803,-0.2617586832931406,-0.22829078963128874,-0.0889171654383884,-0.15707445411409263,0.01156549863375795,-0.15518173878866576,0.04148335842286683,-0.05000367758592866,0.23326158634492608,0.009251948782637118,-0.3186199996258633,0.12621969875121733,0.03182545414544447,0.25716523444718975,-0.08758798244685713,0.14308918612116392,-0.3757356068290556,-0.18925101673114306,-0.19363662878122853,-0.0690928238291326,0.21006346822486902,0.258746619970139,0.24684146599874734,0.26677319784127534,0.012441339251746057,-0.016135715730921908,0.068911797316982,-0.08989646873320856,0.01626928943052846,0.26167464860850026,-0.0531365734295546,0.21527202348732247,0.01679329674892037,-0.23058678334997584,-0.008631886477210311,0.028398627873755293,0.03898403053220394,-0.06581327463872583,0.22649326349670115,0.08320061689823677,0.014702375270571263,0.10708839858838479,0.3157645894754775,-0.029729072142285408,0.2528678274193468,0.03140360479461475,0.11196950011356917,-0.1283243259168602,-0.2911090199714704,-0.19857675622970444,-0.11197418447052071,0.18354795981248564,0.3301996039400003,-0.21183643446171038,-0.04556279354223263,-0.15488111018205214,-0.19062482698360572,-0.09249924840890877,0.2408098475465612,0.13546835139677355,0.039661432476195344,-0.30527678163802036,-0.0009723621673875318,-0.2447194300585559,0.038190412946686576,0.28288690552835716,0.19501188626050506,0.271525394207092,-0.011609700479049807,-0.05064919268746725,0.15468818985893795,-0.37132962504934813,-0.27513740103532847,-0.05665102586576342,0.02123664934444719,0.07214169843505455,-0.018379748260789266,-0.19884496103368363,0.2826099304057101,0.2850991882018113,-0.22368117879431465,-0.07056080994328706,-0.35884989960474245,0.0032677884246387068,-0.18056962569824278,0.37616609616389474,0.02056385785362285,-0.024400964672547647,0.02848625880121823,-0.22507379630567076,0.32964206630551457,0.1342797926469101,-0.028082260599165867,-0.007349378794095682,0.13129784030038316,0.32594542546838245,-0.06283509668143226,0.04737265145411702,0.16739358523622183,0.18365302496868469,0.2224812722418003,-0.011570426478000814,0.09690653519628432,-0.0005364262964729777,-0.28602864506305187,-0.0676074202271714,-0.08739060681109442,-0.16252022484120954,0.14404839732241345,-0.3384249318055285,-0.25874109952459867,-0.12092931709777074,-0.21625172627824865,-0.11059659274165487,-0.26132072486068003,0.020461803608906988,0.03270140041498806,-0.009906480368468827,-0.17089093357040208,0.08298603028715906,-0.05644888981566585,-0.3768285639596234,0.019990478864689784,0.2750014287538148,0.26152332756929986,-0.2276003249855587,-0.14980262493679058,0.22105084929811808,0.31568599954900417,-0.26471947414523367,0.017552408483382173,-0.1159447079232355,0.25417086922853493,0.039551240327200726,0.3624273530127437,0.038208091786928904,-0.14548438230485938,0.037511487264964956,0.10020455839633977,0.008027815462985005,0.13663221736626005,-0.04331709989907375,-0.3984035250831626,0.21188638616402175,0.16451715585366422,-0.01855555703822593,0.13452697466831662,-0.06230481280411694,-0.040244611242711975,-0.08852373048411895,-0.26687311523676427,-0.42777946134022415,0.059009836273053215,0.045300145098442844,0.37568271250427054,-0.02654430522917768,-0.17595346687179508,0.13208800167918838,0.1581535738905873,0.06924952350466049,0.18181120306825405,0.16545048540075438,-0.2500692841044673,-0.1826560206488162,0.21344711060738292,0.08067724900134762,-0.3358995492349694,0.0621213925082051,-0.10017891041806973,-0.09282965805755325,-0.20896776573102288,0.19790281049621983,0.032123884449839206,0.0790919018801387,-0.03166180374160751,0.15612073681233385,-0.19944252077667884,-0.21050381412040065,-0.19301490975068755,-0.1168862257177321,0.20652071538078395,-0.03670490014828127,-0.07740124686202587,0.1482998354491721,-0.38936035911832,0.10508766238649689,0.3597239717575698,0.13214995099240356,-0.13059702951632157,-0.3219492277273003,0.1373774006554017,0.1812643951612269,-0.09107940228529585,-0.06708441597517686,-0.14695426697609054,-0.09599181997621434,0.00013044789651304317,-0.13163738783353995,0.03787264472855958,0.44543620240046894,-0.0657363536391363,-0.14786022700455004,-0.18996365734453363,0.131439517360306,0.34335417741223845,-0.09336394267983227,0.1912693059552048,0.2431165964235984,0.08180589960279219,-0.30130223404480033,0.27101731148913993,-0.09830002292714302,0.008180353754821353,-0.26677910352469547,0.15793195314343378,-0.16067189381870484,-0.3565700419352344,-0.04713521148120513,-0.3393374031384979,-0.04570923086508866,0.3677402363225356,-0.27573102541106786,-0.052629614205669764,0.12738353348724385,-0.1817682063339346,0.3629589060607825,-0.43807403135627465,-0.09825982415572077,0.02365043248946159,0.029306041118786992,-0.13013820095721046,0.16910495242917398,0.01681811123735536,-0.18034303192752288,-0.09895865712972925,0.15116605418775303,-0.03880053365278314,0.03981278313562165,-0.03315163834814637,-0.009811799067105852,-0.3641560318353462,0.2800521306206905,0.1261956584124121,-0.5493105046615903,0.3040521604249754,-0.1323469811514147,0.1837362956922365,0.21700116454274734,-0.13830921320696954,-0.07906384155038312,-0.2708310733114859,-0.314203172273565,0.11072391962949274,0.17161824567643957,-0.2727491348284719});
}


inline float LR::dot (const vector<float> & vec1, const vector<float> & vec2) {

#ifndef NO_NEON
    int len = vec1.size();
    const float * p_vec1 = &vec1[0];
    const float * p_vec2 = &vec2[0];


    float sum=0;
    float32x4_t sum_vec = vdupq_n_f32(0),left_vec,right_vec;
    for(int i = 0; i < len; i += 4)
    {
        left_vec = vld1q_f32(p_vec1 + i);
        right_vec = vld1q_f32(p_vec2 + i);
        sum_vec = vmlaq_f32(sum_vec, left_vec, right_vec);
    }

    float32x2_t r = vadd_f32(vget_high_f32(sum_vec), vget_low_f32(sum_vec));
    sum += vget_lane_f32(vpadd_f32(r,r),0);
#else
    float sum = 0.0;
    for (int i = 0; i < vec1.size(); i++) {
        sum += vec1[i] * vec2[i];
    }
#endif
    return sum;
}


inline float LR::sigmoidCalc(const float wxb) {
    float expv = exp(-1 * wxb);
    float expvInv = 1 / (1 + expv);
    return expvInv;
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

inline vector<Data> LR::LoadData(const string & filename, bool last_label_exist)
{

    FILE * fp = NULL;
    char * line, * record;
    char buffer[8192];


    if ((fp = fopen(filename.c_str(), "rb")) == NULL) {
        printf("file [%s] doesnnot exist \n", filename.c_str());
        exit(1);
    }

    vector<Data> data_set;
    int label0_cnt = 0;
    int label1_cnt = 0;

    while ((line = fgets(buffer, sizeof(buffer), fp)) != NULL) {

        if (last_label_exist && label0_cnt >= read_line_num0 && label1_cnt >= read_line_num1) break;

        vector<float> feature(features_num + (last_label_exist? 1: 0), 0.0);

        int f_cnt = 0;
        record = strtok(line, ",");
        int flag = 0;
        while (record != NULL) {
//            printf("%s ", record);
            feature[f_cnt++] = atof(record);
            if (feature[f_cnt - 1] < 0) {
                flag = 1;
                break;
            }
            record = strtok(NULL, ","); // 每个特征值
        }

        if (flag) continue;


        if (last_label_exist && (label0_cnt < read_line_num0 || label1_cnt < read_line_num1)) {
            int ftf = (int) feature.back();
            feature.pop_back();
            data_set.push_back(Data(feature, ftf));

            if (ftf == 0) {
                label0_cnt++;
            } else {
                label1_cnt++;
            }

        } else {
            data_set.emplace_back(Data(feature, 0));
        }
//        if (last_label_exist) {
//            int ftf = (int) feature.back();
//            feature.pop_back();
//
//            if (ftf == 0 && label0_cnt < read_line_num0 ) {
//                data_set.push_back(Data(feature, ftf));
//                label0_cnt++;
//            } else if (ftf == 1 && label1_cnt < read_line_num1) {
//                data_set.push_back(Data(feature, ftf));
//                label1_cnt++;
//            }

//        } else {
//            data_set.push_back(Data(feature, 0));
//        }
    }
    fclose(fp);
    fp = NULL;

#ifdef TEST
    printf("0 数量: %d, 1 数量: %d \n", label0_cnt, label1_cnt);
#endif

    return data_set;
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


    LR logist(train_file, test_file, predict_file);


    logist.train();

    logist.predict();


#ifdef TEST
    clock_t end_time = clock();
    printf("总耗时（s）: %f \n", (double) (end_time - start_time) / CLOCKS_PER_SEC);
#endif

#ifdef TEST
    Test(answer_file, predict_file);
#endif

    return 0;
}