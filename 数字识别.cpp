#include<unordered_map>
#include<filesystem>
#include<windows.h>
#include<algorithm>
#include<iostream>
#include<fstream>
#include<iomanip>
#include<cstring>
#include<chrono>
#include<math.h>
#include<vector>
#include<random>
#include<string>
#include<atomic>
#include<omp.h>
#include<cmath>
using namespace std;
namespace fs=std::filesystem;

static std::unordered_map<std::string,fs::file_time_type>fileModifyTimes;
static std::unordered_map<std::string,std::uintmax_t>fileSizes;

#define INPUT_SIZE 784
#define HIDDEN_SIZE 2048
#define OUTPUT_SIZE 10

float input[INPUT_SIZE];
float qz1[INPUT_SIZE][HIDDEN_SIZE];
float ans1[HIDDEN_SIZE];
float qz2[HIDDEN_SIZE][OUTPUT_SIZE];
float ans2[OUTPUT_SIZE];
float wc1[HIDDEN_SIZE];
float wc2[OUTPUT_SIZE];

enum OptimizerType
{
	SGD,
	MOMENTUM,
	ADAM
};

OptimizerType current_optimizer = ADAM;
float learning_rate = 0.00002f;
float momentum_beta = 0.9f;
float adam_beta1 = 0.9f;
float adam_beta2 = 0.999f;
float adam_epsilon = 1e-6f;
float lambda_l2 = 0.0000f;

float m_qz1[INPUT_SIZE][HIDDEN_SIZE] = {0};
float m_qz2[HIDDEN_SIZE][OUTPUT_SIZE] = {0};
float m1_qz1[INPUT_SIZE][HIDDEN_SIZE] = {0};
float m1_qz2[HIDDEN_SIZE][OUTPUT_SIZE] = {0};
float m2_qz1[INPUT_SIZE][HIDDEN_SIZE] = {0};
float m2_qz2[HIDDEN_SIZE][OUTPUT_SIZE] = {0};
std::atomic<int> adam_t(0);

bool rans[OUTPUT_SIZE];
short rians;
short ans;
vector<string>trainingnumberfilepathlist;
vector<short>trainingnumberfileans;
vector<int>indices;

bool data_loaded = false;
vector<vector<float>> global_preloaded_data;
vector<short> global_preloaded_labels;
int global_total_samples = 0;

inline float relu(float x)
{
	return x > 0 ? x : 0;
}
inline float backrelu(float x)
{
	return x > 0 ? 1 : 0;
}
inline void softmax()
{
	float maxval = ans2[0];
	for(int i = 1; i < OUTPUT_SIZE; i++)
	{
		maxval = max(maxval, ans2[i]);
	}
	float anssum = 0;
	for(int i = 0; i < OUTPUT_SIZE; i++)
	{
		ans2[i] = exp(ans2[i] - maxval);
		anssum += ans2[i];
	}
	for(int i = 0; i < OUTPUT_SIZE; i++)
	{
		ans2[i] = ans2[i] / anssum;
	}
	return;
}
inline float Cross_Entropy()
{
	return -log(ans2[rians] + 1e-7f);
}
void Forward_Propagation()
{
	for(int i = 0; i < HIDDEN_SIZE; i++)
	{
		float sum = 0;
		for(int j = 0; j < INPUT_SIZE; j++)
		{
			sum += input[j] * qz1[j][i];
		}
		ans1[i] = relu(sum);
	}
	for(int i = 0; i < OUTPUT_SIZE; i++)
	{
		float sum = 0;
		for(int j = 0; j < HIDDEN_SIZE; j++)
		{
			sum += ans1[j] * qz2[j][i];
		}
		ans2[i] = sum;
	}
	softmax();
	ans = 0;
	for(int i = 1; i < OUTPUT_SIZE; i++)
	{
		if(ans2[ans] < ans2[i])
		{
			ans = i;
		}
	}
	return;
}
void Compute_Gradients()
{
	for(int i = 0; i < OUTPUT_SIZE; i++)
	{
		wc2[i] = ans2[i] - rans[i];
	}
	for(int i = 0; i < HIDDEN_SIZE; i++)
	{
		float sum = 0;
		for(int j = 0; j < OUTPUT_SIZE; j++)
		{
			sum += wc2[j] * qz2[i][j];
		}
		wc1[i] = sum * backrelu(ans1[i]);
	}
}
void Update_Weights_SGD()
{
	for(int i = 0; i < HIDDEN_SIZE; i++)
	{
		for(int j = 0; j < INPUT_SIZE; j++)
		{
			float grad = wc1[i] * input[j] + lambda_l2 * qz1[j][i];
			qz1[j][i] = qz1[j][i] - learning_rate * grad;
		}
	}
	for(int i = 0; i < OUTPUT_SIZE; i++)
	{
		for(int j = 0; j < HIDDEN_SIZE; j++)
		{
			float grad = wc2[i] * ans1[j] + lambda_l2 * qz2[j][i];
			qz2[j][i] = qz2[j][i] - learning_rate * grad;
		}
	}
}
void Update_Weights_Momentum()
{
	for(int i = 0; i < HIDDEN_SIZE; i++)
	{
		for(int j = 0; j < INPUT_SIZE; j++)
		{
			float grad = wc1[i] * input[j] + lambda_l2 * qz1[j][i];
			m_qz1[j][i] = momentum_beta * m_qz1[j][i] + learning_rate * grad;
			qz1[j][i] = qz1[j][i] - m_qz1[j][i];
		}
	}
	for(int i = 0; i < OUTPUT_SIZE; i++)
	{
		for(int j = 0; j < HIDDEN_SIZE; j++)
		{
			float grad = wc2[i] * ans1[j] + lambda_l2 * qz2[j][i];
			m_qz2[j][i] = momentum_beta * m_qz2[j][i] + learning_rate * grad;
			qz2[j][i] = qz2[j][i] - m_qz2[j][i];
		}
	}
}
void Update_Weights_Adam()
{
	int t = adam_t.fetch_add(1) + 1;
	float beta1_t = 1.0f - pow(adam_beta1, t);
	float beta2_t = 1.0f - pow(adam_beta2, t);
	
	for(int i = 0; i < HIDDEN_SIZE; i++)
	{
		for(int j = 0; j < INPUT_SIZE; j++)
		{
			float grad = wc1[i] * input[j] + lambda_l2 * qz1[j][i];
			m1_qz1[j][i] = adam_beta1 * m1_qz1[j][i] + (1.0f - adam_beta1) * grad;
			m2_qz1[j][i] = adam_beta2 * m2_qz1[j][i] + (1.0f - adam_beta2) * grad * grad;
			float m1_hat = m1_qz1[j][i] / beta1_t;
			float m2_hat = m2_qz1[j][i] / beta2_t;
			qz1[j][i] = qz1[j][i] - learning_rate * m1_hat / (sqrt(m2_hat) + adam_epsilon);
		}
	}
	for(int i = 0; i < OUTPUT_SIZE; i++)
	{
		for(int j = 0; j < HIDDEN_SIZE; j++)
		{
			float grad = wc2[i] * ans1[j] + lambda_l2 * qz2[j][i];
			m1_qz2[j][i] = adam_beta1 * m1_qz2[j][i] + (1.0f - adam_beta1) * grad;
			m2_qz2[j][i] = adam_beta2 * m2_qz2[j][i] + (1.0f - adam_beta2) * grad * grad;
			float m1_hat = m1_qz2[j][i] / beta1_t;
			float m2_hat = m2_qz2[j][i] / beta2_t;
			qz2[j][i] = qz2[j][i] - learning_rate * m1_hat / (sqrt(m2_hat) + adam_epsilon);
		}
	}
}
void Backward_Propagation()
{
	Compute_Gradients();
	switch(current_optimizer)
	{
		case SGD:
			Update_Weights_SGD();
			break;
		case MOMENTUM:
			Update_Weights_Momentum();
			break;
		case ADAM:
			Update_Weights_Adam();
			break;
	}
}
void initialize_weights()
{
	random_device rd;
	default_random_engine generator(rd());
	normal_distribution<float> dist1(0.0, sqrt(2.0 / INPUT_SIZE) / 0.5);
	normal_distribution<float> dist2(0.0, sqrt(2.0 / HIDDEN_SIZE));
	for(int i = 0; i < INPUT_SIZE; i++)
	{
		for(int j = 0; j < HIDDEN_SIZE; j++)
		{
			qz1[i][j] = dist1(generator);
		}
	}
	for(int i = 0; i < HIDDEN_SIZE; i++)
	{
		for(int j = 0; j < OUTPUT_SIZE; j++)
		{
			qz2[i][j] = dist2(generator);
		}
	}
	memset(m_qz1, 0, sizeof(m_qz1));
	memset(m_qz2, 0, sizeof(m_qz2));
	memset(m1_qz1, 0, sizeof(m1_qz1));
	memset(m1_qz2, 0, sizeof(m1_qz2));
	memset(m2_qz1, 0, sizeof(m2_qz1));
	memset(m2_qz2, 0, sizeof(m2_qz2));
	adam_t.store(0);
	return;
}
void Save_Weights_Binary()
{
	ofstream w("w.txt", ios::binary);
	w.write((char*)qz1, sizeof(qz1));
	w.write((char*)qz2, sizeof(qz2));
	w.close();
	ofstream opt("optimizer_state.bin", ios::binary);
	int adam_t_value = adam_t.load();
	opt.write((char*)&current_optimizer, sizeof(current_optimizer));
	opt.write((char*)&adam_t_value, sizeof(adam_t_value));
	opt.write((char*)&lambda_l2, sizeof(lambda_l2));
	opt.write((char*)m1_qz1, sizeof(m1_qz1));
	opt.write((char*)m1_qz2, sizeof(m1_qz2));
	opt.write((char*)m2_qz1, sizeof(m2_qz1));
	opt.write((char*)m2_qz2, sizeof(m2_qz2));
	opt.close();
	return;
}
void Load_Weights_Binary()
{
	ifstream w("w.txt", ios::binary);
	if(!w)
	{
		initialize_weights();
		Save_Weights_Binary();
		return;
	}
	w.read((char*)qz1, sizeof(qz1));
	w.read((char*)qz2, sizeof(qz2));
	w.close();
	ifstream opt("optimizer_state.bin", ios::binary);
	if(opt)
	{
		int saved_adam_t;
		opt.read((char*)&current_optimizer, sizeof(current_optimizer));
		opt.read((char*)&saved_adam_t, sizeof(saved_adam_t));
		opt.read((char*)&lambda_l2, sizeof(lambda_l2));
		opt.read((char*)m1_qz1, sizeof(m1_qz1));
		opt.read((char*)m1_qz2, sizeof(m1_qz2));
		opt.read((char*)m2_qz1, sizeof(m2_qz1));
		opt.read((char*)m2_qz2, sizeof(m2_qz2));
		opt.close();
		if(saved_adam_t < 0 || saved_adam_t > 100000)
		{
			adam_t.store(0);
			memset(m1_qz1, 0, sizeof(m1_qz1));
			memset(m1_qz2, 0, sizeof(m1_qz2));
			memset(m2_qz1, 0, sizeof(m2_qz1));
			memset(m2_qz2, 0, sizeof(m2_qz2));
		}
		else
		{
			adam_t.store(saved_adam_t);
		}
	}
	else
	{
		adam_t.store(0);
		memset(m1_qz1, 0, sizeof(m1_qz1));
		memset(m1_qz2, 0, sizeof(m1_qz2));
		memset(m2_qz1, 0, sizeof(m2_qz1));
		memset(m2_qz2, 0, sizeof(m2_qz2));
	}
	return;
}
bool Load_Number(const string& numberpath)
{
	ifstream w(numberpath);
	if(!w.is_open())
	{
		return 0;
	}
	for(int i = 0; i < INPUT_SIZE; i++)
	{
		int pixel_value;
		w >> pixel_value;
		input[i] = (pixel_value > 128) ? 1.0f : 0.0f;
	}
	return 1;
}
bool canopen(const string& filepath)
{
	ifstream w(filepath);
	return w.is_open();
}
void paint()
{
	cout<<"=================\n";
	cout<<"= MNIST数字识别 =\n";
	cout<<"=================\n";
	cout<<"网络结构: 784->2048->10\n";
	cout<<"优化器: ";
	switch(current_optimizer)
	{
		case SGD:
			cout << "SGD";
			break;
		case MOMENTUM:
			cout << "Momentum";
			break;
		case ADAM:
			cout << "Adam";
			break;
	}
	cout << " (学习率: " << learning_rate << ")\n";
	cout<<"L2正则化: " << lambda_l2 << " (输入二值化: 0/1)\n";
	cout<<"   1.预测结果\n";
	cout<<"   2.手动训练\n";
	cout<<"   3.批量训练\n";
	cout<<"   4.保存权重\n";
	cout<<"   5.对拍识别\n";
	cout<<"   6.设置优化器和学习率\n";
	cout<<"   7.重新初始化网络\n";
	cout<<"   8.设置L2正则化参数\n";
	return;
}
void Set_Optimizer()
{
	system("cls");
	cout<<"=== 优化器设置 ===\n";
	cout<<"当前优化器: ";
	switch(current_optimizer)
	{
		case SGD:
			cout << "SGD";
			break;
		case MOMENTUM:
			cout << "Momentum";
			break;
		case ADAM:
			cout << "Adam";
			break;
	}
	cout<<"\n\n选择选项:\n";
	cout<<"1. SGD (随机梯度下降)\n";
	cout<<"2. Momentum (动量优化)\n";
	cout<<"3. Adam (自适应矩估计)\n";
	cout<<"4. 设置学习率\n";
	cout<<"5. 设置L2正则化参数 (当前: " << lambda_l2 << ")\n";
	int choice;
	cin>>choice;
	if(choice == 1)
	{
		current_optimizer = SGD;
		cout<<"已切换为SGD优化器\n";
	}
	else if(choice == 2)
	{
		current_optimizer = MOMENTUM;
		cout<<"已切换为Momentum优化器\n";
		cout<<"动量系数beta: " << momentum_beta << endl;
	}
	else if(choice == 3)
	{
		current_optimizer = ADAM;
		cout<<"已切换为Adam优化器\n";
		cout<<"beta1: " << adam_beta1 << ", beta2: " << adam_beta2 << endl;
	}
	else if(choice == 4)
	{
		cout<<"当前学习率: " << learning_rate << endl;
		cout<<"输入新的学习率: ";
		cin >> learning_rate;
		cout<<"学习率已设置为: " << learning_rate << endl;
	}
	else if(choice == 5)
	{
		cout<<"当前L2正则化参数: " << lambda_l2 << endl;
		cout<<"输入新的L2参数 (推荐范围: 0.0001 ~ 0.01): ";
		cin >> lambda_l2;
		cout<<"L2正则化参数已设置为: " << lambda_l2 << endl;
	}
	system("pause");
}
bool filewa(const string& filepath)
{
	try
	{
		if (!fs::exists(filepath))
		{
			std::cout << "文件不存在: " << filepath << std::endl;
			return false;
		}
		auto currentModifyTime = fs::last_write_time(filepath);
		auto currentSize = fs::file_size(filepath);
		if (fileModifyTimes.find(filepath) == fileModifyTimes.end())
		{
			fileModifyTimes[filepath] = currentModifyTime;
			fileSizes[filepath] = currentSize;
			return true;
		}
		bool isModified = false;
		if (currentModifyTime != fileModifyTimes[filepath])
		{
			isModified = true;
			std::cout << "文件修改时间已改变" << std::endl;
		}
		if (currentSize != fileSizes[filepath])
		{
			isModified = true;
			std::cout << "文件大小已改变: " << fileSizes[filepath] << " -> " << currentSize << std::endl;
		}
		fileModifyTimes[filepath] = currentModifyTime;
		fileSizes[filepath] = currentSize;
		return isModified;
	}
	catch (const std::exception& e)
	{
		std::cerr << "检查文件时出错(" << filepath << "): " << e.what() << std::endl;
		return false;
	}
}
void dpclut()
{
	string dpfilepath;
	system("cls");
	cout<<"输入对拍数据文件地址：";
	cin>>dpfilepath;
	while(1)
	{
		Sleep(500);
		if(filewa(dpfilepath))
		{
			if(Load_Number(dpfilepath))
			{
				Forward_Propagation();
				system("cls");
				cout<<"网络预测结果:\n";
				if(ans == 0)
				{
					cout<<"[";
				}
				for(int i = 0; i < OUTPUT_SIZE; i++)
				{
					cout<<fixed<<setprecision(4)<<ans2[i];
					if(i == ans)
					{
						cout<<"] ";
					}
					else if(i + 1 == ans)
					{
						cout<<" [";
					}
					else
					{
						cout<<" ";
					}
				}
				cout<<endl<<"预测数字: "<<ans<<endl;
			}
			else
			{
				cout<<"文件打开失败！";
				system("pause");
				return;
			}
		}
	}
	return;
}

void Load_All_Data(const string& rootpath)
{
    if(data_loaded) return;
    
    string trainingnumberfilerootpath = rootpath + "\\";
    cout<<"正在扫描文件夹..."<<endl;
    vector<string> file_list;
    vector<short> file_labels;
    
    for(const auto& entry : fs::directory_iterator(trainingnumberfilerootpath))
    {
        if(entry.path().extension() == ".txt")
        {
            string filename = entry.path().stem().string();
            size_t pos = filename.find('_');
            if(pos != string::npos)
            {
                int label = stoi(filename.substr(0, pos));
                if(label >= 0 && label <= 9)
                {
                    file_list.push_back(entry.path().string());
                    file_labels.push_back(label);
                }
            }
        }
    }
    
    global_total_samples = file_list.size();
    cout<<"找到 "<<global_total_samples<<" 个有效文件"<<endl;
    cout<<"\n正在预加载训练数据到内存..."<<endl;
    
    global_preloaded_data.resize(global_total_samples);
    global_preloaded_labels = file_labels;
    
    #pragma omp parallel for
    for(int i = 0; i < global_total_samples; i++)
    {
        ifstream file(file_list[i]);
        vector<float> pixels(INPUT_SIZE);
        for(int j = 0; j < INPUT_SIZE; j++)
        {
            int val;
            file >> val;
            pixels[j] = (val > 128) ? 1.0f : 0.0f;
        }
        global_preloaded_data[i] = pixels;
        
        #pragma omp critical
        {
            cout<<"\r预加载进度: "<<i+1<<"/"<<global_total_samples
                <<" ("<<(i+1)*100/global_total_samples<<"%)";
            cout.flush();
        }
    }
    
    cout<<"\n预加载完成！"<<endl;
    data_loaded = true;
}

void Batch_Training()
{
	system("cls");
	
	if(!data_loaded)
	{
		string trainingnumberfilerootpath;
		cout<<"训练数据文件根目录:";
		cin>>trainingnumberfilerootpath;
		Load_All_Data(trainingnumberfilerootpath);
	}
	
	int total_samples = global_total_samples;
	vector<short>& preloaded_labels = global_preloaded_labels;
	vector<vector<float>>& preloaded_data = global_preloaded_data;
	
	vector<int> train_indices(total_samples);
	for(int i = 0; i < total_samples; i++)
	{
		train_indices[i] = i;
	}
	random_shuffle(train_indices.begin(), train_indices.end());
	cout<<"优化器: Adam, 学习率: "<<learning_rate<<endl;
	cout<<"L2正则化系数: "<<lambda_l2<<endl;
	cout<<"批量大小: 128"<<endl;
	cout<<"数据总数: "<<total_samples<<endl;
	cout<<"按任意键开始训练...";
	system("pause");
	int correct = 0;
	float total_loss = 0;
	int epoch = 0;
	int batch_size = 128;
	int t = clock();
	while(true)
	{
		epoch++;
		int epoch_correct = 0;
		float epoch_loss = 0;
		random_shuffle(train_indices.begin(), train_indices.end());
		system("cls");
		cout<<"=== Epoch "<<epoch<<" 开始 ==="<<endl;
		int total_batches = (total_samples + batch_size - 1) / batch_size;
		for(int batch_start = 0; batch_start < total_samples; batch_start += batch_size)
		{
			int batch_end = min(batch_start + batch_size, total_samples);
			int current_batch = batch_start / batch_size + 1;
			int local_correct = 0;
			float local_loss = 0;
			for(int b = batch_start; b < batch_end; b++)
			{
				int idx = train_indices[b];
				rians = preloaded_labels[idx];
				memcpy(input, preloaded_data[idx].data(), INPUT_SIZE * sizeof(float));
				Forward_Propagation();
				if(ans == rians)
				{
					local_correct++;
				}
				local_loss += Cross_Entropy();
				for(int j = 0; j < OUTPUT_SIZE; j++)
				{
					rans[j] = (j == rians);
				}
				Compute_Gradients();
				Backward_Propagation();
			}
			epoch_correct += local_correct;
			epoch_loss += local_loss;
			if(current_batch % 5 == 0 || current_batch == total_batches)
			{
				cout<<"\r处理批次: "<<current_batch<<"/"<<total_batches;
				cout<<" [";
				int bar_width = 50;
				int progress = (current_batch * bar_width) / total_batches;
				for(int i = 0; i < bar_width; i++)
				{
					cout<<(i < progress ? "=" : i == progress ? ">" : " ");
				}
				cout<<"] "<<(current_batch * 100) / total_batches<<"%";
				cout.flush();
			}
		}
		correct += epoch_correct;
		total_loss += epoch_loss;
		cout<<"\n\n=== Epoch "<<epoch<<" 完成 ==="<<endl;
		cout<<"用时: "<<clock() - t <<endl;
		cout<<"学习率: "<<learning_rate<<endl;
		cout<<"本epoch准确率: "<<epoch_correct * 100.0 / total_samples<<"%"<<endl;
		cout<<"本epoch平均损失: "<<epoch_loss / total_samples<<endl;
		cout<<"累计准确率: "<<correct * 100.0 / (total_samples * epoch)<<"%"<<endl;
		cout<<"累计平均损失: "<<total_loss / (total_samples * epoch)<<endl;
		if(epoch % 5 == 0)
		{
			Save_Weights_Binary();
			cout<<"权重已保存"<<endl;
		}
		cout<<"\n1. 继续训练下一个epoch"<<endl;
		cout<<"2. 停止训练"<<endl;
		cout<<"选择: ";
		int choice;
		cin>>choice;
		if(choice == 2)
		{
			cout<<"\n训练完成！共训练"<<epoch<<"个epoch\n";
			system("pause");
			return;
		}
	}
}
int main()
{
	omp_set_num_threads(4);
	cout<<"加载中...";
	Load_Weights_Binary();
	system("cls");
	cout << fixed << setprecision(4);
	paint();
	short e;
	while(cin>>e)
	{
		if(e == 1)
		{
			system("cls");
			string numberpath;
			cout<<"请输入数据文件地址:";
			cin>>numberpath;
			cout<<"请输入真实标签(0-9，按-1跳过):";
			cin>>rians;
			if(Load_Number(numberpath))
			{
				Forward_Propagation();
				cout<<"网络预测结果:\n概率分布: ";
				if(ans == 0)
				{
					cout<<"[";
				}
				for(int i = 0; i < OUTPUT_SIZE; i++)
				{
					cout<<ans2[i];
					if(i == ans)
					{
						cout<<"] ";
					}
					else if(i + 1 == ans)
					{
						cout<<" [";
					}
					else
					{
						cout<<" ";
					}
				}
				cout<<endl<<"预测数字: "<<ans<<endl;
				if(rians >= 0 && rians < OUTPUT_SIZE)
				{
					cout<<"交叉熵损失:"<<Cross_Entropy()<<endl;
					cout<<"是否正确: "<<(ans == rians ? "是" : "否")<<endl;
				}
				system("pause");
			}
			else
			{
				cout<<"读入数据文件错误!";
				system("pause");
			}
		}
		else if(e == 2)
		{
			system("cls");
			string numberpath;
			cout<<"请输入训练数据文件地址:";
			cin>>numberpath;
			cout<<"请输入数据文件答案(0-9):";
			cin>>rians;
			if(Load_Number(numberpath))
			{
				Forward_Propagation();
				cout<<"训练前预测结果: ";
				if(ans == 0)
				{
					cout<<"[";
				}
				for(int i = 0; i < OUTPUT_SIZE; i++)
				{
					cout<<ans2[i];
					if(i == ans)
					{
						cout<<"]";
					}
					else if(i + 1 == ans)
					{
						cout<<"[";
					}
					else
					{
						cout<<" ";
					}
				}
				cout<<endl<<"预测: "<<ans<<", 实际: "<<rians<<endl;
				cout<<"训练前交叉熵损失:"<<Cross_Entropy()<<endl;
				for(int i = 0; i < OUTPUT_SIZE; i++)
				{
					rans[i] = (i == rians);
				}
				Backward_Propagation();
				Forward_Propagation();
				cout<<"训练后预测结果: ";
				if(ans == 0)
				{
					cout<<"[";
				}
				for(int i = 0; i < OUTPUT_SIZE; i++)
				{
					cout<<ans2[i];
					if(i == ans)
					{
						cout<<"]";
					}
					else if(i + 1 == ans)
					{
						cout<<"[";
					}
					else
					{
						cout<<" ";
					}
				}
				cout<<endl<<"预测: "<<ans<<", 实际: "<<rians<<endl;
				cout<<"训练后交叉熵损失:"<<Cross_Entropy()<<endl;
				system("pause");
			}
			else
			{
				cout<<"读入训练数据文件错误!";
				system("pause");
			}
		}
		else if(e == 3)
		{
			Batch_Training();
		}
		else if(e == 4)
		{
			Save_Weights_Binary();
			cout<<"权重已保存到 w.txt\n";
			cout<<"优化器状态已保存\n";
			system("pause");
		}
		else if(e == 5)
		{
			dpclut();
		}
		else if(e == 6)
		{
			Set_Optimizer();
		}
		else if(e == 7)
		{
			initialize_weights();
			cout<<"网络已重新初始化！\n";
			system("pause");
		}
		else if(e == 8)
		{
			system("cls");
			cout<<"=== 设置L2正则化参数 ===\n";
			cout<<"当前L2正则化参数: " << lambda_l2 << endl;
			cout<<"L2正则化用于防止过拟合，参数越大正则化效果越强\n";
			cout<<"推荐范围: 0.0001 ~ 0.01\n";
			cout<<"输入新的L2参数: ";
			cin >> lambda_l2;
			cout<<"L2正则化参数已设置为: " << lambda_l2 << endl;
			system("pause");
		}
		system("cls");
		paint();
	}
	return 0;
}
