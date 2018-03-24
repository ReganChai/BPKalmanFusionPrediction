// BPKalmanFusionPrediction.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <stdlib.h>

using namespace std;
using namespace cv;
using namespace ml;

char ci, co;
float trans = 0;
int index1 = 0, index2 = 0, index3 = 0;
int number = 0;	//计算样本个数
float inputData[300], outputData[300];
stringstream si, so;
float storage1[2000] = { 0 };
float storage2[2000] = { 0 };
int i = 0;
vector<Point> Kalman;
vector<Point> BP;
vector<Point> BP_Kalman;
int main()
{
	// 1、 读取输入样本
	ifstream read_Input;
	read_Input.open("inPut3.txt");

	while (!read_Input.eof()) {
		read_Input >> ci;
		if (ci != ',')
			si << ci;

		else {
			number++;
			si >> trans;
			storage1[index1] = trans;   //将读取到的样本坐标值存入数组
			index1++;
			trans = 0;
			si.clear();
		}
	}
	index1 = 0;
	read_Input.close();


	// 做差
	int num_after1 = (number - 3) - (number - 3) % 4;
	for (int i = 0; i < num_after1; i++) {
		if (i % 2 == 0) {
			float delta_x = storage1[i + 2] - storage1[i];
			//delta_x = delta_x / 100;
			inputData[index1] = delta_x;
		}
		else {
			float delta_y = storage1[i + 2] - storage1[i];
			//delta_y = delta_y / 100;
			inputData[index1] = delta_y;
		}
		index1++;

		if ((index1 % 4 == 0) && (index1 < 2 * (num_after1 - 2))) {
			inputData[index1] = inputData[index1 - 2];
			inputData[index1 + 1] = inputData[index1 - 1];

			index1 += 2;
		}
	}
	number = 0;

	int row1 = index1 / 4;
	cout << "训练样本对为：" << row1 << endl << endl;
	index1 = 0;
	Mat trainingDataMat(row1, 4, CV_32FC1, inputData);  //最终训练的输入样本
	//cout << trainingDataMat << endl << endl;
	//system("pause");

	// 2、读取输出样本
	ifstream read_Output;
	read_Output.open("outPut3.txt");

	while (!read_Output.eof()) {
		read_Output >> co;
		if (co != ',')
			so << co;
		else {
			number++;
			so >> trans;
			storage2[index2] = trans;
			index2++;
			trans = 0;
			so.clear();
		}
	}
	index2 = 0;
	read_Output.close();

	// 做差
	int num_after2 = 2 * row1;
	for (int i = 0; i < num_after2; i++) {
		if (i % 2 == 0) {
			float delta_x = storage2[i + 2] - storage2[i];
			//delta_x = delta_x / 100;
			outputData[index2] = delta_x;
		}
		else {
			float delta_y = storage2[i + 2] - storage2[i];
			//delta_y = delta_y / 100;
			outputData[index2] = delta_y;
		}
		index2++;
	}
	index2 = 0;
	number = 0;
	Mat trainingDataMat_res(row1, 2, CV_32FC1, outputData);  //最终的输出样本
	//cout << trainingDataMat_res << endl << endl;

	// 3、构建神经网络
	Mat layerSizes = (Mat_<int>(1, 3) << 4, 9, 2);
	Ptr<ANN_MLP> ann = ANN_MLP::create();
	ann->setLayerSizes(layerSizes);
	ann->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1, 1);
	ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, FLT_EPSILON));
	ann->setTrainMethod(ANN_MLP::BACKPROP, 0.1, 0.1);

	// 4、构建卡尔曼滤波，并初始化
	RNG rng;
	const float T = 0.9;        //采样周期
	float v_x = 1.0, v_y = 1.0, a_x = 0.1, a_y = 0.1;
	//const int stateNum = 4;	//状态数，包括（x，y，dx，dy）坐标及速度（每次移动的距离）
	const int stateNum = 6;		//状态数，包括（x, y, dx, dy, d(dx), d(dy)）坐标、速度（每次移动的距离）、加速度
	const int measureNum = 2;       //测量值2×1向量(x,y)
	KalmanFilter KF(stateNum, measureNum, 0);

	KF.transitionMatrix = (Mat_<float>(6, 6) << 1, 0, T, 0, (T*T) / 2, 0,
						    0, 1, 0, T, 0, (T*T) / 2,
						    0, 0, 1, 0, T, 0,
						    0, 0, 0, 1, 0, T,
						    0, 0, 0, 0, 1, 0,
						    0, 0, 0, 0, 0, 1);

	setIdentity(KF.measurementMatrix);			        //测量矩阵H    
	setIdentity(KF.processNoiseCov, Scalar::all(1e-5));		//系统噪声方差矩阵Q    
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));	        //测量噪声方差矩阵R    
	setIdentity(KF.errorCovPost, Scalar::all(1));			//P(k)
	Mat measurement = Mat::zeros(measureNum, 1, CV_32F);
	KF.statePost = (Mat_<float>(6, 1) << 0, 0, v_x, v_y, a_x, a_y);

	// 5、神经网络训练及测试
	Mat trainMat, resMat, sampleMat, responseMat;
	Point point1, point2, point3, point4;
	Point predicte, state;

	for (int i = 0; i < row1 - 3; i++) {
		cout << "第" << i + 1 << "次训练预测" << endl;
		trainMat = trainingDataMat.rowRange(i, i + 3);
		//cout << "训练差值：" << trainMat << endl;
		resMat = trainingDataMat_res.rowRange(i, i + 3);
		Ptr<TrainData> tData = TrainData::create(trainMat, ROW_SAMPLE, resMat);
		ann->train(tData);

		// 测试样本的差值（一组(两对)差值——对应三对真实点坐标）
		sampleMat = trainingDataMat.rowRange(i + 3, i + 4);
		//cout << "神经网络测试样本：" << sampleMat << endl;

		// 计算测试样本对应的真实坐标(三对)
		point1.x = storage1[2 * (i + 3)];
		point1.y = storage1[2 * (i + 3) + 1];
		point2.x = storage1[2 * (i + 4)];
		point2.y = storage1[2 * (i + 4) + 1];
		point3.x = storage1[2 * (i + 5)];
		point3.y = storage1[2 * (i + 5) + 1];
		//cout << "point1 = " << point1 << "   " << "point2 = " << point2 << "   " << "point3 = " << point3 << endl << endl;

		ann->predict(sampleMat, responseMat);
		//cout << "神经网络预测数值：" << responseMat << endl;
		point4.x = point3.x + responseMat.at<float>(0) + 0.5;    //四舍五入
		point4.y = point3.y + responseMat.at<float>(1) + 0.5;
		//cout << "神经网络预测值point4 = " << point4 << endl << endl;
		BP.push_back( point4 );

		// 6、kalman预测
		KF.statePost.at<float>(0) = point3.x;
		KF.statePost.at<float>(1) = point3.y;

		//预测  
		Mat prediction = KF.predict();
		predicte.x = prediction.at<float>(0) + 0.5;
		predicte.y = prediction.at<float>(1) + 0.5;
		//cout << "状态方程计算值point4'= " << predicte << endl;
		Kalman.push_back( predicte );
		//计算测量值  
		measurement.at<float>(0) = (float)point4.x;
		measurement.at<float>(1) = (float)point4.y;
		//更新  
		KF.correct(measurement);
		state.x = KF.statePost.at<float>(0) + 0.5;
		state.y = KF.statePost.at<float>(1) + 0.5;

		//输出结果  
		//cout << "最优估计值correct =" << state << endl << endl;//下一时刻预测到的值
		BP_Kalman.push_back( state );

	}

	// 6、将神经网络预测值、卡尔曼状态方程计算值、融合值（最优估计值）保存到本地文本
	ofstream bpfile("BP_Point.txt");
	for ( vector<Point>::iterator it = BP.begin( ); it != BP.end( ); it++ ) { 
		bpfile << it->x << "," << it->y << "," << endl;
	}
	bpfile.close( );

	ofstream kalmanfile( "Kalman_Point.txt" );
	for ( vector<Point>::iterator it = Kalman.begin( ); it != Kalman.end( ); it++ ) {
		kalmanfile << it->x << "," << it->y << "," << endl;
	}
	kalmanfile.close( );

	ofstream bpkalmanfile( "BpKalman_Point.txt" );
	for ( vector<Point>::iterator it = BP_Kalman.begin( ); it != BP_Kalman.end( ); it++ ) {
		bpkalmanfile << it->x << "," << it->y << "," << endl;
	}
	bpkalmanfile.close( );

	return 0;
}
