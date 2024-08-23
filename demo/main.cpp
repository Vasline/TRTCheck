#include <fstream>
#include <limits>
#include <iostream>
#include <map>
#include <vector>
#include <cmath>
#include "tensorRTClass.h"
using namespace DLH;

// 计算两个向量之间的平均绝对误差
double meanAbsoluteError(const std::vector<float>& a, const std::vector<float>& b) {
	double sum = 0.0;
	for (size_t i = 0; i < a.size(); ++i) {
		sum += std::abs(a[i] - b[i]);
	}
	return sum / a.size();
}

// 计算两个向量之间的均方误差
double meanSquaredError(const std::vector<float>& a, const std::vector<float>& b) {
	double sum = 0.0;
	for (size_t i = 0; i < a.size(); ++i) {
		sum += std::pow(a[i] - b[i], 2);
	}
	return sum / a.size();
}

// 计算两个结果之间的差异程度
void compareResults(const std::vector<std::vector<float>>& fp32_result,
	const std::vector<std::vector<float>>& fp16_result) {
	double totalMAE = 0.0;
	double totalMSE = 0.0;

	// 遍历每个输出张量的结果
	for (size_t i = 0; i < fp32_result.size(); ++i) {
		double mae = meanAbsoluteError(fp32_result[i], fp16_result[i]);
		double mse = meanSquaredError(fp32_result[i], fp16_result[i]);

		totalMAE += mae;
		totalMSE += mse;

		std::cout << "Output " << i << " MAE: " << mae << ", MSE: " << mse << std::endl;
	}

	// 计算平均 MAE 和 MSE
	double averageMAE = totalMAE / fp32_result.size();
	double averageMSE = totalMSE / fp32_result.size();

	std::cout << "Average MAE: " << averageMAE << ", Average MSE: " << averageMSE << std::endl;
}


// 将一维向量重塑为二维向量
std::vector<std::vector<float>> reshapeVector(const std::vector<float>& vec, int height, int width) {
	std::vector<std::vector<float>> reshapedVec(height, std::vector<float>(width, 0.0f));
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			reshapedVec[i][j] = vec[i * width + j];
		}
	}
	return reshapedVec;
}

// 重塑整个结果集
std::vector<std::vector<std::vector<float>>> reshapeResults(const std::vector<std::vector<float>>& results) {
	std::vector<std::vector<std::vector<float>>> reshapedResults;
	for (const auto& result : results) {
		// 假设每个结果的大小为 100 * 1024
		int height = 100;
		int width = 1024;
		if (result.size() == height * width) {
			std::vector<std::vector<float>> reshapedResult = reshapeVector(result, height, width);
			reshapedResults.push_back(reshapedResult);
		}
		else {
			std::cerr << "Error: Invalid size for reshaping." << std::endl;
		}
	}
	return reshapedResults;
}



std::vector<std::vector<float>> getTrtResults(const std::string& onnx_path, const std::string& trt_save_path, bool fp16_flag) {
	//const std::string onnx_path = "H:/wav2lip_demo/code/master/fugan/data/engine/audio_hbt_large_02s.onnx";
	std::vector<std::string> set_output_vec;
	set_output_vec.push_back("/feature_extractor/conv_layers.0/conv/Conv");
	set_output_vec.push_back("/feature_extractor/conv_layers.1/conv/Conv");
	set_output_vec.push_back("/feature_extractor/conv_layers.2/conv/Conv");
	set_output_vec.push_back("/feature_extractor/conv_layers.3/conv/Conv");
	set_output_vec.push_back("/feature_extractor/conv_layers.4/conv/Conv");
	set_output_vec.push_back("/feature_extractor/conv_layers.5/conv/Conv");
	set_output_vec.push_back("/feature_extractor/conv_layers.6/conv/Conv");
	std::shared_ptr<trtModel> trt_model = std::make_shared<trtModel>(onnx_path, set_output_vec, fp16_flag);
	trt_model->saveEngine(trt_save_path);

	std::map<int, std::vector<int>> input_shapes = trt_model->getInputShape();
	std::map<int, std::vector<int>> output_shapes = trt_model->getOutputShape();
	std::vector<std::vector<float>> input_data;
	for (auto iter = input_shapes.begin();iter != input_shapes.end();++iter) {
		int sub_shape = 1;
		for (int j = 0;j < iter->second.size();++j) {
			sub_shape *= iter->second[j];
			std::cout << j << " --- " << iter->second[j] << std::endl;
		}
		std::cout << "\n" << iter->first << " <-------------------------------> " << sub_shape << std::endl;

		std::vector<float> tmp_data(sub_shape, 0.0f);
		for (int k = 0;k < sub_shape;++k) {
			tmp_data[k] = 0.5;
		}
		input_data.push_back(tmp_data);
	}

	//for (int i = 0;i < output_shapes.size();++i) {
	//	for (int j = 0;j < output_shapes[i].size();++j) {
	//		std::cout << j << " --- " << output_shapes[i][j] << std::endl;
	//	}
	//	std::cout << "\n" << i << " <------------------------------->" << std::endl;
	//}
	std::vector<std::vector<float>> output_data;
	bool status = trt_model->forward(input_data, output_data);
	if (!status) {
		std::vector<std::vector<float>> tmp_output_data;
		std::cerr << "trt_model forward failed !" << std::endl;
		return tmp_output_data;
	}
	return output_data;

}


int main(int argc, char** argv) {
	const std::string onnx_path = argv[1];
	const std::string onnx_fp32_path = argv[2];
	const std::string onnx_fp16_path = argv[3];
	std::vector<std::vector<float>> fp32_result = getTrtResults(onnx_path, onnx_fp32_path, false);
	std::vector<std::vector<float>> fp16_result = getTrtResults(onnx_path, onnx_fp16_path, true);
	compareResults(fp32_result, fp16_result);
	return 0;
}
