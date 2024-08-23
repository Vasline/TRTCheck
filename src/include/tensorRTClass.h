#pragma once
#include <fstream>
#include <limits>
#include <iostream>
#include <map>
#include <vector>
#include <NvOnnxParser.h>
#include <NvInfer.h>
#include <../samples/common/logger.h>
using namespace nvinfer1;
using namespace sample;

namespace DLH {

	class trtModel {
	public:
		trtModel(const std::string& onnx_path, const std::vector<std::string>& set_output_name, bool fp16_flag = false);
		~trtModel();
		bool forward(const std::vector<std::vector<float>>& input_data, std::vector<std::vector<float>>& output_results);
		std::map<int, std::vector<int>> getOutputShape();
		std::map<int, std::vector<int>> getInputShape();
		int32_t getTypeSize(DataType flag);
		bool saveEngine(const std::string& trt_save_path);
		int getNbOutput();
		int getNbInput();

	private:
		void initInputsShape();
		void initOutputsShape();
		static const int BATCH_SIZE = 1;
		bool _fp16_flag = false;

		// model infomation
		INetworkDefinition* _network = nullptr;
		IBuilder* _builder = nullptr;
		ICudaEngine* _engine = nullptr;
		IExecutionContext* _context = nullptr;
		IBuilderConfig* _config = nullptr;

		// node information
		int _layer_number = 0;
		int _output_number = 0;
		int _input_number = 0;
		int _binding_number = 0;
		int _out_start_index = 10000000;
		std::vector<int> _mark_output_index;

		// input and output information
		std::map<int, int> _inputs_size_map;
		std::map<int, int> _outputs_size_map;
		std::map<int, DataType> _inputs_type_map;
		std::map<int, DataType> _outputs_type_map;
		std::map<int, std::vector<int>> _inputs_shape_map;
		std::map<int, std::vector<int>> _outputs_shape_map;
		std::vector<int> _outputs_index;
		std::vector<int> _intputs_index;

	};

}