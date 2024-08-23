#include <cuda_fp16.h>
#include "tensorRTClass.h"
Logger logger;
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)
namespace DLH {
	trtModel::trtModel(const std::string& onnx_path, const std::vector<std::string>& set_output_name, bool fp16_flag) {
		const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
		_builder = createInferBuilder(logger);
		_network = _builder->createNetworkV2(explicitBatch);
		// Parse ONNX file
		nvonnxparser::IParser* parser = nvonnxparser::createParser(*_network, logger);
		bool parser_status = parser->parseFromFile(onnx_path.c_str(), static_cast<int>(ILogger::Severity::kWARNING));
		_config = _builder->createBuilderConfig();
		if (fp16_flag) {
			_fp16_flag = true;
			_config->setFlag(BuilderFlag::kFP16);
		}
		_layer_number = _network->getNbLayers();
		int origin_output_number = _network->getNbOutputs();
		if (!set_output_name.empty()) {
			for (int out_i = 0;out_i < set_output_name.size();++out_i) {
				bool exist_flag = false;
				for (int i = 0;i < _layer_number;++i) {
					ILayer* layer = _network->getLayer(i);
					if (strcmp(layer->getName(), set_output_name[out_i].c_str()) == 0) {
						std::cout << "Mark " << layer->getName() << " as a network output. and the layer output number is : " << layer->getNbOutputs() << std::endl;
						for (int out_i = 0;out_i < layer->getNbOutputs();++out_i) {
							layer->setOutputType(out_i, DataType::kFLOAT);
						}
						_network->markOutput(*layer->getOutput(0));
						_mark_output_index.push_back(origin_output_number + out_i);
						exist_flag = true;
						break;
					}
				}
				if (!exist_flag) {
					std::cerr << "The " << set_output_name[out_i] << " not in the network" << std::endl;
				}
			}
		}

		// Get the name of network input
		Dims dim = _network->getInput(0)->getDimensions();
		for (int i = 0;i < dim.nbDims;++i) {
			std::cout << "dim " << i << " : " << dim.d[i] << " ; ";
		}
		std::cout << std::endl;
		if (dim.d[1] == -1)  // -1 means it is a dynamic model
		{
			const char* name = _network->getInput(0)->getName();
			std::cout << "input is dynamic shape and modify " << name << std::endl;
			IOptimizationProfile* profile = _builder->createOptimizationProfile();
			profile->setDimensions(name, OptProfileSelector::kMIN, Dims2(1, 32080));
			profile->setDimensions(name, OptProfileSelector::kOPT, Dims2(1, 32080));
			profile->setDimensions(name, OptProfileSelector::kMAX, Dims2(1, 32080));
			_config->addOptimizationProfile(profile);
		}
		_config->setMaxWorkspaceSize(1 << 30);
		std::vector<std::vector<float>> input_output;
		_engine = _builder->buildEngineWithConfig(*_network, *_config);
		_context = _engine->createExecutionContext();
		parser->destroy();
		//config->destroy();
		_output_number = _network->getNbOutputs();
		_input_number = _network->getNbInputs();
		_binding_number = _engine->getNbBindings();
		initInputsShape();
		initOutputsShape();
		std::cout << "init sucessfully" << std::endl;

	}
	trtModel::~trtModel() {
		if (_context != nullptr) {
			_context->destroy();
		}
		if (_engine != nullptr) {
			_engine->destroy();
		}
		if (_config != nullptr) {
			_config->destroy();
		}
		if (_network != nullptr) {
			_network->destroy();
		}
		if (_builder != nullptr) {
			_builder->destroy();
		}
	}
	bool trtModel::forward(const std::vector<std::vector<float>>& input_data, std::vector<std::vector<float>>& output_results) {

		for (int i = 0;i < _engine->getNbBindings();++i) {
			auto layer_name = _engine->getBindingName(i);
			int64_t layer_index = _engine->getBindingIndex(layer_name);
			auto dims = _engine->getBindingDimensions(layer_index);
			int tmp_dims_size = 1;
			for (int i = 0;i < dims.nbDims;++i) {
				std::cout << i << " : " << dims.d[i] << std::endl;
				tmp_dims_size *= dims.d[i];
			}
			std::cout << i << " ------------> " << layer_name << " ; layer index : " << layer_index << " ; size : " << tmp_dims_size << std::endl;
		}

		output_results.resize(_outputs_size_map.size());
		for (auto iter = _outputs_size_map.begin();iter != _outputs_size_map.end();++iter) {
			std::vector<float> tmp_output(iter->second, 0.0f);
			output_results[iter->first - _out_start_index] = tmp_output;
		}

		// Create stream
		cudaStream_t stream;
		CHECK(cudaStreamCreate(&stream));
		void** deviceBuffers = new void* [_binding_number];
		std::cout << "_binding_number : " << _binding_number << " ; input size + output size : " << _inputs_size_map.size() + _outputs_size_map.size() << std::endl;
		for (int i = 0;i < _inputs_size_map.size();++i) {
			CHECK(cudaMalloc(&deviceBuffers[i], BATCH_SIZE * _inputs_size_map[i] * this->getTypeSize(_inputs_type_map[i])));
			CHECK(cudaMemcpyAsync(deviceBuffers[i], input_data[i].data(), BATCH_SIZE * _inputs_size_map[i] * this->getTypeSize(_inputs_type_map[i]), cudaMemcpyHostToDevice, stream));
		}

		for (auto iter = _outputs_size_map.begin();iter != _outputs_size_map.end();++iter) {
			CHECK(cudaMalloc(&deviceBuffers[iter->first], BATCH_SIZE * iter->second * this->getTypeSize(_outputs_type_map[iter->first])));
		}

		// 执行推理
		bool status = _context->enqueueV2(deviceBuffers, stream, nullptr);
		//bool status = _context->executeV2(deviceBuffers);
		if (!status) {
			std::cerr << "model inference failed !" << std::endl;
			return status;
		}

		// 获取并打印每一层的输出
		cudaStreamSynchronize(stream);
		for (auto iter = _outputs_size_map.begin(); iter != _outputs_size_map.end(); ++iter) {
			switch (_outputs_type_map[iter->first])
			{
			case DataType::kFLOAT:
				CHECK(cudaMemcpyAsync(output_results[iter->first - _out_start_index].data(), deviceBuffers[iter->first], BATCH_SIZE * _outputs_size_map[iter->first] * sizeof(float), cudaMemcpyDeviceToHost, stream));
				break;
			case DataType::kHALF:
			{
				std::vector<half> tmp_output_fp16(iter->second);
				CHECK(cudaMemcpyAsync(tmp_output_fp16.data(), deviceBuffers[iter->first], BATCH_SIZE * _outputs_size_map[iter->first] * sizeof(half), cudaMemcpyDeviceToHost, stream));
				// convert FP16 data FP32
				for (size_t j = 0; j < tmp_output_fp16.size(); ++j) {
					output_results[iter->first - _out_start_index][j] = static_cast<float>(tmp_output_fp16[j]);
				}
				break;
			}
			default:
				std::cerr << "Only support kFLOAT and kHALF type" << std::endl;
				status = false;
				break;
			}
		}

		delete[] deviceBuffers;
		cudaStreamDestroy(stream);
		return status;
	}

	int32_t trtModel::getTypeSize(DataType flag) {
		int32_t size = 1;
		switch (flag)
		{
		case DataType::kFLOAT:
			size = sizeof(float);
			break;
		case DataType::kHALF:
			size = sizeof(half);
			break;
		case DataType::kINT8:
			size = sizeof(int8_t);
			break;
		case DataType::kINT32:
			size = sizeof(int);
			break;
		case DataType::kBOOL:
			size = sizeof(bool);
			break;
		case DataType::kUINT8:
			size = sizeof(uint8_t);
			break;
		default:
			break;
		}
		return size;
	}

	bool trtModel::saveEngine(const std::string& trt_save_path) {
		if (_engine == nullptr) {
			std::cerr << "engine is empty !" << std::endl;
			return false;
		}
		std::ofstream out(trt_save_path, std::ios::binary);
		if (!out.is_open()) {
			std::cerr << "open " << trt_save_path << " failed !" << std::endl;
			return false;
		}
		IHostMemory* trtModelStream = _engine->serialize(); //序列化 保存trt
		out.write(reinterpret_cast<const char*>(trtModelStream->data()), trtModelStream->size());
		out.close();
		return true;
	}

	std::map<int, std::vector<int>> trtModel::getInputShape() {
		return _inputs_shape_map;
	}
	std::map<int, std::vector<int>> trtModel::getOutputShape() {
		return _outputs_shape_map;
	}
	int trtModel::getNbOutput() {
		return _output_number;
	}
	int trtModel::getNbInput() {
		return _input_number;
	}

	void trtModel::initInputsShape() {
		_intputs_index.clear();
		_inputs_size_map.clear();
		_inputs_shape_map.clear();
		_inputs_type_map.clear();
		for (int i = 0;i < _network->getNbInputs();++i) {
			std::vector<int> tmp_shape;
			int tmp_size = 1;
			auto input = _network->getInput(i);
			auto layer_name = input->getName();
			int64_t layer_index = _engine->getBindingIndex(layer_name);
			auto dims = input->getDimensions();
			for (int j = 0;j < dims.nbDims;++j) {
				tmp_shape.push_back(dims.d[j]);
				tmp_size *= dims.d[j];
			}
			_intputs_index.push_back(layer_index);
			_inputs_type_map.insert(std::pair<int, DataType>(layer_index, input->getType()));
			_inputs_shape_map.insert(std::pair<int, std::vector<int>>(layer_index, tmp_shape));
			_inputs_size_map.insert(std::pair<int, int>(layer_index, tmp_size));
		}
	}

	void trtModel::initOutputsShape() {
		_outputs_index.clear();
		_outputs_size_map.clear();
		_outputs_shape_map.clear();
		_outputs_type_map.clear();
		for (int i = 0;i < _network->getNbOutputs();++i) {
			std::vector<int> tmp_shape;
			int tmp_size = 1;
			auto output = _network->getOutput(i);
			auto layer_name = output->getName();
			int64_t layer_index = _engine->getBindingIndex(layer_name);
			if (_out_start_index >= layer_index) {
				_out_start_index = layer_index;
			}
			auto dims = output->getDimensions();
			for (int j = 0;j < dims.nbDims;++j) {
				tmp_shape.push_back(dims.d[j]);
				tmp_size *= dims.d[j];
			}
			_outputs_index.push_back(layer_index);
			_outputs_type_map.insert(std::pair<int, DataType>(layer_index, output->getType()));
			_outputs_shape_map.insert(std::pair<int, std::vector<int>>(layer_index, tmp_shape));
			_outputs_size_map.insert(std::pair<int, int>(layer_index, tmp_size));
		}
	}
}