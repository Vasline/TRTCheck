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
		//init(onnx_path, set_output_name, fp16_flag);
		initNonFp16(onnx_path, set_output_name, fp16_flag);
	}

	void trtModel::init(const std::string& onnx_path, const std::vector<std::string>& set_output_name, bool fp16_flag) {
		if (_is_init) {
			return;
		}
		const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
		_builder = createInferBuilder(logger);
		_network = _builder->createNetworkV2(explicitBatch);

		// Parse ONNX file
		nvonnxparser::IParser* parser = nvonnxparser::createParser(*_network, logger);
		bool parser_status = parser->parseFromFile(onnx_path.c_str(), static_cast<int>(ILogger::Severity::kWARNING));
		_config = _builder->createBuilderConfig();
		if (fp16_flag) {
			_fp16_flag = true;
			//_config->setFlag(BuilderFlag::kFP16);
		}
		_layer_number = _network->getNbLayers();
		int origin_output_number = _network->getNbOutputs();
		if (!set_output_name.empty()) {
			int not_exist_num = 0;
			int exist_num = 0;
			for (int out_i = 0;out_i < set_output_name.size();++out_i) {
				bool exist_flag = false;
				for (int i = 0;i < _layer_number;++i) {
					ILayer* layer = _network->getLayer(i);
					if (strcmp(layer->getName(), set_output_name[out_i].c_str()) == 0) {
						std::cout << "layer type -------------------> " << static_cast<int>(layer->getPrecision()) << std::endl;
						/*if (layer->getType() == LayerType::kCONVOLUTION) {
							IConvolutionLayer* convLayer = static_cast<IConvolutionLayer*>(layer);
							Weights conv_weights = convLayer->getKernelWeights();
							float* conv_values = static_cast<float*>(const_cast<void*>(conv_weights.values));
							for (int64_t k = 0;k < conv_weights.count;++k) {
								if (std::fabs(reinterpret_cast<const float*>(conv_weights.values)[k]) > 65504.f) {
									std::cout << "layer name : " << convLayer->getName() << " ; k : " << k << " value : " << reinterpret_cast<const float*>(conv_weights.values)[k] << std::endl;
									layer->setPrecision(DataType::kFLOAT);
									break;
									conv_values[k] = 65503.f;
								}
								if (std::fabs(reinterpret_cast<const float*>(conv_weights.values)[k]) < 0.00006103) {
									std::cout << "layer name : " << convLayer->getName() << " ; k : " << k << " value : " << reinterpret_cast<const float*>(conv_weights.values)[k] << std::endl;
									layer->setPrecision(DataType::kFLOAT);
									break;
									if (conv_values[k] > 0.0) {
										conv_values[k] = FP16_MIN_POS_NORMALIZED;
									}
									else {
										conv_values[k] = FP16_MIN_NEG_NORMALIZED;
									}

								}
							}
						}*/

						std::cout << "Mark " << layer->getName() << " as a network output and set outputType is kFLOAT. Output number is : " << layer->getNbOutputs() << std::endl;

						for (int out_i = 0;out_i < layer->getNbOutputs();++out_i) {
							//layer->setOutputType(out_i, DataType::kFLOAT);
							if (_fp16_flag) {
								layer->setOutputType(out_i, DataType::kHALF);
							}
							_network->markOutput(*layer->getOutput(0));
							_mark_output_index.push_back(origin_output_number + out_i);
						}
						exist_flag = true;
						exist_num++;
						break;
					}
				}
				if (!exist_flag) {
					std::cerr << "The " << set_output_name[out_i] << " not in the network" << std::endl;
					not_exist_num++;
				}
			}
			std::cout << "The " << exist_num << " layers in network and " << not_exist_num << " layers not in network" << std::endl;
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
		std::cout << "_engine layer number : " << _engine->getNbLayers() << " ; network layer number : " << _network->getNbLayers() << std::endl;
		_context = _engine->createExecutionContext();
		parser->destroy();
		_output_number = _network->getNbOutputs();
		_input_number = _network->getNbInputs();
		_binding_number = _engine->getNbBindings();
		initInputsShape();
		initOutputsShape();
		std::cout << "init sucessfully" << std::endl;
		_is_init = true;
	}

	void trtModel::initNonFp16(const std::string& onnx_path, const std::vector<std::string>& set_output_name, bool fp16_flag) {
		if (_is_init) {
			return;
		}
		const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
		_builder = createInferBuilder(logger);
		_network = _builder->createNetworkV2(explicitBatch);

		// Parse ONNX file
		nvonnxparser::IParser* parser = nvonnxparser::createParser(*_network, logger);
		bool parser_status = parser->parseFromFile(onnx_path.c_str(), static_cast<int>(ILogger::Severity::kWARNING));
		_config = _builder->createBuilderConfig();
		if (fp16_flag) {
			_fp16_flag = true;
			//_config->setFlag(BuilderFlag::kFP16);
		}
		_layer_number = _network->getNbLayers();
		int origin_output_number = _network->getNbOutputs();
		if (!set_output_name.empty()) {
			int not_exist_num = 0;
			int exist_num = 0;
			for (int i = 0;i < _layer_number;++i) {
				bool non_exist_flag = true;
				ILayer* layer = _network->getLayer(i);
				for (int out_i = 0;out_i < set_output_name.size();++out_i) {
					if (strcmp(layer->getName(), set_output_name[out_i].c_str()) == 0) {
						non_exist_flag = false;
						//if ((layer->getType() == LayerType::kCONVOLUTION) && _fp16_flag) {
						//	layer->setPrecision(DataType::kFLOAT);
						//}
						for (int out_idx = 0;out_idx < layer->getNbOutputs();++out_idx) {
							_network->markOutput(*layer->getOutput(0));
							_mark_output_index.push_back(origin_output_number + out_idx);
						}
						break;
					}
				}
				if (non_exist_flag && _fp16_flag) {
					// 跳过一些固定的无法设置为fp16的层
					if (layer->getType() == nvinfer1::LayerType::kSHAPE || layer->getType() == nvinfer1::LayerType::kIDENTITY ||
						layer->getType() == nvinfer1::LayerType::kSHUFFLE || layer->getType() == nvinfer1::LayerType::kSLICE || layer->getType() == nvinfer1::LayerType::kCONCATENATION) {
						continue;
					}
					if (layer->getPrecision() == nvinfer1::DataType::kINT32) {
						continue;
					}
					if (strcmp(layer->getName(), "Tile") == 0) {
						continue;
					}
					layer->setPrecision(DataType::kHALF);
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
		std::cout << "_engine layer number : " << _engine->getNbLayers() << " ; network layer number : " << _network->getNbLayers() << std::endl;

		_context = _engine->createExecutionContext();
		parser->destroy();
		_output_number = _network->getNbOutputs();
		_input_number = _network->getNbInputs();
		_binding_number = _engine->getNbBindings();
		initInputsShape();
		initOutputsShape();
		std::cout << "init sucessfully" << std::endl;
		_is_init = true;
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

	bool trtModel::forward(const std::vector<std::vector<float>>& input_data) {
		if (!_is_init) {
			std::cerr << "Please init model first ......" << std::endl;
			return false;
		}
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

		for (auto iter = _outputs_size_map.begin();iter != _outputs_size_map.end();++iter) {
			std::vector<float> tmp_output(iter->second, 0.0f);
			_outputs_index_result_map.insert(std::pair<int, std::vector<float>>(iter->first, tmp_output));
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

		// inference
		bool status = _context->enqueueV2(deviceBuffers, stream, nullptr);
		if (!status) {
			std::cerr << "model inference failed !" << std::endl;
			return status;
		}

		// obtain every output result
		cudaStreamSynchronize(stream);
		for (auto iter = _outputs_size_map.begin(); iter != _outputs_size_map.end(); ++iter) {
			switch (_outputs_type_map[iter->first])
			{
			case DataType::kFLOAT:
				CHECK(cudaMemcpyAsync(_outputs_index_result_map[iter->first].data(), deviceBuffers[iter->first], BATCH_SIZE * _outputs_size_map[iter->first] * sizeof(float), cudaMemcpyDeviceToHost, stream));
				break;
			case DataType::kHALF:
			{
				std::vector<half> tmp_output_fp16(iter->second);
				CHECK(cudaMemcpyAsync(tmp_output_fp16.data(), deviceBuffers[iter->first], BATCH_SIZE * _outputs_size_map[iter->first] * sizeof(half), cudaMemcpyDeviceToHost, stream));
				// convert FP16 data FP32
				for (size_t j = 0; j < tmp_output_fp16.size(); ++j) {
					//output_results[iter->first - _out_start_index][j] = static_cast<float>(tmp_output_fp16[j]);
					_outputs_index_result_map[iter->first][j] = static_cast<float>(tmp_output_fp16[j]);
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

	bool trtModel::getOutputResults(const std::vector<std::string>& outputs_name, std::map<std::string, std::vector<float>>& output_results) {
		if (_outputs_index_result_map.empty()) {
			std::cerr << "output results is empty !" << std::endl;
			return false;
		}
		for (std::string out_name : outputs_name) {
			auto iter = _output_name_index_map.find(out_name);
			if (iter != _output_name_index_map.end()) {
				output_results.insert(std::pair<std::string, std::vector<float>>(out_name, _outputs_index_result_map[iter->second]));
				_outputs_index_result_map.erase(iter->second);
			}
			else {
				std::cerr << out_name << " not in output list" << std::endl;
			}
		}
		return !output_results.empty();
	}

	std::vector<std::string> trtModel::getOutputsName() {
		return _outputs_name_vec;
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
		_output_name_index_map.clear();
		_outputs_name_vec.clear();
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
			_output_name_index_map.insert(std::pair<std::string, int>(layer_name, layer_index));
			_outputs_name_vec.push_back(layer_name);
		}
	}
}