#include "TRTParser.h"

Box::Box(unsigned x1, unsigned y1, unsigned x2, unsigned y2, unsigned classID, float score){
	this->x1 = x1;
	this->y1 = y1;
	this->x2 = x2;
	this->y2 = y2;
	this->classID = classID;
	this->score = score;
}

TRTParser::TRTParser() {
	this->engineSize = 0;
	this->maxBatchSize = 0;
	this->context = nullptr;
	this->engine = nullptr;
}

TRTParser::~TRTParser() {
	this->inputTensors.clear();
	this->outputTensors.clear();
	this->context->destroy();
	this->engine->destroy();
}

size_t TRTParser::GetDimensionSize(const nvinfer1::Dims& dims) {	
	size_t size = 1;
	for (unsigned i = 0; i < dims.nbDims; i++) {
		size *= dims.d[i];
	}
	return size;
}

nvinfer1::ICudaEngine* TRTParser::LoadTRTEngine(const string enginePath) {
	ifstream gieModelStream(enginePath, ios::binary);
	if (!gieModelStream.good()) {
		cerr << "[ERROR] Could not read engine! \n";
		gieModelStream.close();
		return nullptr;
	}
	gieModelStream.seekg(0, ios::end);
	size_t modelSize = gieModelStream.tellg();
	gieModelStream.seekg(0, ios::beg);

	void* modelData = malloc(modelSize);
	if(!modelData) {
		cerr << "[ERROR] Could not allocate memory for onnx engine! \n";
		gieModelStream.close();
		return nullptr;
	}
	gieModelStream.read((char*)modelData, modelSize);
	gieModelStream.close();

	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
	if (runtime == nullptr) {
		cerr << "[ERROR] Could not create InferRuntime! \n";
		return nullptr;
	}
	return runtime->deserializeCudaEngine(modelData, modelSize, nullptr);
}


bool TRTParser::Init(const string enginePath) {
	this->engine = this->LoadTRTEngine(enginePath);
	if (this->engine == nullptr) {
		return false;
	}
	else{
		this->maxBatchSize = this->engine->getMaxBatchSize();
		this->context = this->engine->createExecutionContext();
		this->engineSize = this->engine->getDeviceMemorySize();
		size_t totalDevMem, freeDevMem;
		if (!CudaCheck(cudaMemGetInfo(&freeDevMem, &totalDevMem))) return false;
		if (this->engineSize > freeDevMem) {
			cerr << "[ERROR] Not enough Gpu Memory! Model's WorkspaceSize: " << this->engineSize/1048576 << "MB. Free memory left: " << freeDevMem/1048576 <<"MB. \nReduce workspacesize to continue.\n";
			return false;
		}
		for (unsigned i = 0; i < this->engine->getNbBindings(); i++) {
			Tensor tensor(this->engine, i);
			if (this->engine->bindingIsInput(i)) {
				this->inputTensors.emplace_back(tensor);
			}
			else {
				this->outputTensors.emplace_back(tensor);
			}	
		}
		if (this->inputTensors.empty() || this->outputTensors.empty()) {
			cerr << "[ERROR] Expect at least one input and one output for network \n";
			return false;
		}
		if (!ShowEngineInfo(engine)){
			return false;
		}
		return true;
	}
}


bool TRTParser::AllocateImageInput(vector<cv::Mat> images, float* gpuInputBuffer, const unsigned inputIndex) {
	if (inputIndex >= this->inputTensors.size()) {
		cerr << "[ERROR] inputIndex is greater than number of inputTensor's index!\n";
		return false;
	}
	unsigned imgH, imgW, imgC;
	if (!inputTensors.at(inputIndex).isCHW){
		imgH = this->inputTensors.at(inputIndex).dims.d[1];
		imgW = this->inputTensors.at(inputIndex).dims.d[2];
		imgC = this->inputTensors.at(inputIndex).dims.d[3];
	}
	else { 
		imgH = this->inputTensors.at(inputIndex).dims.d[2];
		imgW = this->inputTensors.at(inputIndex).dims.d[3];
		imgC = this->inputTensors.at(inputIndex).dims.d[1];
	}
	auto imageSize = cv::Size(imgW, imgH);
	for (unsigned i = 0; i < images.size(); i++) {
		//Upload images to GPU
		cv::Mat image = images.at(i);
		if (image.empty()) {
			cerr << "[ERROR] Could not load Input image!! \n";
			return false;
		}
		cv::cuda::GpuMat gpuImage;
		gpuImage.upload(image);
		//Resize
		cv::cuda::GpuMat gpuResized, gpuImageFloat;
		cv::cuda::resize(gpuImage, gpuResized, imageSize);
		//Normalize
		gpuResized.convertTo(gpuImageFloat, CV_32FC3, 1.f / 255.f);
		// cv::cuda::subtract(gpuImageFloat, cv::Scalar(0.485f, 0.456f, 0.406f), gpuImageFloat, cv::noArray(), -1);
		// cv::cuda::divide(gpuImageFloat, cv::Scalar(0.229f, 0.224f, 0.225f), gpuImageFloat, 1, -1);
		//Allocate
		if (imgC == 3){
			if (this->inputTensors.at(inputIndex).isCHW){
				try {
					cv::Mat test(gpuImageFloat);
					vector< cv::cuda::GpuMat > chw;
					for (unsigned j = 0; j < imgC; j++) {
						chw.emplace_back(cv::cuda::GpuMat(imageSize, CV_32FC1, gpuInputBuffer + (i*imgC+j)*imgW*imgH));
					}
					cv::cuda::split(gpuImageFloat, chw);
				}
				catch (cv::Exception& e) {
    				cout << "[ERROR] [OpenCV] Exception caught: " << e.what();
    				return false;
				}
			}
			else {
				size_t inputBufferSize = this->GetDimensionSize(this->inputTensors.at(inputIndex).dims);
				if (!CudaCheck(cudaMemcpyAsync(gpuInputBuffer, gpuImageFloat.ptr<float>(), inputBufferSize*sizeof(float), cudaMemcpyDeviceToDevice))) return false;
			}
		}
		else if (imgC == 1) {
			if (!CudaCheck(cudaMemcpyAsync(gpuInputBuffer, gpuImageFloat.ptr<float>(), gpuImageFloat.rows*gpuImageFloat.step, cudaMemcpyDeviceToDevice))) return false;
		}
		else {
			cerr << "[ERROR] Undefined image channel!\n";
			return false;
		}
	}
	return true;
}

bool TRTParser::AllocateNonImageInput(void *pData, float* gpuInputBuffer, const unsigned inputIndex){
	if (inputIndex >= this->inputTensors.size()){
		cerr << "[ERROR] inputIndex is greater than number of inputTensor's index!\n";
		return false;
	}
	size_t inputBufferSize = this->GetDimensionSize(this->inputTensors.at(inputIndex).dims);
	if (!CudaCheck(cudaMemcpyAsync(gpuInputBuffer, pData, inputBufferSize * this->inputTensors.at(inputIndex).tensorSize, cudaMemcpyHostToDevice))) return false;
	return true;
}


vector<float> TRTParser::PostprocessResult(float *gpuOutputBuffer, const unsigned batchSize, const unsigned outputIndex, const bool softMax) {
	if (outputIndex >= this->outputTensors.size()){
		throw std::overflow_error("[ERROR] outputIndex is greater than number of outputTensor's index!\n");
	}
	//Create CPU buffer.
	size_t outputSize = this->GetDimensionSize(this->outputTensors.at(outputIndex).dims)/this->outputTensors.at(outputIndex).dims.d[0];
	vector< float > cpu_output(outputSize * batchSize);

	//Transfer data from GPU buffer to CPU buffer.
	if (!CudaCheck(cudaMemcpyAsync(cpu_output.data(), gpuOutputBuffer, cpu_output.size() * this->outputTensors.at(outputIndex).tensorSize, cudaMemcpyDeviceToHost))) {
		throw std::overflow_error("[ERROR] Get data from device to host failure!\n");
		abort();
	}
	return cpu_output;
}

float TRTParser::iou(Box A, Box B){
	int xA = max(A.x1, B.x1);
	int yA = max(A.y1, B.y1);
	int xB = min(A.x2, B.x2);
	int yB = min(A.y2, B.y2);

	float interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1);
	float boxAArea = (A.x2 - A.x1 + 1) * (A.y2 - A.y1 + 1);
	float boxBArea = (B.x2 - B.x1 + 1) * (B.y2 - B.y1 + 1);

	float iou = interArea / float(boxAArea + boxBArea - interArea);
	return iou;
}

vector<Box> TRTParser::NMS(vector<vector<float>> prediction, const float confidentThres, const float iouThres, const unsigned maxDetect){
	unsigned nc = this->outputTensors.at(3).dims.d[2] - 5;
	vector <Box> boxes, result;
	vector<float> classScores;

	for (unsigned i = 0; i < prediction.size(); i++){
		float boxScore = prediction.at(i).at(4);
		if (boxScore > confidentThres){
			unsigned x1, y1, x2, y2, w, h, classID;
			float classScore;
			w = prediction.at(i).at(2);
			h = prediction.at(i).at(3);
			x1 = prediction.at(i).at(0) - w/2;
			y1 = prediction.at(i).at(1) - h/2;
			x2 = x1 + w;
			y2 = y1 + h;

			vector<float> scores(&prediction.at(i).at(5), &prediction.at(i).at(5)+nc);
			auto maxElement = max_element(std::begin(scores), std::end(scores));
			classScore = *maxElement;
			classID = std::distance(scores.begin(), maxElement);

			Box box(x1, y1, x2, y2, classID, classScore);
			boxes.emplace_back(box);
			classScores.emplace_back(classScore);
			scores.clear();
		}
	}

	while (boxes.size() > 0){
		unsigned m = std::distance(classScores.begin(), max_element(std::begin(classScores), std::end(classScores)));
		Box M = boxes.at(m);
		result.emplace_back(M);
		
		classScores.erase(classScores.begin() + m);
		boxes.erase(boxes.begin() + m);

		for (unsigned i = 0; i < boxes.size(); i++){
			if (boxes.at(i).classID == M.classID ){
				if (iou(M, boxes.at(i)) > iouThres){
					classScores.erase(classScores.begin() + i);
					boxes.erase(boxes.begin() + i);
					i--;
				}
			}
		}
	}
	
	return result;
}

bool TRTParser::Inference(vector<cv::Mat> images, const bool softMax) {
	unsigned batchSize = images.size();
	unsigned nrofInputs = this->inputTensors.size();
	if (batchSize > this->maxBatchSize){
		cerr << "[ERROR] Batch size must be smaller or equal " << this->maxBatchSize << endl;
		return false;
	}

	//Create buffer on GPU device
	vector< void* > buffers(this->engine->getNbBindings());
	for (unsigned i = 0; i < this->engine->getNbBindings(); i++) {
		auto dims = this->engine->getBindingDimensions(i);
		size_t bindingSize;
		if (this->engine->bindingIsInput(i)){
			bindingSize = this->GetDimensionSize(dims) * this->inputTensors.at(i).tensorSize;
		}
		else{
			bindingSize = this->GetDimensionSize(dims) * this->outputTensors.at(i - nrofInputs).tensorSize;
		}
		if (!CudaCheck(cudaMalloc(&buffers[i], bindingSize))) return false;
	}

	//Allocate data to GPU. 
	//If you have multiple inputs add AllocateImageInput or AllocateNonImageInput with coresponding inputIndex
	if (!this->AllocateImageInput(images, (float*)buffers[0], 0)){
		cerr << "[ERROR] Allocate Input error!\n";
		return false;
	}

	if (nrofInputs > 1) {
		cerr << "[ERROR] Your must add AllocateImageInput or AllocateNonImageInput with coresponding inputIndex for other inputs above / add data for Inference()'s arguments then remove this condition at " << __FILE__ << ":" << __LINE__<< " to continue!\n";
		return false;
	}
	//Model Inference on GPU
	this->context->enqueueV2(buffers.data(), 0, nullptr);

	//Transfer result from GPU to CPU
	vector<float> result;
	cout << "'"<<this->outputTensors.at(3).tensorName << "':\n";
	try {
		result = this->PostprocessResult((float *)buffers[3+nrofInputs], batchSize, 3, softMax);
	}
	catch (exception& err) {
		cerr << err.what();
		return false;
	}

	
	//Transform 1D-result to 3D-vector
	vector<vector<vector<float>>> predictions;
	unsigned L = this->outputTensors.at(3).dims.d[1];
	unsigned K =  this->outputTensors.at(3).dims.d[2];

	for (unsigned l = 0; l < batchSize; l++){
		vector<vector<float>> predPerImage;
		for (unsigned k = 0; k < L; k++){
			vector<float> pred(&result.at(l*K*L + k*K) , &result.at(l*K*L + k*K) + K);
	 		predPerImage.emplace_back(pred);
	 		pred.clear();
		}
		predictions.emplace_back(predPerImage);
		predPerImage.clear();
	}

	unsigned imgH = this->inputTensors.at(0).dims.d[2];
	unsigned imgW = this->inputTensors.at(0).dims.d[3];
	auto imageSize = cv::Size(imgW, imgH);

	for (unsigned i = 0; i < batchSize; i++){
		auto boxes = NMS(predictions.at(i));
		auto image = images.at(i);
		cv::resize(image, image, imageSize, 0, 0, cv::INTER_AREA);
		for (auto&box : boxes){
			cout << box.classID << " " << box.score << " " << box.x1 << " " << box.y1 << " " << box.x2 << " " << box.y2 <<endl;

			cv::Point pt1(box.x1, box.y1);
			cv::Point pt2(box.x2, box.y2);
			cv::rectangle(image, pt1, pt2, cv::Scalar(0, 255, 0));
		}
		cv::imwrite("result/"+to_string(temp)+".png", image);
		temp++;
		
		cv::imshow("test", image);
		cv::waitKey();
	}

	//Deallocate memory to avoid memory leak
	predictions.clear();
	result.clear();
	for (void* buf : buffers) {
		if (!CudaCheck(cudaFree(buf))) return false;
	}
	return true;
}
