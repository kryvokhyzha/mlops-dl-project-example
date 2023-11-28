run_export_script: export_model.py
	python export_model.py
run_tensorrt:
	docker run -it --rm -v ./model_repository:/model nvcr.io/nvidia/tensorrt:23.04-py3
run_tensorrt_gpus:
	docker run -it --rm --gpus '"device=2"' -v ./model_repository:/model nvcr.io/nvidia/tensorrt:23.04-py3
convert_onnx_to_trt:
	trtexec --onnx=/model/onnx-bert/1/model.onnx --builderOptimizationLevel=5 --hardwareCompatibilityLevel=ampere+ --saveEngine=/model/onnx-bert-trt/1/model.plan --minShapes=INPUT_IDS:1x16,ATTENTION_MASK:1x16 --maxShapes=INPUT_IDS:8x16,ATTENTION_MASK:8x16 --optShapes=INPUT_IDS:4x16,ATTENTION_MASK:4x16 --fp16 --profilingVerbosity=detailed
run_triton_sdk:
	docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:23.04-py3-sdk
perf_analyzer:
	perf_analyzer -m onnx-bert -u localhost:8500 --concurrency-range 1:5 --shape INPUT_IDS:1,16 --shape ATTENTION_MASK:1,16 --input-data zero
