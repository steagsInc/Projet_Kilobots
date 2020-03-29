SOURCES=main.c perceptron.c Layer.c matmul.cu

refreshWeights: $(SOURCES)
	nvcc main.c perceptron.c Layer.c matmul.cu -lm -o refreshWeights
