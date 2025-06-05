
NVCC        := nvcc
NVCC_FLAGS  := -O2 -arch=sm_75

# Find all CUDA source files
SRCS        := $(wildcard *.cu)
OBJS        := $(SRCS:.cu=.o)
TARGET      := main

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
