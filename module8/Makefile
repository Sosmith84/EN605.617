NVCC = nvcc
CFLAGS = -std=c++17 -O2
LIBS = -lcusolver -lcusparse -lcudart
TARGET = assignment
SRC = assignment.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) $(SRC) -o $(TARGET) $(LIBS)

run: $(TARGET)
	./$(TARGET) $(ARGS)

clean:
	rm -f $(TARGET)

