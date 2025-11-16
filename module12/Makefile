TARGET = poly
SRC    = assignment.cpp
KERNEL = polynomial_transform.cl
CXX = g++
CXXFLAGS = -std=c++17 -O2 -I".\CL" -L".\CL"
LIBS = -lOpenCL

$(TARGET): $(SRC) $(KERNEL)
	$(CXX) $(CXXFLAGS) $(SRC) $(LIBS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET) $(N)

clean:
	rm -f $(TARGET)
