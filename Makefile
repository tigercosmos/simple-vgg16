CXX=clang++
CXXFLAGS=-Iobjs/ -O3 -std=c++17 -Wall -fopenmp 

APP_NAME=vgg16
TEST_NAME=vgg16_test
OBJDIR=objs

ifeq ($(BENCHMARK),1)
	CXXFLAGS += -DBENCHMARK
endif

default: $(APP_NAME)

.PHONY: dirs clean

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(APP_NAME) $(TEST_NAME)

OBJS :=  $(OBJDIR)/main.o
TEST_OBJS :=  $(OBJDIR)/test.o

$(APP_NAME): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

test: dirs $(TEST_OBJS)
		$(CXX) $(CXXFLAGS) -o $(TEST_NAME) $(TEST_OBJS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@