# Executable
EXE    = DomainsReorder
	
# Compiler, Linker Defines
CC      = g++
CFLAGS  = -Wall -O3 -Wno-deprecated -m64 -I. -Wno-unused -std=c++11 

# Compile and Assemble C++ Source Files into Object Files
%.o: %.cc
	echo compiling...$<
	$(CC) $(CFLAGS) -c $<
# -o $@
# Source and Object files
SRC    = $(wildcard *.cpp)
OBJ    = $(patsubst %.cpp, %.o, $(SRC))

# Link all Object Files with external Libraries into Binaries
$(EXE): $(OBJ)
	echo linking...
	$(CC) $(CFLAGS) $(OBJ) $(LDFLAGS) liblbfgs.a -o $(EXE) -lz -lpthread

.PHONY: clean
clean:
	 -rm -f core *.o

