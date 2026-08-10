CC=g++
CFLAGS=-std=c++17 -DCOMMANDER_DISABLE_LEGACY_LOG_MACROS -I../../include -I/usr/include/eigen3
BIN=client example test
LFLAGS=-L../../../lib -lcommander -lboost_program_options -lfmt -lzmq
#LFLAGS=-lcommander -lboost_program_options -lfmt -lzmq

all: $(BIN)

%: %.cpp
	$(CC) -o $@ $< $(CFLAGS) $(LFLAGS)

clean:
	rm -f $(BIN)
