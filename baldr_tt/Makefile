CC=g++

CFLAGS  = -std=c++17 -Wall -Wextra -ggdb -O1
LDFLAGS = -lcommander -lm -lzmq -lImageStreamIO -lb64 -lpthread -lboost_program_options -lfmt -lfftw3 -lcfitsio
EXEC    = baldr_tt
OBJECTS = baldr_tt.o servo_loop.o

# PREFIX is environment variable, but if it is not set, then set default value
ifeq ($(PREFIX),)
    PREFIX := /usr/local
endif

# the makefile instructions

all: $(EXEC)

baldr_tt: $(OBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.cpp baldr_tt.h
	$(CC) -o $@ -c $< $(CFLAGS)

clean:
	rm -rf *.o *.so
	rm -rf *~

mrproper: clean
	rm -rf $(EXEC)

install:
	install -D $(EXEC) $(PREFIX)/bin/
	chmod +s $(PREFIX)/bin/$(EXEC)
	cp def.toml $(PREFIX)/etc
	
