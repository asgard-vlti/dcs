CC=g++

# FYI: it turns out that the order in which the libraries are linked matters!
# they probably are linked in the reverse order, and lpthread should be after
# ImageStreamIO.


CFLAGS  = -std=c++17 -W -Wall -Wextra
LDFLAGS = -lcommander -lm -lzmq -lImageStreamIO -lpthread -lboost_program_options -lfmt -lcjson -lcfitsio
EXEC    = asgard_cam_server
# OBJECTS = asgard_ZMQ_CRED1_server.o
OBJECTS = cred1_onsky_server.o cred1_cli_tools.o

# PREFIX is environment variable, but if it is not set, then set default value
ifeq ($(PREFIX),)
    PREFIX := /usr/local
endif

# adding the PCI EDT specific compilation options
CFLAGS  += -I /opt/EDTpdv
LDFLAGS += -L /opt/EDTpdv -lpdv

# the makefile instructions

all: $(EXEC)

debug: debug_kaya.o
	$(CC) -o $@ $^ $(LDFLAGS)

# A test program for thread priorities
test: test.c
	$(CC) -o test test.c -lpthread

asgard_cam_server: $(OBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) -o $@ -c $< $(CFLAGS)

clean:
	rm -rf *.o *.so
	rm -rf *~

mrproper: clean
	rm -rf $(EXEC)

install:
	install -D $(EXEC) $(PREFIX)/bin/
	cp cred1_split.json ~/.config/
