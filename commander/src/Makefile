CC=g++
CFLAGS=-std=c++17 -I../include
OBJ=server/interactive.o server/single_command.o server/socket.o client/socket.o registry.o server.o module.o utils.o

all: $(OBJ)
	ar rcs ../../lib/libcommander.a $(OBJ)

%.o: %.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

install:
	cp ../../lib/libcommander.a /usr/local/lib
	cp -r ../include/commander /usr/local/include

clean:
	rm -f *.o
	rm -f server/*.o
	rm -f client/*.o
