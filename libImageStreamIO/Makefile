CC=gcc
CFLAGS  = -W -Wall -ggdb

CC=g++
CFLAGS  = -std=c++17 -Wall -Wextra -ggdb -O1

LIB     = libImageStreamIO.a
OBJECTS = ImageStreamIO.o

# PREFIX is environment variable, but if it is not set, then set default value
ifeq ($(PREFIX),)
    PREFIX := /usr/local
endif

all: $(LIB)

$(LIB): $(OBJECTS)
	ar rcs $(LIB) $(OBJECTS)

%.o: %.c
	$(CC) -o $@ -c $< $(CFLAGS)

imft: ImCreate_ft.c
	$(CC) -o $@ ImageStreamIO.c $< $(CFLAGS) -O3 -lm -lpthread -lfftw3 

clean:
	rm -rf *.o *~

install:
	install -D $(LIB) $(PREFIX)/lib/
	install -D *.h $(PREFIX)/include/
