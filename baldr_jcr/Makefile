CC=g++

CFLAGS  = -std=c++17 -Wall -Wextra -ggdb -O1 -I../libImageStreamIO -I../commander/include -DEIGEN_STACK_ALLOCATION_LIMIT=259200
LDFLAGS = -L../libImageStreamIO -L../lib -lcommander -lm -lzmq -lImageStreamIO -lb64 -lpthread -lboost_program_options -lfmt -lfftw3 -lcfitsio
EXEC    = baldr
OBJECTS = baldr.o servo_loop.o

# PREFIX is environment variable, but if it is not set, then set default value
ifeq ($(PREFIX),)
    PREFIX := /usr/local
endif

# the makefile instructions

all: $(EXEC)

baldr: $(OBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS) 

%.o: %.cpp baldr.h
	$(CC) -o $@ -c $< $(CFLAGS)

clean:
	rm -rf *.o *.so
	rm -rf *~

mrproper: clean
	rm -rf $(EXEC)

# This was copied over from baldr_tt but I don't recommend using it:
# install:
# 	install -D $(EXEC) $(PREFIX)/bin/
# 	setcap 'cap_sys_nice=eip' $(PREFIX)/bin/$(EXEC)
# 	cp def1.toml $(PREFIX)/etc
# 	cp def2.toml $(PREFIX)/etc
# 	cp def3.toml $(PREFIX)/etc
# 	cp def4.toml $(PREFIX)/etc
# 	cp modes.fits $(PREFIX)/etc
