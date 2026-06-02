CC=g++

# -ggdb and -g  for debugging
CFLAGS  = -std=c++17 -Wall -Wextra -O3 -march=native -I../libImageStreamIO \
	-Icommander -Icontrollers -I../commander/include
#pthread is only needed for ImageStreamIO
LDFLAGS = -L../libImageStreamIO -L../lib -lm -lzmq -lboost_program_options \
	-lfmt -lpthread -lcfitsio -lImageStreamIO -lcommander
EXEC    = baldr
OBJECTS = baldr.o rtc.o telemetry.o controllers/controller.o

# added 
LDFLAGS += -L../lib
LDLIBS  += -lcommander

# PREFIX is environment variable, but if it is not set, then set default value
ifeq ($(PREFIX),)
    PREFIX := /usr/local
endif

all: $(EXEC)

baldr: $(OBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.cpp 
	$(CC) -o $@ -c $< $(CFLAGS)

clean:
	rm -rf *.o *.so
	rm -rf *~

mrproper: clean
	rm -rf $(EXEC)

install:
	install -D $(EXEC) $(PREFIX)/bin/
