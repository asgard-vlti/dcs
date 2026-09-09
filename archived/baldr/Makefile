CC=g++

# -ggdb and -g  for debugging
CFLAGS  = -std=c++17 -Wall -Wextra -O3 -march=native -I../libImageStreamIO -Icommander -I/usr/include/eigen3 -Icontrollers -I../commander/include
#pthread is only needed for ImageStreamIO
LDFLAGS = -lcommander -lm -lzmq -lImageStreamIO -lb64 -lboost_program_options -lfmt -lpthread -lcfitsio
EXEC    = baldr
OBJECTS = baldr.o rtc.o telemetry.o controllers/controller.o

# added 
LDFLAGS += -L../lib
LDLIBS  += -lcommander

# PREFIX is environment variable, but if it is not set, then set default value
ifeq ($(PREFIX),)
    PREFIX := /usr/local
endif

# the makefile instructions

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

# CC=g++

# # -ggdb and -g  for debugging
# CFLAGS  = -std=c++17 -Wall -Wextra -O3 -march=native \
#           -I/Documents/dcs/libImageStreamIO -I/usr/include/eigen3
# #-Icontrollers #<- need to bugshoot controllers 
# #pthread is only needed for ImageStreamIO
# LDFLAGS = -lcommander -lm -lzmq -lImageStreamIO -lb64 -lboost_program_options -lfmt -lpthread -lcfitsio
# EXEC    = baldr
# OBJECTS = baldr.o rtc.o telemetry.o
# #controllers/controller.o # <- need to bugshoot compilation of contorllers 

# # PREFIX is environment variable, but if it is not set, then set default value
# ifeq ($(PREFIX),)
#     PREFIX := /usr/local
# endif

# # the makefile instructions

# all: $(EXEC)

# baldr: $(OBJECTS)
# 	$(CC) -o $@ $^ $(LDFLAGS)

# %.o: %.cpp 
# 	$(CC) -o $@ -c $< $(CFLAGS)

# clean:
# 	rm -rf *.o *.so
# 	rm -rf *~

# mrproper: clean
# 	rm -rf $(EXEC)

# install:
# 	install -D $(EXEC) $(PREFIX)/bin/


# ### below is stable before adding in controllers directory 
# # CC=g++

# # # -ggdb and -g  for debugging
# # CFLAGS  = -std=c++17 -Wall -Wextra -O3 -march=native -I/Documents/dcs/libImageStreamIO -I/usr/include/eigen3
# # #pthread is only needed for ImageStreamIO
# # LDFLAGS = -lcommander -lm -lzmq -lImageStreamIO -lb64 -lboost_program_options -lfmt -lpthread -lcfitsio
# # EXEC    = baldr
# # OBJECTS = baldr.o rtc.o telemetry.o

# # # PREFIX is environment variable, but if it is not set, then set default value
# # ifeq ($(PREFIX),)
# #     PREFIX := /usr/local
# # endif

# # # the makefile instructions

# # all: $(EXEC)

# # baldr: $(OBJECTS)
# # 	$(CC) -o $@ $^ $(LDFLAGS)

# # %.o: %.cpp 
# # 	$(CC) -o $@ -c $< $(CFLAGS)

# # clean:
# # 	rm -rf *.o *.so
# # 	rm -rf *~

# # mrproper: clean
# # 	rm -rf $(EXEC)

# # install:
# # 	install -D $(EXEC) $(PREFIX)/bin/
