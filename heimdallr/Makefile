CC=g++

CFLAGS  = -std=c++17 -Wall -Wextra -ggdb -O1
PREDICTIVE_CFLAGS = -std=c++17 -Wall -Wextra -O3 -DNDEBUG -I/usr/include/eigen3
PREDICTIVE_BENCHMARK = /tmp/heimdallr_predictive_benchmark
PREDICTIVE_SIMULATION = /tmp/heimdallr_predictive_simulation
LDFLAGS = -lcommander -lm -lzmq -lImageStreamIO -lb64 -lpthread -lboost_program_options -lfmt -lfftw3  
EXEC    = heimdallr
OBJECTS = heimdallr.o forward_ft.o fringe_tracker.o predictive_control.o dl_offload.o cam_client.o
SIM_OBJECTS = $(OBJECTS:.o=.sim.o)
SIM_CFLAGS = $(CFLAGS) -DSIMULATE -I../commander/include -I../libImageStreamIO -I/usr/include/eigen3 -I/usr/include/toml++

# PREFIX is environment variable, but if it is not set, then set default value
ifeq ($(PREFIX),)
    PREFIX := /usr/local
endif

# the makefile instructions

all: $(EXEC)

sim: heimdallr_sim

heimdallr_sim: $(SIM_OBJECTS)
	$(CC) -o $@ $^ -L../lib $(LDFLAGS)

%.sim.o: %.cpp heimdallr.h
	$(CC) -o $@ -c $< $(SIM_CFLAGS)

heimdallr: $(OBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.cpp heimdallr.h
	$(CC) -o $@ -c $< $(CFLAGS)

.PHONY: predictive_benchmark
predictive_control.o: predictive_control.cpp predictive_control.hpp
	$(CC) $(PREDICTIVE_CFLAGS) -o $@ -c $<

predictive_control.sim.o: predictive_control.cpp predictive_control.hpp
	$(CC) $(PREDICTIVE_CFLAGS) -DSIMULATE -o $@ -c $<

predictive_benchmark: tests/predictive_benchmark.cpp predictive_control.o
	$(CC) $(PREDICTIVE_CFLAGS) -o $(PREDICTIVE_BENCHMARK) $< predictive_control.o
	$(PREDICTIVE_BENCHMARK)

.PHONY: predictive_simulation
predictive_simulation: tests/predictive_simulation.cpp predictive_control.o
	$(CC) $(PREDICTIVE_CFLAGS) -o $(PREDICTIVE_SIMULATION) $< predictive_control.o
	$(PREDICTIVE_SIMULATION)

clean:
	rm -rf *.o *.so
	rm -rf *~

mrproper: clean
	rm -rf $(EXEC) heimdallr_sim

install:
	install -D $(EXEC) $(PREFIX)/bin/
	setcap 'cap_sys_nice=eip' $(PREFIX)/bin/$(EXEC)
	cp def.toml $(PREFIX)/etc
