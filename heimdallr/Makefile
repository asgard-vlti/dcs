CC=g++

CFLAGS  = -std=c++17 -Wall -Wextra -ggdb -O1 -I../commander/include
PREDICTIVE_CFLAGS = -std=c++17 -Wall -Wextra -O3 -DNDEBUG -I/usr/include/eigen3
PREDICTIVE_BENCHMARK = /tmp/heimdallr_predictive_benchmark
PREDICTIVE_SIMULATION = /tmp/heimdallr_predictive_simulation
PREDICTIVE_PARAMETERS = /tmp/heimdallr_predictive_parameters
PREDICTIVE_SNAPSHOT = /tmp/heimdallr_predictive_snapshot
COMMANDER_LIB = ../lib/libcommander.a
LDFLAGS = -L../lib -lm -lzmq -lImageStreamIO -lb64 -lpthread -lboost_program_options -lfmt -lfftw3
EXEC    = heimdallr
OBJECTS = heimdallr.o forward_ft.o fringe_tracker.o predictive_control.o ddspc_snapshot.o dl_offload.o cam_client.o
SIM_OBJECTS = $(OBJECTS:.o=.sim.o)
SIM_CFLAGS = $(CFLAGS) -DSIMULATE -I../commander/include -I../libImageStreamIO -I/usr/include/eigen3 -I/usr/include/toml++

# PREFIX is environment variable, but if it is not set, then set default value
ifeq ($(PREFIX),)
    PREFIX := /usr/local
endif

# the makefile instructions

all: $(EXEC)

sim: heimdallr_sim

heimdallr_sim: $(SIM_OBJECTS) $(COMMANDER_LIB)
	$(CC) -o $@ $^ $(LDFLAGS)

%.sim.o: %.cpp heimdallr.h
	$(CC) -o $@ -c $< $(SIM_CFLAGS)

heimdallr: $(OBJECTS) $(COMMANDER_LIB)
	$(CC) -o $@ $^ $(LDFLAGS)

$(COMMANDER_LIB): ../commander/src/server/socket.cpp ../commander/src/server/interactive.cpp ../commander/include/commander/server/inline_arguments.h
	$(MAKE) -C ../commander/src

%.o: %.cpp heimdallr.h
	$(CC) -o $@ -c $< $(CFLAGS)

$(OBJECTS) $(SIM_OBJECTS): predictive_control.hpp
heimdallr.o heimdallr.sim.o fringe_tracker.o fringe_tracker.sim.o: ddspc_snapshot.hpp
heimdallr.o heimdallr.sim.o fringe_tracker.o fringe_tracker.sim.o: servo_transitions.hpp
fringe_tracker.o fringe_tracker.sim.o: fringe_frame_wait.hpp
heimdallr.o heimdallr.sim.o: ddspc_tuning.hpp

.PHONY: predictive_benchmark
predictive_control.o: predictive_control.cpp predictive_control.hpp
	$(CC) $(PREDICTIVE_CFLAGS) -o $@ -c $<

predictive_control.sim.o: predictive_control.cpp predictive_control.hpp
	$(CC) $(PREDICTIVE_CFLAGS) -DSIMULATE -o $@ -c $<

ddspc_snapshot.o: ddspc_snapshot.cpp ddspc_snapshot.hpp
	$(CC) $(PREDICTIVE_CFLAGS) -o $@ -c $<

ddspc_snapshot.sim.o: ddspc_snapshot.cpp ddspc_snapshot.hpp
	$(CC) $(PREDICTIVE_CFLAGS) -DSIMULATE -o $@ -c $<

predictive_benchmark: tests/predictive_benchmark.cpp predictive_control.o
	$(CC) $(PREDICTIVE_CFLAGS) -o $(PREDICTIVE_BENCHMARK) $< predictive_control.o
	$(PREDICTIVE_BENCHMARK)

.PHONY: predictive_simulation
predictive_simulation: tests/predictive_simulation.cpp predictive_control.o
	$(CC) $(PREDICTIVE_CFLAGS) -o $(PREDICTIVE_SIMULATION) $< predictive_control.o
	$(PREDICTIVE_SIMULATION)

.PHONY: predictive_parameters
predictive_parameters: tests/predictive_parameters.cpp predictive_control.o
	$(CC) $(PREDICTIVE_CFLAGS) -I/usr/include/toml++ -I../commander/include -o $(PREDICTIVE_PARAMETERS) $< predictive_control.o -L../lib -lcommander -lzmq -lboost_program_options -lfmt -lpthread
	$(PREDICTIVE_PARAMETERS)

.PHONY: predictive_snapshot
predictive_snapshot: tests/predictive_snapshot.cpp predictive_control.o ddspc_snapshot.o
	$(CC) $(PREDICTIVE_CFLAGS) -o $(PREDICTIVE_SNAPSHOT) $< predictive_control.o ddspc_snapshot.o -lpthread
	$(PREDICTIVE_SNAPSHOT)

clean:
	rm -rf *.o *.so
	rm -rf *~

mrproper: clean
	rm -rf $(EXEC) heimdallr_sim

install:
	install -D $(EXEC) $(PREFIX)/bin/
	setcap 'cap_sys_nice=eip' $(PREFIX)/bin/$(EXEC)
	cp def.toml $(PREFIX)/etc

priority:
	setcap 'cap_sys_nice=eip' $(EXEC)