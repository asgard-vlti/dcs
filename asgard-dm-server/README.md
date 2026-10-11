# asgard-dm-server
My working version of the "classical" shared memory driven control of deformable mirror(s) for ASGARD

## Note

I'm a little rusty with with semaphores, so I'm not fully taking advantage of the features of the library... but it seems that implemented as it is, the code can monitor and control four DMs at the ~3.5 kHz rate, which is the best that we'll ever be able to operate all this stuff here.

## The ASGARD project and its DMs

ASGARD is the name of a suite of instruments that was officially accepted for installation at the VLTI by ESO in 2023. ASGARD is a suite of instruments consisting of 4 distinct modules:

- HEIMDALLR: an interferometric recombiner that serves as the primary fringe tracker for ASGARD
- BALDR: a tune-up AO system located in the VLTI focal laboratory to compensate for aberrations introduced in the VLTI delay, after the initial AO correction provided at the coudé focus of the telescopes (NAOMI for the ATs or GPAO for the UTs)
- NOTT: a medium-resolution L-band high-contrast imaging mode using nulling interferometry
- BIFROST: a high-spectral resolution J-band instrument that will eventually feature a dual arm setup

The AO and fringe-tracking services respectively provided by BALDR and HEIMDALLR are going to primarily rely on a series of four (one per telescope input beam) identical deformable mirrors (DMs). These DMs are the first optical elements that the beam provided by the VLTI will encounter on the ASGARD optical table.

This repository is to keep track of the development of my take on the DM software development. The idea is to have each DM driven on the HEIMDALLR+BALDR (HEIBAL?) control machine (a linux box) by a server, monitoring a series of pre-allocated shared memory data structures that look like 2D DM maps, and whenever a change occurs on any of these input channels, update the command to be sent to the DM (according to a set of pre-agreed rules - usually, a straightforward linear combination of the different command channels), and send the command to the electronics DM driver, connected to the computer via a USB2 bus.

This architecture is very much in the spirit of the approach used for the SCExAO instrument. The scale is different: instead of a single instance of server controlling a 50x50 actuator grid, the HEIBAL computer will run four instances of a server, each talking to its pre-assigned 12x12 actuator DM.

This code relies on the [ImageStreamIO C](https://github.com/milk-org/ImageStreamIO) library developed by O. Guyon that is a part of the [milk-org](https://github.com/milk-org) organisation

A local version of this library is part of the software running the HEIMDALLR+BALDR DCS

## Compilation

Refer to the provided [Makefile](./Makefile) and ensure that you have all the required libraries installed. Assuming that all is in place, simply compile the code with:

> make

Like any regular piece of C code that comes with a makefile :-)

> sudo make install

Will install the executable in the =/usr/local/bin/= directory.

## Using the DM server

With the driver properly installed, the following command should run from anywhere on the computer:

> asgard_dm_server

The program expects to receive commands through a simple ZMQ messaging interface. The program offers a rudimentary ZMQ command line environment that understands a small number of commands. The interface is not very sophisticated because this server runs in the background and does most things without requiring frequent input from the user. Most of the action happens at the shared memory data structure level: the user modifies a channel, the server automatically updates the combined command, converts it into a command for the driver and sends this command to the PCI board connected to the driver.

For testing purposes, a simple python script is provided that will give a prompt to send ZMQ commands to the driver:

> python DM_ZMQ_control_client.py

### Configuration

In the early stages of design/operation, since we're not quite sure of how many virtual mirrors we'd like to work with, the DM server requires at least one =start= command for the DM server to start monitoring the SHM structurees and interacting with the actual BMC DM drivers. Expected ZMQ command examples:

> set_nch 8
> 
> start
>
> status

The first of these commands sets the number of channels (virtual DMs) to set up. By default, the code will start 4... but who knows how crazy are going to be. In this example, we set the number of channels to 8 (obviously).

Now that the channels have been requested (and the shared memory allocated, which can be confirmed by looking at the content of the **/dev/shm/** directory, you can start the main DM server loop with the =start= command. From now on, any valid modification of one of the virtual DM channels will be taken into account to send a command to the DM.

### Clean quitting

To quit the execution of the program in a clean manner, back to the ZMQ interface to send:

> stop
>
> quit

- The **stop** command interrupts the thread that was monitoring the different channels.
- The **quit** command closes the program and frees any dynamically allocated memory during the execution of the server.

### Changing the number of channels

If you ever need to change the number of virtual DM channels, you first need to **stop** the thread, then you set the new number of channels, and then you **start** the thread again.

### Summary of available commands

The commands made available allow for configuration / reconfiguration of the server.


| command | description                                              | parameter |
|---------|----------------------------------------------------------|-----------|
| help    | prints the help menu                                     | N.A       |
| set_nch | sets/updates the number of channels used by the server   | integer   |
| get_nch | prints the number of channels used by the server         | N.A.      |
| start   | triggers the channel monitoring process                  | N.A.      |
| status  | queries the status of the driver ("running" or "waiting")| N.A.      |
| save    | enables or disables DM command archiving                 | 1 or 0    |
| get_save_status | returns `{"mode": 0 or 1, "started_unix_s": timestamp}` | N.A. |
| quit    | closes the shm data structures and shuts the driver down | N.A.      |

When `save 1` is active, the server writes combined DM command FITS files as
`/data/YYYYMMDD/dms_THH:MM:SS.fits` on `mimir`. `started_unix_s` is zero while
saving is off and records the start of the current save session while it is on.

One convenient "reset" command is currently missing and will be added back in a bit.

It'll take two integer parameters: the dm_id and the channel_id.

## Conclusion

Much of the action will take place at the shared memory level directly so you should have very little to do here beyond issuing **start/stop** commands.
