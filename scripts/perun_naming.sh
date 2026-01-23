#!/usr/bin/env bash

perun monitor --app_name="name that I need for the hdf5 file"

# or: writes to the same hdf5 file but creates levels on the run
perun monitor --run_id="1,2,3"
