#!/usr/bin/env bash

perun monitor --app_name="name that I need for the hdf5 file"

# or: writes to the same hdf5 file but creates levels on the run
perun monitor --run_id="SWIN_MLP512" code.py config
perun monitor --run_id="SWIN_MLP256" code.py config
