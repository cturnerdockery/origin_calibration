from __future__ import annotations
from pathlib import Path
from collections import namedtuple
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

# origin_tools is a Sonardyne internal data processing utility library
import origin_tools
from Tools.decoder import GramDecoder


"""

 Calibration Window Finder
================================

Orignal Author: Thomas Comerford
Adpated by: Chloe Turner-Dockery 

1. Loads B-gram data using GramDecoder.
2. Displays a full-record echogram for quick inspection.
3. Extracts data corresponding to calibration sphere depths.
4. Saves each depth interval as a separate B-gram file.
5. Produces a panel plot showing the sphere at each calibration depth.

Steps:
1. Set MODE = "preview"
2. Run the script.
3. Note the sphere windows.
4. Add Window entries to the schedule.
5. Set MODE = "extract"
6. Run the script again.
7. Calibration excerpts will be written to folder_path/Excerpts/
8. Defined windows can then be fed into calibration_main 

"""

### User configurations

# Path containing either ADCP or Echo bgrams from a calibration run
folder_path = r"path\to\bgram"


#Set MODE

# preview : Display the entire B-gram to find sphere windows.
# extract : Create calibration excerpts defined in the windows list.

MODE = "preview"
# MODE = "extract"


# Create Windows from preview
Window = namedtuple(
    "Window",["depth","beam","start_time","end_time"])

Schedule = namedtuple(
    "Schedule",["schedule_name","instrument","sphere_diameter_m","ping_length_ms","theoretical_target_strength","windows"],)

schedule = Schedule(
    schedule_name="calibration_x",
    instrument="name",
    sphere_diameter_m=0.01,
    ping_length_ms=100,
    theoretical_target_strength=-51.46,
    windows=[
        Window(19.95, 4, datetime(2025, 9, 23, 13, 2, 15), datetime(2025, 9, 23, 13, 3, 16)),
        Window(19.11, 4, datetime(2025, 9, 23, 13, 3, 36), datetime(2025, 9, 23, 13, 4, 41)),

    ],)


### Mode scripts

if MODE == "preview":

    print(f"Plotting overview for {schedule.schedule_name}")
    gd = GramDecoder(folder_path)

    data = gd.extract(["complex_iq", "time"], max_length=1000, pings_per_ensemble=10)
    z = data["complex_iq"][:, 4, :]
    time = data["time"]

    plt.imshow(np.log10(1 + np.abs(z)).T,
               origin="lower",
               aspect="auto",
               extent=[time[0]/86400, time[-1]/86400, 0, 5000*0.012],
               cmap='magma')
    plt.colorbar()
    plt.xlabel("Time (BST)",fontsize=10)
    plt.ylabel("Slant range (m)",fontsize=10)
    plt.title("calibration - unitless intensity", fontsize=10)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.show()
    exit()

if MODE == "extract":

    print(f"Schedule {schedule.schedule_name}")

    gd = GramDecoder(folder_path)
    n_windows = len(schedule.windows)

    fig, axes = plt.subplots(ncols=n_windows)

    for i, window in enumerate(schedule.windows):

        print(
            f"  Window {i+1}/{n_windows} "
            f"| Depth {window.depth} m"
        )

        gd.start_time = window.start_time + timedelta(hours=1)
        gd.end_time = window.end_time + timedelta(hours=1)

        excerpt_folder = Path(folder_path) / "Excerpts"
        excerpt_folder.mkdir(exist_ok=True)

        outfile = (excerpt_folder/ f"{schedule.schedule_name}_beam_{window.beam}_depth_{int(window.depth * 100)}cm.bgram")

        with open(outfile, "wb") as f:
            for ping in gd.get_pings():
                f.write(origin_tools.acdm.adcp_coredata_messages_serialise(ping))

        print(f"\tSaved: {window.depth}")

        data = gd.extract(["complex_iq", "time"], max_length=3000)
        z = data["complex_iq"][:, window.beam, :]
        time = data["time"]

        axes[i].imshow(
            np.log10(1 + np.abs(z)).T,
            origin="lower",
            aspect="auto",
            extent=[time[0] / 86400,time[-1] / 86400,0,5000 * 0.012,],
            clim=[2, 4.5],
            cmap="magma",
        )

        axes[i].set_ylim(window.depth - 1,window.depth + 1)
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        axes[i].tick_params(axis="x", rotation=45)

    plt.show()
