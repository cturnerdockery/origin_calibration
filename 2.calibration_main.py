import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import csv
from datetime import datetime, timedelta
from collections import namedtuple

# Sonardyne internal data processing utility library
from origin_tools.decoder import GramDecoder
from origin_tools.geometry import cell_length


"""
ADCP and Echosounder Calibration Pipeline
========================================

This script processes calibration sphere data from both ADCP and Echosounder
(B-gram) datasets and produces calibrated backscatter and unitless intensity results.

It performs the following steps:

1. Loads ADCP and Echosounder B-gram data using GramDecoder.
2. Applies user-defined calibration windows (time + depth + beam index) from "window_finder".
3. Extracts intensity profiles for each calibration window.
4. Detects the calibration sphere peak (with pulse train smoothing for ADCP data).
5. Writes per-window results to a calibration CSV file.
6. Generates diagnostic plots for each window.
7. Performs nonlinear calibration fitting (saturation-aware).
8. Produces final comparison plots for ADCP vs Echosounder.

Outputs:
- calibration.csv (per-window metrics)
- window_###.png (diagnostic intensity profiles)
- calibration_summary.txt (fit results)
- calibration_plot.png (final comparison plots)

Workflow:
1. Set ADCP_BGRAM_PATH and ECHO_BGRAM_PATH
2. Set user specified constants
3. Define calibration windows in `schedule`
4. Run script (main())
5. Inspect plots and CSV output
6. Use results for further calibration analysis
    The calibration factor produced in calibration_summary.txt can be applied to the data using TS_calibration and Sv_calbration scripts

Notes:
- Times are shifted by +1 hour (BST adjustment)
"""

ADCP_BGRAM_PATH = Path(r"E:\path\to\A\bgram")
ECHO_BGRAM_PATH = Path(r"E:\path\to\B\bgram")
OUTPUT_ROOT = Path(r"E:\path\to\folder\for\output")

# User specified constants
ABSORPTION_COEFFICIENT = 0.183
c = 1500

NOAA_TS_ADCP = -51.30 # << https://www.fisheries.noaa.gov/data-tools/standard-sphere-target-strength-calculator
NOAA_TS_ECHO = -51.46

TEST_NUMBER = 1
SUSPENSION = 1
UNIT = 1
SPHERE_SIZE = 10
PULSE_LENGTH_US = 100
GAIN = 0

K_SIGMA = 2

PLOT_LABEL_ADCP = "ADCP"
PLOT_LABEL_ECHO = "ECHO"
plot_colour_adcp = "lightseagreen"
plot_colour_echo = "darkorange"
plot_marker = "o"

depths_m = np.linspace(-30, -0.5, 1000)


# Specify expected sphere depths

# Window(expectedDepth, beamNumber, datetime(yyyy, mm/m, dd/d, hh/h, MM/M, ss/s), datetime(yyyy, mm/m, dd/d, hh/h, MM/M, ss/s)),
Window = namedtuple("Window", ["depth", "beam_index", "start_time", "end_time"])


# Example:
schedule = [
    Window(14.08,4, datetime(2025, 7, 18, 10, 24, 5), datetime(2025, 7, 18, 10, 26, 47)),
    Window(13.04,4, datetime(2025, 7, 18, 10, 26, 56), datetime(2025, 7, 18, 10, 29, 53)),
]

CSV_COLS = {
    "test_number": 0,
    "suspension": 1,
    "unit": 2,
    "sphere_size": 3,
    "pulse_length_us": 4,
    "depth": 5,
    "intensity": 6,
    "blank": 7,
    "blank2": 8,
    "gain": 9,
    "noaa_ts": 10,
    "max_recorded_intensity": 11,
}


def locate_intensity_peak(intensity, range_estimate, cell_size):
    """
       Locate the peak backscatter intensity an expected range window.

       This function finds the maximum intensity value in a vertical profile
       (range-backscatter), restricted to a small search window
       around the expected target range.

       The method:
       1. Collapses the intensity array to a single profile using max over pings.
       2. Restricts search to ±0.25 m around the expected range.
       3. Identifies the index of the maximum value within this window.
       4. Returns the corresponding depth and intensity value.

       Parameters
       ----------
       intensity : np.ndarray
       range_estimate : float
           Expected target depth (m) used as the center of the search window.
       cell_size : float
           Vertical resolution per cell (m).

       Returns
       -------
       peak_depth : float
           Depth (m) of the detected intensity maximum.
       peak_intensity : float
           Intensity value at the detected peak.
       max_profile : np.ndarray
           Maximum intensity profile collapsed over pings/beams.
       """
    max_profile = np.max(intensity, axis=0)

    start = max(0, int((range_estimate - 0.25) / cell_size))
    end = min(len(max_profile), int((range_estimate + 0.25) / cell_size))

    local = max_profile[start:end]
    peak_idx = start + np.argmax(local)

    return peak_idx * cell_size, max_profile[peak_idx], max_profile


def locate_intensity_peak_moving_avg(intensity, range_estimate, cell_size, pulse_length_m):
    """
       Locate the peak backscatter intensity using a moving-average smoothed profile.

        This function finds the maximum intensity value in a vertical profile
       (range-backscatter), restricted to a small search window around the expected target range.
       The ADCP pings produce a pulse train, To obtain an accurate representation of the sphere,
       the maximum intensity is averaged over a depth window equal to the pulse train length.

       Method:
       1. Collapse intensity data into a single range profile using a max over pings.
       2. Apply a moving-average filter with window size proportional to pulse train length.
       3. Search for the maximum value within a ±0.5 m window around the expected range.
       4. Return the detected peak position and associated intensity values.

       Parameters
       ----------
       intensity : np.ndarray
       range_estimate : float
           Expected target depth (m), used to center the search window.
       cell_size : float
           Vertical resolution per range cell (m).
       pulse_length_m : float
           Acoustic pulse length expressed in meters; used to define smoothing window.

       Returns
       -------
       peak_depth : float or None
           Depth (m) of detected intensity maximum. Returns None if no valid window.
       peak_intensity : float or None
           Smoothed intensity value at peak location.
       max_profile : np.ndarray
           Raw maximum intensity profile (collapsed over pings/beams).
       smooth : np.ndarray
           Smoothed intensity profile used for peak detection.
       """
    max_profile = np.max(intensity, axis=0)

    window_cells = max(3, int(pulse_length_m / cell_size))
    kernel = np.ones(window_cells) / window_cells
    smooth = np.convolve(max_profile, kernel, mode="same")

    start = max(0, int((range_estimate - 0.5) / cell_size))
    end = min(len(smooth), int((range_estimate + 0.5) / cell_size))

    local = smooth[start:end]
    if len(local) == 0:
        return None, None, max_profile, smooth

    peak_idx = start + np.argmax(local)
    return peak_idx * cell_size, smooth[peak_idx], max_profile, smooth


def append_csv(path, rows):
    """
        Append calibration rows to a CSV file.

        Creates the folder if needed and writes the header.
        """
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_COLS.keys())
        writer.writerows(rows)


def compute_global_max_intensity(decoder, windows, is_adcp):
    """
       Compute global maximum intensity across calibration windows.

       Method:
           1. Loop through each calibration window.
           2. Extract intensity data for the window time range.
           3. Compute maximum intensity per window.
           4. Return the highest value across all windows.

       Parameters
       ----------
       windows : list[Window]
           Calibration windows

       Returns
       -------
       float
           Global maximum intensity across all windows.
       """
    max_val = -np.inf

    for w in windows:

        decoder.start_time = w.start_time + timedelta(hours=1)
        decoder.end_time = w.end_time + timedelta(hours=1)

        if is_adcp:
            data = decoder.extract(["intensity", "time", "temperature"])
        else:
            data = decoder.extract(["intensity"])

        intensity = data["intensity"][:, w.beam_index, :]
        window_max = np.max(intensity)

        if window_max > max_val:
            max_val = window_max

    return float(max_val)


def process_unit(name, decoder, windows, output_folder, is_adcp=False, global_max=None):
    """
       Process calibration windows and generate intensity profiles and CSV output.

       Method:
           1. Loop through calibration windows.
           2. Extract intensity data for each time range.
           3. Locate sphere peak (smoothed for ADCP, max for ECHO).
           4. Store results in a structured CSV row.
           5. Generate and save per-window intensity plots.

       Parameters
       ----------
       name : str
           Name of the dataset (used for plot titles and data identification).
       decoder : GramDecoder
           B-gram decoder used to extract intensity data.
       windows : list[Window]
           Calibration windows defining time, depth, and beam index.
       output_folder : pathlib.Path
           Directory where plots and CSV output will be saved.
       is_adcp : bool, optional
           If True, applies ADCP-specific processing (default False).
       global_max : float or None
           Global maximum intensity used for max saturation intensity.

       Returns
       -------
       None
       """
    rows = []

    for i, w in enumerate(windows):

        decoder.start_time = w.start_time + timedelta(hours=1)
        decoder.end_time = w.end_time + timedelta(hours=1)

        data = decoder.extract(["intensity", "time", "temperature"] if is_adcp else ["intensity"])

        intensity = data["intensity"][:, w.beam_index, :]
        cell_size = cell_length(decoder.first_ping)

        if is_adcp:
            bt = decoder.first_ping.sCommon.u16_SIG_BT
            n_reps = decoder.first_ping.sCommon.u16_SIG_BaseCounts
            bandwidth = decoder.first_ping.sCommon.u32_TxBandwidthHz
            pulse_length_m = (c * bt * n_reps / bandwidth) / 2

            depth, intensity_peak, max_profile, smooth = locate_intensity_peak_moving_avg(
                intensity, w.depth, cell_size, pulse_length_m
            )
        else:
            depth, intensity_peak, max_profile = locate_intensity_peak(
                intensity, w.depth, cell_size
            )

        rows.append([
            i,
            SUSPENSION,
            UNIT,
            SPHERE_SIZE,
            PULSE_LENGTH_US,
            -depth,
            intensity_peak,
            None,
            None,
            GAIN,
            (NOAA_TS_ADCP if is_adcp else NOAA_TS_ECHO),
            global_max
        ])

        y = np.arange(len(max_profile)) * cell_size

        plt.figure(figsize=(6, 8))
        plt.plot(max_profile, y, 'k')

        if is_adcp:
            plt.plot(smooth, y, 'r', alpha=0.7)

        plt.axhline(depth, color='red', linestyle='--')
        plt.gca().invert_yaxis()

        plt.title(f"{name} window {i}")
        plt.xlabel("Intensity")
        plt.ylabel("Depth (m)")

        plt.savefig(output_folder / f"window_{i:03d}.png", dpi=150)
        plt.close()

    append_csv(output_folder / "calibration.csv", rows)


PLOT_DATA = []


def load_csv(filename):
    """
    Load calibration CSV file into a NumPy array.

    Returns a 2D array even when the file contains a single row.
    """
    data = np.genfromtxt(filename,
                         delimiter=",",
                         skip_header=1)

    return data.reshape(1, -1) if data.ndim == 1 else data


def unitless_calibration_factor(intensity, depth, absorb, noaa_ts, gain):
    """
      Compute unitless backscatter calibration factor.

      Method:
          Applies calibration corrections to convert raw intensity
          into The calibration factor.

      Parameters
      ----------
      intensity : float or np.ndarray
          Measured backscatter intensity.
      depth : float or np.ndarray
          Target depth (m).
      absorb : float
         Absorption coefficient. (currently hard coded for user configuration)
      noaa_ts : float
          NOAA theoretical target strength (dB).(currently hard coded for user configuration)
      gain : float
          Instrument gain (dB).

      Returns
      -------
      float or np.ndarray
          Unitless calibration factor.
      """
    return (
            20 * np.log10(intensity)
            + 40 * np.log10(np.abs(depth))
            + 2 * absorb * np.abs(depth)
            - noaa_ts
            + gain
    )


def saturated_curve(depth_array, true_unitless, absorb, noaa_ts, gain, sat_intensity):
    """
       Apply saturation value to unitless calibration curve.

       Method:
           Computes the calibration curve using the max intensity recorded,
           plotting the calibration factor, as it would be at saturation.

       Parameters
       ----------
       depth_array : np.ndarray
           Array of depths (m) over which the curve is evaluated. (can be changed for deeper calibrations)
       true_unitless : float or np.ndarray
           Fitted unitless calibration value.
      absorb : float
         Absorption coefficient. (currently hard coded for user configuration)
      noaa_ts : float
          NOAA theoretical target strength (dB).(currently hard coded for user configuration)
      gain : float
          Instrument gain (dB).
       sat_intensity : float
           Max intensity recorded, value at saturation.

       Returns
       -------
       np.ndarray
           Saturated value calibration curve.
       """
    return np.minimum(
        true_unitless,
        unitless_calibration_factor(sat_intensity,
                             depth_array,
                             absorb,
                             noaa_ts,
                             gain),
    )


def calibrate_sphere_unitless(depth, intensity, absorb, noaa_ts, gain, sat_intensity):
    """
      Fit unitless calibration factor for the recorded sphere intensity
      against the saturated model.

      Method:
          Converts raw intensity into unitless calibration factors.
          Defines a saturation model of expected factor at saturation.
          Fit model to observed data using non-linear least squares.
          Return best-fit calibration factor and uncertainty.

      Parameters
      ----------
      depth : np.ndarray
          Depth values (m) of sphere.
      intensity : np.ndarray
          Measured backscatter sphere intensity.
      absorb : float
         Absorption coefficient. (currently hard coded for user configuration)
      noaa_ts : float
          NOAA theoretical target strength (dB).(currently hard coded for user configuration)
      gain : float
          Instrument gain (dB).
       sat_intensity : float
           Max intensity recorded, value at saturation.

      Returns
      -------
      fitted_value : float
          Best-fit unitless calibration factor.
      fitted_std : float
          Standard deviation (1σ) from covariance matrix.
      """
    def cutoff(depth, true_unitless):
        return saturated_curve(depth,
                               true_unitless,
                               absorb,
                               noaa_ts,
                               gain,
                               sat_intensity)

    unitless_points = unitless_calibration_factor(intensity, depth, absorb, noaa_ts, gain)

    popt, pcov = curve_fit(cutoff, depth, unitless_points)

    return popt[0], np.sqrt(pcov[0, 0])


def process_calibration_file(csv_file,label,colour,marker="o"):
    """
       Process calibration CSV and store results for plotting and summary output.

       Method:
           1. Load calibration CSV file.
           2. Extract intensity, depth, and instrument metadata.
           3. Compute unitless calibration values.
           4. Fit saturation calibration model.
           5. Compute residuals and summary statistics.
           6. Save summary text file.
           7. Store results for final plotting stage.

       Returns
       -------
       None
       """
    data = load_csv(csv_file)

    csv_path = Path(csv_file)
    out_dir = csv_path.parent

    intensity = data[:, CSV_COLS["intensity"]]
    depth = data[:, CSV_COLS["depth"]]

    gain = data[0, CSV_COLS["gain"]]
    noaa_ts = data[0, CSV_COLS["noaa_ts"]]
    sat_intensity = (data[0, CSV_COLS["max_recorded_intensity"]])

    unitless = unitless_calibration_factor(
        intensity,
        depth,
        ABSORPTION_COEFFICIENT,
        noaa_ts,
        gain,
    )

    fitted_unitless, fit_std = calibrate_sphere_unitless(
        depth,
        intensity,
        ABSORPTION_COEFFICIENT,
        noaa_ts,
        gain,
        sat_intensity,
    )

    unitless_at_saturation = saturated_curve(
        depths_m,
        fitted_unitless,
        ABSORPTION_COEFFICIENT,
        noaa_ts,
        gain,
        sat_intensity,
    )

    saturation_curve_only = unitless_calibration_factor(
        sat_intensity,
        depths_m,
        ABSORPTION_COEFFICIENT,
        noaa_ts,
        gain,
    )

    residuals = (
            unitless
            - unitless_calibration_factor(
        sat_intensity,
        depth,
        ABSORPTION_COEFFICIENT,
        noaa_ts,
        gain,
    )
    )

    std_unitless = np.std(unitless, ddof=1)

    summary_text = (
        "\n===================================\n"
        "Calibration Summary\n"
        "===================================\n"
        f"File: {csv_file}\n"
        f"Calibration factor: {fitted_unitless:.1f} ± {std_unitless:.1f} dB\n"
        f"Number of points:     {len(depth)}\n"
        f"Gain:                 {gain:.1f} dB\n"
        f"NOAA TS:              {noaa_ts:.2f} dB\n"
        "===================================\n"
    )

    print(summary_text)

    summary_file = out_dir / "calibration_summary.txt"
    with open(summary_file, "w") as f:
        f.write(summary_text)

    PLOT_DATA.append(
        {
            "label": label,
            "colour": colour,
            "marker": marker,
            "depth": depth,
            "unitless": unitless,
            "fitted_unitless": fitted_unitless,
            "fit_std": fit_std,
            "std_unitless": std_unitless,
            "unitless_at_saturation": unitless_at_saturation,
            "unitless_at_saturation_curve": saturation_curve_only,
            "residuals": residuals,
        }
    )


def plot_all_gains_with_coloured_deviation_fixed(out_dir, k_sigma=2):
    """
       Plot calibration results and residual deviations across datasets.

       Method:
           1. Plot unitless calibration curves for all datasets.
           2. Overlay measured calibration points and fitted models.
           3. Compute residuals from saturation model.
           4. Highlight deviations beyond ksigma threshold.
           5. Save combined comparison figure to output folder.

       Returns
       -------
       None
       """
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        sharey=True,
    )

    for data in PLOT_DATA:

        ax1.plot(
            data["unitless_at_saturation"],
            depths_m,
            color=data["colour"],
            alpha=0.8,
        )

        ax1.scatter(
            data["unitless"],
            data["depth"],
            color=data["colour"],
            marker=data["marker"],
            label=data["label"],
        )

        ax1.plot(
            data["unitless_at_saturation_curve"],
            depths_m,
            "--",
            color="black",
            alpha=0.5,
        )

        ax1.text(
            data["fitted_unitless"],
            -14,
            f"{data['fitted_unitless']:.1f} ± {data['std_unitless']:.1f} dB",
            ha="center",
            fontsize=11,
        )

        ax2.scatter(
            data["residuals"],
            data["depth"],
            color=data["colour"],
            marker=data["marker"],
        )

        ax2.axvline(
            0,
            linestyle="--",
            color="black",
            alpha=0.7,
        )

        threshold = k_sigma * data["std_unitless"]

        deviated_idx = np.where(
            np.abs(data["residuals"]) > threshold
        )[0]

        if len(deviated_idx) > 0:
            first_depth = data["depth"][deviated_idx[0]]

            ax1.axhline(
                first_depth,
                color=data["colour"],
                linestyle="--",
                alpha=0.8,
            )

            ax2.axhline(
                first_depth,
                color=data["colour"],
                linestyle="--",
                alpha=0.8,
            )

    ax1.set_xlabel("Calibration Factor (dB)")
    ax1.set_ylabel("Depth (m)")
    ax1.set_title("Calibration Depth Sweep")

    ax1.invert_yaxis()
    ax1.set_ylim(0, -26)

    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel("Residual (dB)")
    ax2.set_title("Deviation from Saturation Curve")

    ax2.invert_yaxis()
    ax2.grid(True)

    plt.tight_layout()

    plot_file = out_dir / "calibration_plot.png"
    plt.savefig(plot_file, dpi=200, bbox_inches="tight")

    plt.close()


def run_pipeline(name, bgram_path, label, colour, is_adcp):
    """
       Execute full calibration pipeline for a single instrument.

       Method:
           1. Initialise output directory and decoder.
           2. Compute global intensity maximum across windows.
           3. Process calibration windows and generate CSV output.
           4. Fit calibration model and generate summary plots.

       Returns
       -------
       None
       """
    global PLOT_DATA
    PLOT_DATA = []

    out_dir = OUTPUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    decoder = GramDecoder(str(bgram_path))

    global_max = compute_global_max_intensity(decoder, schedule, is_adcp)

    process_unit(
        name,
        decoder,
        schedule,
        out_dir,
        is_adcp=is_adcp,
        global_max=global_max
    )

    csv_file = out_dir / "calibration.csv"

    process_calibration_file(csv_file, label, colour)
    plot_all_gains_with_coloured_deviation_fixed(out_dir)


def main():
    run_pipeline("ADCP", ADCP_BGRAM_PATH, PLOT_LABEL_ADCP, plot_colour_adcp, True)
    run_pipeline("ECHO", ECHO_BGRAM_PATH, PLOT_LABEL_ECHO, plot_colour_echo, False)


if __name__ == "__main__":
    main()
