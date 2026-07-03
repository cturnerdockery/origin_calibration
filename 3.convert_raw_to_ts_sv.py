
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from origin_tools.decoder import GramDecoder
from origin_tools.geometry import cell_length


ADCP_BGRAM_PATH = Path(r"E:\path\to\A\bgram")
ECHO_BGRAM_PATH = Path(r"E:\path\to\B\bgram")


# Pick ping type and method

PING_TYPE = "echo"      # "adcp" or "echo"
METHOD = "ts"          # "ts" or "sv"

#Set constants, from: 2.calibration_main

INSTRUMENTS = {
    "adcp": {
        "calibration_factor": 193.1,
        "noaa_ts": -51.30,
    },
    "echo": {
        "calibration_factor": 193.1,
        "noaa_ts": -51.46,
    },
}

ABSORPTION_COEFFICIENT = 0.183
SPEED_OF_SOUND = 1500



def get_bgram_path(ping_type):
    """
    Return the appropriate B-gram path for the selected instrument.
    """
    ping_type = ping_type.lower()

    if ping_type == "adcp":
        return ADCP_BGRAM_PATH

    if ping_type == "echo":
        return ECHO_BGRAM_PATH

    raise ValueError(
        f"Unknown ping type: {ping_type}"
    )

def convert_intensity_to_ts(intensity, range, absorb, gain, calibration_factor):
    """
        Convert measured acoustic unitless intensity to target strength (TS).

        Calculates target strength in decibels (dB) using the measured
        intensity and applying corrections for transmission loss,
        absorption, instrument gain, and calibration offset.

        Parameters
        ----------
        intensity : float or ndarray
            Measured unitless intensity .
        range : float or ndarray
            Distance from the transducer to the target (m).
        absorb : float
         Absorption coefficient. (currently hard coded for user configuration)
        gain : float
          Instrument gain (dB).
        calibration_factor : float
            Calibration offset (dB).

        Returns
        -------
        float or ndarray
            Target strength (TS) in decibels (dB).

        Notes
        -----
        The calculation is:

            TS = 20 log10(U) + 40 log10(r) + 2 α r - g - CF

        where:
        - U = measured unitless intensity,
        - r = range,
        - α = absorption coefficient,
        - g = instrument gain,
        - CF = calibration factor.
        """

    return (  20 * np.log10(intensity)
            + 40 * np.log10(abs(range))
            + 2 * absorb * abs(range)
            - gain
            - calibration_factor)

def convert_intensity_to_sv(intensity, range, absorb,pulse_length, gain, calibration_factor, c):
    """
    Convert measured acoustic intensity to volume backscattering strength (Sv).

    Calculates Sv in decibels (dB) from measured acoustic intensity,
    applying corrections for transmission loss, absorption, pulse length,
    equivalent beam angle, instrument gain, and the calculated calibration offset.

    Parameters
    ----------
    intensity : float or ndarray
        Measured unitless intensity.
    range : float or ndarray
        Distance from the transducer to the sampled volume (m).
    absorb : float
         Absorption coefficient. (currently hard codes for user configuration)
    pulse_length : float
        Transmitted pulse length, in seconds.
    gain : float
        Instrument gain (dB).
    calibration_factor : float
        Calibration offset (dB).
    c : float
        Speed of sound in water (m/s).

    Returns
    -------
    float or ndarray
        Volume backscattering strength (Sv) in decibels (dB).

    Notes
    -----
    The function creates:
    - A corrected pulse-length.
    - An equivalent beam angle based on the wavenumber and transducer radius.
    - A corrected range.

    The equivalent beam angle is calculated as:
        equivalent_beam_angle = 5.78 / (ka)^2

    where:

    - k = (2πfr) / c
    - f = 625 kHz (fixed operating frequency)
    - a = 0.088 m (transducer radius)
    - r is range
    - c = speed of sound in water

    The Sv calculation is:

        Sv = 20 log10(U) + 40 log10(r_c) + 2 α r_c - τ_c - ψ + G - C

    where:
    - U = measured unitless intensity,
    - r_c = corrected range,
    - α = absorption coefficient,
    - τ_c = corrected pulse length,
    - ψ = equivalent beam angle correction,
    - G = system gain,
    - C = calibration factor.
    """
    pulse_length_warmup = 2.5e-6
    k = (2 * np.pi * 625000 * range)/c
    a = 0.088
    ka = (k * a)
    ka2 = ka**2
    ka2 = np.where(ka2 == 0, np.nan, ka2)
    equivalent_beam_angle = 5.78/ka2

    corrected_range = range - ((c * pulse_length)/4)
    corrected_pulse_length = pulse_length - pulse_length_warmup


    return (20 * np.log10(intensity)
           + 20 * np.log10(abs(corrected_range))
           + 2 * absorb * abs(corrected_range)
           - np.log10(corrected_pulse_length)
           - equivalent_beam_angle
           - gain
           - calibration_factor)

def determine_gain(rx_gain, tx_level):
    if rx_gain == 30 and tx_level == 1:
        return -13
    if rx_gain == 30 and tx_level == 2:
        return 0
    if rx_gain == 40 and tx_level == 2:
        return 0
    if rx_gain == 20 and tx_level == 2:
        return -20

    raise ValueError("Gain not found")

def calibrate(intensity,range,method,absorb,gain,calibration_factor,pulse_length=None,c=1500):

    method = method.lower()

    if method == "ts":
        return convert_intensity_to_ts(
            intensity,
            range,
            absorb,
            gain,
            calibration_factor,
        )

    if method == "sv":
        return convert_intensity_to_sv(
            intensity,
            range,
            absorb,
            pulse_length,
            gain,
            calibration_factor,
            c,
        )

    raise ValueError(f"Unknown method: {method}")


def plot_bgram(data, time, first_ping, metric, ping_type, c=1500):

    metric = metric.lower()

    settings = {
        "ts": {
            "clim": (-150, -0),
            "title": f"{ping_type.upper()} Target Strength",
            "label": "Target Strength (dB re 1 m$^2$ @ 1 m)",
        },
        "sv": {
            "clim": (-150, -0),
            "title": f"{ping_type.upper()} Volume Backscatter",
            "label": "S$_v$ (dB re 1 m$^{2}$ m$^{-3}$)",
        },
    }

    cfg = settings[metric]
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(
        data.T,
        cmap="magma",
        vmin=cfg["clim"][0],
        vmax=cfg["clim"][1],
        origin="lower",
        aspect="auto",
        interpolation ="none",
        extent=[
            time[0] / 86400,
            time[-1] / 86400,
            0,
            data.shape[1] * cell_length(first_ping, c),
        ],
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cfg["label"], fontsize=12)

    ax.set_xlabel("Time (BST)")
    ax.set_ylabel("Slant range (m)")
    ax.set_title(cfg["title"])

# Change this for cut off depths,
    # ax.set_ylim(0, 80)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    plt.tight_layout()
    plt.show()


BGRAM_PATH = get_bgram_path(PING_TYPE)

print(f"Ping Type: {PING_TYPE}")
print(f"Method: {METHOD}")

gd = GramDecoder(BGRAM_PATH)

data = gd.extract(["intensity", "time"],max_length=5000)

n_reps = gd.first_ping.sCommon.u16_SIG_BaseCounts
pulse_length = n_reps * 1e-4

tx_level = gd.first_ping.sCommon.u16_TxLevel
rx_gain = gd.first_ping.sCommon.u16_RxGaindB
gain = determine_gain(rx_gain, tx_level)


intensity = data["intensity"][:, 4, :]
time = data["time"]

vertical_range = (cell_length(gd.first_ping)* np.arange(intensity.shape[1]))

calibrated = calibrate(
    intensity,
    vertical_range,
    method=METHOD,
    absorb=ABSORPTION_COEFFICIENT,
    gain=gain,
    calibration_factor=INSTRUMENTS[PING_TYPE]["calibration_factor"],
    pulse_length=pulse_length,
    c=SPEED_OF_SOUND,
)

# calibrated = np.nan_to_num(calibrated,nan=0.0,posinf=0.0,neginf=0.0)

plot_bgram(
    calibrated,
    time,
    gd.first_ping,
    metric=METHOD,
    ping_type=PING_TYPE,
    c=SPEED_OF_SOUND)