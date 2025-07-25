
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import pandas as pd

def simulate_emri(duration=20000.0, fs=2.0):
    t = np.arange(0, duration, 1/fs)
    f0, a, b = 0.003, 2.0e-7, 0.0
    f = f0 + a*t + b*t**2
    glitches = [
        {"t": 7000.0, "df": 2.5e-4, "d_slope": 5.0e-8, "width": 1200.0},
        {"t": 15000.0, "df": -1.5e-4, "d_slope": -3.0e-8, "width": 1500.0},
    ]
    f_mod = f.copy()
    for g in glitches:
        mask_after = t >= g["t"]
        f_mod[mask_after] += g["df"]
        mask_band = (t >= g["t"]) & (t < g["t"] + g["width"])
        f_mod[mask_band] += g["d_slope"] * (t[mask_band] - g["t"])
    phi = 2*np.pi * np.cumsum(f_mod) / fs
    A = (1 + 0.00005*t) / np.sqrt(1 + t/20000.0)
    rng = np.random.default_rng(0)
    h = A*np.cos(phi) + 0.5*rng.normal(size=t.size)
    return t, h, f_mod

def ridge_and_detect(h, fs=2.0, fmax=0.05, nperseg=1024, overlap=0.9, zthr=2.5):
    noverlap = int(overlap*nperseg)
    f_bins, t_bins, Sxx = spectrogram(h, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling='density', mode='magnitude')
    valid = f_bins <= fmax
    Ssub = Sxx[valid, :]
    fsub = f_bins[valid]
    ridge_idx = np.argmax(Ssub, axis=0)
    ridge_f = fsub[ridge_idx]
    ridge_t = t_bins
    # slopes via local linear regression
    win = 15
    slopes = np.full_like(ridge_f, np.nan, dtype=float)
    for i in range(len(ridge_f)):
        i0 = max(0, i-win); i1 = min(len(ridge_f), i+win+1)
        x = ridge_t[i0:i1]; y = ridge_f[i0:i1]
        if len(x) > 4:
            X = np.vstack([x - x.mean(), np.ones_like(x)]).T
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            slopes[i] = beta[0]
    mu, sd = np.nanmean(slopes), np.nanstd(slopes)
    z = (slopes - mu) / sd
    cand = np.where(np.abs(z) > zthr)[0]
    events = []
    if cand.size:
        s = cand[0]; p = cand[0]
        for idx in cand[1:]:
            if idx == p+1:
                p = idx
            else:
                events.append((s, p)); s = idx; p = idx
        events.append((s, p))
    detected = [ridge_t[(s+e)//2] for s,e in events]
    return (f_bins, t_bins, Sxx), (ridge_t, ridge_f, slopes, z), detected

def main():
    t, h, f_model = simulate_emri()
    (f_bins, t_bins, Sxx), (rt, rf, slopes, z), detected = ridge_and_detect(h)
    # save csvs
    pd.DataFrame({"t_s": rt, "f_Hz": rf, "slope": slopes, "z_slope": z}).to_csv("EMRI_ridge_and_detection.csv", index=False)
    pd.Series(detected, name="t_detected_s").to_csv("EMRI_detected_glitch_times.csv", index=False)
    pd.DataFrame({"t_s": t, "f_model_Hz": f_model}).to_csv("EMRI_model_frequency.csv", index=False)
    # plots
    plt.figure()
    plt.pcolormesh(t_bins, f_bins, Sxx, shading='auto')
    plt.plot(rt, rf, linewidth=1.0)
    for tt in detected:
        plt.axvline(tt, linestyle='--')
    plt.ylim(0, 0.02); plt.xlabel("Time [s]"); plt.ylabel("Frequency [Hz]")
    plt.title("EMRI spectrogram with ridge and detected glitches")
    plt.colorbar(label="|S|"); plt.tight_layout()
    plt.savefig("emri_spectrogram_ridge_detected.png", dpi=180)
    print("Detected glitch times (s):", detected)

if __name__ == "__main__":
    main()
