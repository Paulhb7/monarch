import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from gwpy.timeseries import TimeSeries
from gwosc import datasets

def qnm(t,A1,f1,tau1,phi1,A2=0.0,f2=300.0,tau2=0.01,phi2=0.0):
    return A1*np.exp(-t/tau1)*np.cos(2*np.pi*f1*t+phi1)+A2*np.exp(-t/tau2)*np.cos(2*np.pi*f2*t+phi2)

def aic(y,yhat,k):
    r=y-yhat
    sse=np.sum(r**2)
    n=y.size
    return n*np.log(sse/n)+2*k

def fetch(event,dets=("H1","L1"),pad=32,fs=16384):
    gps=datasets.event_gps(event)
    s={}
    for d in dets:
        s[d]=TimeSeries.fetch_open_data(d,gps-pad,gps+pad,sample_rate=fs)
    return gps,s

def clean(ts,flow=20,fhigh=2048,seg=8):
    return ts.bandpass(flow,fhigh).whiten(seg,seg)

def spectro(ts,gps,title):
    sp=ts.spectrogram2(fftlength=0.25,overlap=0.20).crop(gps-1.0,gps+0.5)
    ax=sp.plot(norm="log"); ax.set_title(title); plt.show()

def ringdown(ts,after_peak=0.004,duration=0.06):
    t=ts.times.value; y=ts.value
    tp=t[np.argmax(np.abs(y))]
    m=(t>=tp+after_peak)&(t<=tp+after_peak+duration)
    return t[m]-tp,y[m]

def fit(tr,yr):
    p1=[np.std(yr),200.,0.01,0.0]
    popt1,_=curve_fit(lambda tt,A1,f1,tau1,phi1:qnm(tt,A1,f1,tau1,phi1,0,0,1,0),tr,yr,p0=p1,maxfev=40000)
    y1=qnm(tr,popt1[0],popt1[1],popt1[2],popt1[3],0,0,1,0)
    A1=aic(yr,y1,4)
    p2=[popt1[0],popt1[1],popt1[2],popt1[3],0.3*np.std(yr),350.,0.006,0.0]
    popt2,_=curve_fit(qnm,tr,yr,p0=p2,maxfev=60000)
    y2=qnm(tr,*popt2)
    A2=aic(yr,y2,8)
    return (popt1,A1,y1),(popt2,A2,y2)

def main():
    event="GW231123_135430"
    gps,strain=fetch(event)
    det="H1" if "H1" in strain else list(strain.keys())[0]
    ts=clean(strain[det])
    spectro(ts,gps,f"{det} spectrogram around {event}")
    tr,yr=ringdown(ts)
    (p1,A1,y1),(p2,A2,y2)=fit(tr,yr)
    print("1-mode:",p1,"AIC=",A1)
    print("2-modes:",p2,"AIC=",A2,"ΔAIC=",A2-A1)
    plt.plot(tr*1e3,yr,label="data")
    plt.plot(tr*1e3,y1,label="QNM1")
    plt.plot(tr*1e3,y2,label="QNM2")
    plt.xlabel("t - t_peak [ms]"); plt.legend(); plt.title(f"Ringdown fits {det}"); plt.tight_layout(); plt.savefig("gw231123_ringdown_fits.png",dpi=180); plt.show()

if __name__=="__main__":
    main()

#analyse ringdown GW231123 et spectrogramme — by Paul Barbaste.

