import functools
import time
from scipy import interpolate
import numpy as np

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__}() in {run_time:.6f} secs")
        return value
    return wrapper_timer

def flux_sampling(time_array,df_FD,maxT):
    interpolated_flux = np.zeros_like(time_array)
    time_array_length = len(time_array)
    df_FD_forward = df_FD.iloc[:int(len(df_FD)/2)]
    df_FD_reverse = df_FD.iloc[int(len(df_FD)/2):]
    forward_scan_time_array = time_array[:int(time_array_length/2)]
    reverse_scan_time_array  = time_array[int(time_array_length/2):]
    f_forward = interpolate.interp1d(np.linspace(0.0,maxT/2.0,num=len(df_FD_forward)),df_FD_forward.iloc[:,1],fill_value="extrapolate")
    f_reverse = interpolate.interp1d(np.linspace(maxT/2.0,maxT,num=len(df_FD_reverse)),df_FD_reverse.iloc[:,1],fill_value="extrapolate")
    interpolated_flux[:int(time_array_length/2)] = f_forward(forward_scan_time_array)
    interpolated_flux[int(time_array_length/2):] = f_reverse(reverse_scan_time_array)
    return interpolated_flux



def exp_flux_sampling(time_array,df_exp,FullScanT,PortionAnalyzed=0.75):
    interpolated_flux = np.zeros_like(time_array)
    time_array_length = len(time_array)
    df_exp_forward = df_exp.iloc[:int(len(df_exp)*0.5)]
    df_exp_reverse = df_exp.iloc[int(len(df_exp)*0.5):int(len(df_exp)*PortionAnalyzed)]
    forward_scan_time_array = time_array[time_array<FullScanT*0.5]
    reverse_scan_time_array  = time_array[time_array>=FullScanT*0.5]
    assert time_array_length == len(forward_scan_time_array) + len(reverse_scan_time_array)
    f_forward = interpolate.interp1d(np.linspace(0,FullScanT*0.5,num=len(df_exp_forward)),df_exp_forward.iloc[:,1])
    f_reverse = interpolate.interp1d(np.linspace(FullScanT*0.5,FullScanT*PortionAnalyzed,num=len(df_exp_reverse)),df_exp_reverse.iloc[:,1])
    interpolated_flux[:len(forward_scan_time_array)] = f_forward(forward_scan_time_array)
    interpolated_flux[len(forward_scan_time_array):] = f_reverse(reverse_scan_time_array)
    return interpolated_flux
