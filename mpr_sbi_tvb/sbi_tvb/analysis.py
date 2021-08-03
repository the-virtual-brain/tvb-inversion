import numpy as np


def compute_fcd(ts, win_len=30, win_sp=1):
    """Compute dynamic functional connectivity.

    Arguments:
        ts:      time series of shape [time,nodes]
        win_len: sliding window length in samples
        win_sp:  sliding window step in samples

    Returns:
        FCD: matrix of functional connectivity dynamics
        fcs: windowed functional connectivity matrices
        speed_fcd: rate of changes between FC frames
    """
    n_samples, n_nodes = ts.shape 
    fc_triu_ids = np.triu_indices(n_nodes, 1) #returns the indices for upper triangle
    n_fcd = len(fc_triu_ids[0]) #
    fc_stack    = []
    speed_stack = []
    fc_prev = []


    for t0 in range( 0, ts.shape[0]-win_len, win_sp ):
        t1=t0+win_len
        fc = np.corrcoef(ts[t0:t1,:].T)
        fc = fc[fc_triu_ids]
        fc_stack.append(fc)
        if t0 > 0 :
            corr_fcd  = np.corrcoef([fc,fc_prev])[0,1]
            speed_fcd = 1-corr_fcd
            speed_stack.append(speed_fcd)
            fc_prev   = fc
        else:
            fc_prev   = fc
            

    fcs      = np.array(fc_stack)
    speed_ts = np.array(speed_stack)
    FCD = np.corrcoef(fcs)
    return FCD, fcs, speed_ts

def compute_fcd_filt(ts, mat_filt, win_len=30, win_sp=1):
    """Compute dynamic functional connectivity with SC filtering

    Arguments:
        ts:      time series of shape [time,nodes]
        win_len: sliding window length in samples
        win_sp:  sliding window step in samples

    Returns:
        FCD: matrix of functional connectivity dynamics
        fcs: windowed functional connectivity matrices
        speed_fcd: rate of changes between FC frames
    """
    n_samples, n_nodes = ts.shape 
    fc_triu_ids = np.triu_indices(n_nodes, 1) #returns the indices for upper triangle
    n_fcd = len(fc_triu_ids[0]) #
    fc_stack    = []
    speed_stack = []


    for t0 in range( 0, ts.shape[0]-win_len, win_sp ):
        t1=t0+win_len
        fc = np.corrcoef(ts[t0:t1,:].T)
        fc = fc*(fc>0)*(mat_filt)
        fc = fc[fc_triu_ids]
        fc_stack.append(fc)
        if t0 > 0 :
            corr_fcd  = np.corrcoef([fc,fc_prev])[0,1]
            speed_fcd = 1-corr_fcd
            speed_stack.append(speed_fcd)
            fc_prev   = fc
        else:
            fc_prev   = fc
            

    fcs      = np.array(fc_stack)
    speed_ts = np.array(speed_stack)
    FCD = np.corrcoef(fcs)
    return FCD, fcs, speed_ts

