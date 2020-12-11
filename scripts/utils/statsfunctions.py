
import numpy as np

def getwhiskersvalues(values):
    #Computes the box-plot whiskers values.
    #Computed values: min, low_whiskers, Q1, median, Q3, high_whiskers, max
    #Ordering differs from whiskers plot ordering.
    Q1, median, Q3 = np.percentile(np.asarray(values), [25, 50, 75])
    IQR = Q3 - Q1

    loval = Q1 - 1.5 * IQR
    hival = Q3 + 1.5 * IQR

    wiskhi = np.compress(values <= hival, values)
    wisklo = np.compress(values >= loval, values)
    actual_hival = np.max(wiskhi)
    actual_loval = np.min(wisklo)

    minv = np.min(values)
    maxv = np.max(values)

    Qs = [minv, actual_loval, loval, Q1, median, Q3, hival, actual_hival, maxv]
    Qname = ["Min", "Actual LO", "Q1-1.5xIQR", "Q1", "median", "Q3", "Q3+1.5xIQR", 
             "Actual HI", "Max"]
    logstr = ''.join(["\t{}:{} \n".format(a,b) for a,b in zip(Qname,Qs)])
    return logstr