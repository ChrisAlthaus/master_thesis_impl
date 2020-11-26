import numpy as np
import ipyplot

def recursive_print_dict( d, indent = 0 ):
    if not isinstance(d, list) and not isinstance(d, dict):
        print("\t" * (indent+1), d)
        return
        
    for k, v in d.items():
        if isinstance(v, dict):
            print("\t" * indent, f"{k}:")
            recursive_print_dict(v, indent+1)
        elif isinstance(v, list):
            print("\t" * indent, f"{k}")
            allsingle = True
            for item in v:
                if isinstance(item, list) or isinstance(item, dict):
                    allsingle = False
            if allsingle:
                print("\t" * indent, v)
            else:
                for item in v:
                    recursive_print_dict(item, indent)
        else:
            print("\t" * indent, f"{k}:{v}")

def logscorestats(scores):
    #Computes the box-plot whiskers values.
    #Computed values: min, low_whiskers, Q1, median, Q3, high_whiskers, max
    #Ordering differs from whiskers plot ordering.
    Q1, median, Q3 = np.percentile(np.asarray(scores), [25, 50, 75])
    IQR = Q3 - Q1

    loval = Q1 - 1.5 * IQR
    hival = Q3 + 1.5 * IQR

    wiskhi = np.compress(scores <= hival, scores)
    wisklo = np.compress(scores >= loval, scores)
    actual_hival = np.max(wiskhi)
    actual_loval = np.min(wisklo)

    Qs = [Q1, median, Q3, loval, hival, actual_loval, actual_hival]
    Qname = ["Q1", "median", "Q3", "Q1-1.5xIQR", "Q3+1.5xIQR", 
            "Actual LO", "Actual HI"]
    logstr = ''.join(["{}:{} ".format(a,b) for a,b in zip(Qname,Qs)])
    return logstr