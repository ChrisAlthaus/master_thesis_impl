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


def checkbyrecalculate(resultlist, docs, method , featurevector, maskstr):
    #Check results given by elasticsearch by calculations localy
    from scipy.spatial import distance
    from numpy import dot
    from numpy.linalg import norm
    vecs1 = [item['_source']['gpd-array'] for item in docs]
    ms1 = [item['_source']['mask'] for item in docs]
    v1dbscores = [item['_source']['score'] for item in docs]
    #vecs2 = [item['_source']['tempvec'] for item in docs]
    vecs2 = [featurevector for item in docs]
    ms2 = [maskstr for item in docs]
    #print('db vectors:', vecs1)
    #print("query vectors:", vecs2)
    #print('masks 1      :', ms1)
    #print('masks 1      :', ms2)
    penalties = []
    penalty = 0.5/len(vecs1[0])
    
    for i in range(len(vecs1)):
        vecs2[i] = [vecs1[i][n] if (ms1[i][n] == '0' or ms2[i][n] == '0') else vecs2[i][n] for n in range(len(vecs1[i]))]
        c = sum( [penalty if (ms1[i][n] == '0' or ms2[i][n] == '0') else 0 for n in range(len(vecs1[i]))] )
        penalties.append(c)
    print("masked query vectors: ", vecs2)
    if method == 'COSSIM':
        c = 0
        for v1,v2,s in zip(vecs1, vecs2, v1dbscores):
            cossim = dot(v1, v2)/(norm(v1)*norm(v2))
            dst = (1 + cossim) * s
            #print("Distance between {} and {} = {}".format(v1, v2, dst))
            print("Distance {} = {} ,db result= {}".format(c, dst, resultlist[c][1]))
            c = c + 1
    elif method == 'L1':
        c = 0
        for v1,v2,s in zip(vecs1, vecs2, v1dbscores):
            l1norm = sum(abs(a - b) for a, b in zip(v1,v2))
            dst = 1 / (1 + l1norm + penalties[c]) * s
            #print("Distance between {} and {} = {}".format(v1, v2, dst))
            print("Distance {} = {} ,db result= {}".format(c, dst, resultlist[c][1]))
            c = c + 1
    elif method == 'L2':
        c = 0
        for v1,v2,s in zip(vecs1, vecs2, v1dbscores):
            dst = 1 / (1 + distance.euclidean(v1,v2) + penalties[c]) * s
            #print("Distance between {} and {} = {}".format(v1, v2, dst))
            print("Distance {} = {} ,db result= {}".format(c, dst, resultlist[c][1]))
            c = c + 1
    else:
        raise ValueError()