import numpy as np
from sklearn.decomposition import PCA

def normalizevec(featurevector, rangemin=0, rangemax=1, mask=False):
    if mask:
        #Filter out unvalid features (=-1) for normalization
        #Then combine normalization and unvalid features in same ordering
        maskvalid = []
        for n in featurevector:
            if n == -1:
                maskvalid.append(False)
            else:
                maskvalid.append(True)
        subvec = [n for n,l in zip(featurevector, maskvalid) if l=='True']
        normsubvec = [ (x-min(featurevector)) * (rangemax - rangemin)/(max(featurevector) - min(featurevector)) + rangemin for x in subvec]
        normvec = []
        c_valid = 0
        for l in maskvalid:
            if l=='True':
                normvec.append(normsubvec[c_valid])
                c_valid += 1
            else:
                normvec.append(-1)
        return normvec
    else:
        #Normalize aka rescale input vector to new range
        return [ (x-min(featurevector)) * (rangemax - rangemin)/(max(featurevector) - min(featurevector)) + rangemin for x in featurevector]

def applyPCA(json_data, dim=None, pca=None):
    gpds = [item['gpd'] for item in json_data]
    if pca is None:
        pca = PCA(n_components=dim)
        pca_result = pca.fit_transform(gpds)
    else:
        pca_result = pca.transform(gpds)
    for i,item in enumerate(json_data):
        item['gpd'] = list(pca_result[i])
    
    return pca

def dict_to_item_list(d, level=1):
        #Ressolves double list items k:[[a],[b]] -> [k:[a]],[k:[b]] 
        items = []
        for k,item in d.items():
            if level==1:
                if isinstance(item[0], list):
                    for e in item:
                        items.append([k,e])
                else:
                    items.append([k,item])
            if level==0:
                if isinstance(item, list):
                    for e in item:
                        items.append([k,e])
                else:
                    items.append([k,item])
        return items

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    div = det(xdiff, ydiff)
    if div == 0:
        print('Info: lines do not intersect')
        return -1
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]

def angle(l1,l2):
    [p11,p12] = list(l1.coords)
    [p21,p22] = list(l2.coords)
    #print("coords1: ", list(l1.coords))
    #print("coords2: ", list(l2.coords))
    
    #When line is a point (due to prediction overlay kpts) return 0
    if (p11[0] == p12[0] and p11[1] == p12[1]) or (p21[0] == p22[0] and p21[1] == p22[1]):
        return 0
    #Find intersection point for error prone angle computation
    if p11==p21:
        j1 = np.subtract(p12,p11)
        j2 = np.subtract(p22,p21)
    elif p11==p22:
        j1 = np.subtract(p12,p11)
        j2 = np.subtract(p21,p22)
    elif p12==p21:
        j1 = np.subtract(p11,p12)
        j2 = np.subtract(p22,p21)
    elif p12==p22:
        j1 = np.subtract(p11,p12)
        j2 = np.subtract(p21,p22)
    else:
        #Rare call to this computation
        #Don't using shapely cause no extended lines
        intersect = line_intersection(l1.coords, l2.coords)
        if intersect == -1: #parallel lines
            return 0.0
        j1 = np.subtract(p12,intersect)
        j2 = np.subtract(p22,intersect)
    
    #plt.plot([0,j1[0]], [0,j1[1]], 'g-', lw=1)
    #plt.plot([0,j2[0]], [0,j2[1]], 'g-', lw=1)
    j1_norm = j1/np.linalg.norm(j1)
    j2_norm = j2/np.linalg.norm(j2)
    return np.arccos(np.clip(np.dot(j1_norm, j2_norm), -1.0, 1.0))