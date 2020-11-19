script = """
def m1 = doc['mask'].value;
def m2 = params.queryMask;
int[] x = new int[m1.length]; 
for(int i; i < m1.length; i++) {
    if (m1.charAt(i) == '1' && m2.charAt(i) == '1') {
        x[i] = 1;
    }
Debug.explain(x);
def vec1 = params.queryVector * x;
def vec2 = doc['gpd'] * x;
return cosineSimilarity(vec1, vec2) + 1.0;
"""
print(len(script))
print(script[724:749])