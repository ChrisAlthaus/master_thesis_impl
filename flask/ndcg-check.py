# import required package 
from sklearn.metrics import ndcg_score, dcg_score 
import numpy as np 

# Releveance scores in Ideal order 
true_relevance = np.asarray([[2, 2, 2, 2, 2]]) 

# Releveance scores in output order 
relevance_score = np.asarray([[1,1,1,1,1]]) 

# DCG score 
dcg = dcg_score(true_relevance, relevance_score) 
print("DCG score : ", dcg) 

# IDCG score 
idcg = dcg_score(true_relevance, true_relevance) 
print("IDCG score : ", idcg) 

# Normalized DCG score 
ndcg = dcg / idcg 
print("nDCG score : ", ndcg) 

# or we can use the scikit-learn ndcg_score package 
print("nDCG score (from function) : ", ndcg_score( 
	true_relevance, relevance_score)) 