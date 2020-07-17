# In[1]: Load Library
import pandas as pd
import numpy as np
from pylab import rcParams
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import normalize 
from sklearn.decomposition import PCA 
from datetime import datetime

# In[2]: Load Dataset
dataset = pd.read_excel("DatasetFraud.xlsx")
dataset = dataset.fillna(0)
dataset['satuan_kerja_nama'] = dataset['satuan_kerja_nama'].astype('category')
dataset['unit_kerja_nama'] = dataset['unit_kerja_nama'].astype('category')
dataset.info()

# In[3]: Label Encoding
labelencoder = LabelEncoder()
dataset['satker'] = labelencoder.fit_transform(dataset['satuan_kerja_nama'].astype(str))
dataset['uker'] = labelencoder.fit_transform(dataset['unit_kerja_nama'].astype(str))
dataset['session'] = labelencoder.fit_transform(dataset['session_id'].astype(str))
dataset['browser'] = labelencoder.fit_transform(dataset['browser_id'].astype(str))
dataset['ip_cat'] = labelencoder.fit_transform(dataset['ip'].astype(str))

# In[4]: Select Column and normalize
df = dataset[["mereview_score_avg","mereview_score_stdev","satker","session","browser","ip_cat"]]
df.info()
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(df) 
X_normalized = normalize(X_scaled)  
X_normalized = pd.DataFrame(X_normalized) 

# In[5]: Implementasi DBSCAN
model = DBSCAN(eps = 0.0375, min_samples = 3).fit(X_normalized) 
labels = model.labels_ 
print("Berhasil Menemukan sebanyak {} Cluster".format(np.unique(labels).size))
dataset['cluster'] = labels

# In[6]: Save Data to Excel
now = datetime.now()
tanggal = now.strftime("%Y-%m-%d %H.%M.%S")
dataset["peg_nip"] = dataset["peg_nip"].apply(lambda x: f"'{x}")
dataset.to_excel("Hasil Tanggal {}.xlsx".format(tanggal))