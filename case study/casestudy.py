import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA      
from datetime import datetime
import random
from numpy import *
import pandas as pd                                     
from openpyxl import load_workbook 
from openpyxl import Workbook

start = datetime.now()
# training samples
dataset_x = pd.read_excel('dataset_x.xlsx')
dataset_y = pd.read_excel('dataset_y.xlsx')

# candidate samples
dataset = pd.read_excel('candidate_x.xlsx')

wb1 = Workbook()
ws1 = wb1.active
ws1.title = 'Top10 validation status'
ws1.append(['rank', 'name', 'score'])   
   
mice_x = mat(dataset_x)
mice_y = mat(dataset_y)
dataset = mat(dataset)

index = [i for i in range(len(mice_x))]
random.shuffle(index) 

x_train = mice_x[index]  
y_train = mice_y[index]    
x_test = dataset

models = [] 
r_matrices = []  
feature_subsets =[] 
d = 500       
for i in range(d):
    feature_index = [a for a in range(mice_x.shape[1])]
    np.random.shuffle(feature_index) 
    # 16组
    subset1 = feature_index[0:79]
    subset2 = feature_index[79:158]  
    subset3 = feature_index[158:237]
    subset4 = feature_index[237:316]  
    subset5 = feature_index[316:395]
    subset6 = feature_index[395:474]      
    subset7 = feature_index[474:553]
    subset8 = feature_index[553:632]  
    subset9 = feature_index[632:711]
    subset10 = feature_index[711:790]  
    subset11 = feature_index[790:869]  
    subset12 = feature_index[869:948]
    subset13 = feature_index[948:1027]      
    subset14 = feature_index[1027:1106]
    subset15 = feature_index[1106:1185]  
    subset16 = feature_index[1185:1264]
    feature_subsets = [subset1,subset2,subset3,subset4,subset5,subset6,subset7,subset8,
                        subset9,subset10,subset11,subset12,subset13,subset14,subset15,subset16]     

    R_matrix = np.zeros((mice_x.shape[1],mice_x.shape[1]),dtype=float)
    x_transformed = []
    for i_2 in range(16):
      each_subset = feature_subsets[i_2]         
      # sample 75% training data 
      index0 = [k for k in range(x_train.shape[0])]
      np.random.shuffle(index0)    
      x_subset0 = x_train[:,each_subset]     
      x_subset1 = x_subset0[index0]
      x_subset = x_subset1[:int(len(x_train)*0.75)]   
      
      # PCA
      pca = PCA()
      pca.fit(x_subset)
      Com_pca = pca.components_.T
      
      R_matrix_orig = np.zeros((len(pca.components_),len(pca.components_)),dtype=float)  
      for ii in range(len(pca.components_)):
          for jj in range(len(pca.components_)):
              R_matrix_orig[ii, jj] = Com_pca[ii, jj]
              R_matrix[ii + i_2 * len(pca.components_), each_subset[jj]] = R_matrix_orig[ii, jj]  
                       
    x_transformed = x_train.dot(R_matrix)
    
    model = DecisionTreeClassifier(criterion='entropy', max_depth = 3, min_samples_leaf = 4)    
    model.fit(x_transformed,y_train)
    models.append(model)
    r_matrices.append(R_matrix)

for j in range(len(x_test)):    
    x_test0 =  np.delete(x_test, [0], axis=1)           
    predicted_y_all = []
    for s,model in enumerate(models):  
        x_test_new = x_test0[j].dot(r_matrices[s])    
        predicted_y = model.predict(x_test_new)
        predicted_y_all.append(predicted_y)
    is_one = sum(predicted_y_all)/len(models)       
    ws1.append([j+1, x_test[j,0], is_one])   
   
wb1.save(r'result.xlsx')

end = datetime.now()
print((end-start))
print('--------------------------------------------------------------')   


