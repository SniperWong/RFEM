import random
import numpy as np
from sklearn import metrics
from numpy import *
from sklearn.decomposition import PCA      
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt
from datetime import datetime
start = datetime.now()

from sklearn.tree import DecisionTreeClassifier

################################################### RFEM
def RFEM38(x, y, times):

    AUC3 = []
    AUPR3 = []
    
    for i3 in range(1,times):
        print('the-',i3,'-group')   
   
        start3 = datetime.now()
        all_auc3 = [] 
        all_aupr3 = [] 
        
        mice_x3 = x.copy()
        mice_y3 = y.copy()    
        index3 = [ii for ii in range(len(mice_x3))]
        random.shuffle(index3) 
        # 5-fold 
        for a3 in range(5):  
            print('the',a3+1,'times')
            
            if a3 == 3:
                x_train3 = np.delete(mice_x3, index3[36*a3:36*a3+37], axis=0)  
                x_test3 = mice_x3[index3[36*a3:36*a3+37]]
                y_train3 = np.delete(mice_y3, index3[36*a3:36*a3+37], axis=0)     
                y_test3 = mice_y3[index3[36*a3:36*a3+37]]  
            elif a3 == 4:
                x_train3 = np.delete(mice_x3, index3[36*a3+1:36*a3+38], axis=0)  
                x_test3 = mice_x3[index3[36*a3+1:36*a3+38]]
                y_train3 = np.delete(mice_y3, index3[36*a3+1:36*a3+38], axis=0)     
                y_test3 = mice_y3[index3[36*a3+1:36*a3+38]]  
            else:
                x_train3 = np.delete(mice_x3, index3[36*a3:36*a3+36], axis=0)  
                x_test3 = mice_x3[index3[36*a3:36*a3+36]]
                y_train3 = np.delete(mice_y3, index3[36*a3:36*a3+36], axis=0)     
                y_test3 = mice_y3[index3[36*a3:36*a3+36]] 
                
            models = [] 
            r_matrices = []  
            feature_subsets =[] 
            d = 500       
            for i_1 in range(d):
                feature_index = [i for i in range(mice_x3.shape[1])]
                np.random.shuffle(feature_index)  
                
                subset1 = feature_index[0:2]
                subset2 = feature_index[2:4]  
                subset3 = feature_index[4:6]
                subset4 = feature_index[6:8]  
                subset5 = feature_index[8:10]
                subset6 = feature_index[10:12]      
                subset7 = feature_index[12:14]
                subset8 = feature_index[14:16]  
                subset9 = feature_index[16:18]
                subset10 = feature_index[18:20]  
                subset11 = feature_index[20:23]  
                subset12 = feature_index[23:26]
                subset13 = feature_index[26:29]      
                subset14 = feature_index[29:32]
                subset15 = feature_index[32:35]  
                subset16 = feature_index[35:38]
                feature_subsets = [subset1,subset2,subset3,subset4,subset5,subset6,subset7,subset8,
                                    subset9,subset10,subset11,subset12,subset13,subset14,subset15,subset16]  
                
                R_matrix = np.zeros((mice_x3.shape[1],mice_x3.shape[1]),dtype=float)  
                x_transformed = []
                for i_2 in range(16):
                    each_subset = feature_subsets[i_2]         
                    # sample 75% training data 
                    index0 = [i for i in range(x_train3.shape[0])]
                    np.random.shuffle(index0)    
                    x_subset0 = x_train3[:,each_subset]     
                    x_subset1 = x_subset0[index0]
                    x_subset = x_subset1[:int(len(x_train3)*0.75)]   
                    
                    # PCA
                    pca = PCA()
                    pca.fit(x_subset)
                    Com_pca = pca.components_.T
                    
                    R_matrix_orig = np.zeros((len(pca.components_),len(pca.components_)),dtype=float)  
                    
                    if i_2 <= 9:      
                        for ii in range(len(pca.components_)):
                            for jj in range(len(pca.components_)):
                                R_matrix_orig[ii, jj] = Com_pca[ii, jj]
                                R_matrix[ii + i_2 * len(pca.components_), each_subset[jj]] = R_matrix_orig[ii, jj] 
                                
                    if i_2 >= 10:             
                        for ii in range(len(pca.components_)):
                            for jj in range(len(pca.components_)):
                                R_matrix_orig[ii, jj] = Com_pca[ii, jj]
                                R_matrix[ii + (i_2-10) * len(pca.components_) + 20, each_subset[jj]] = R_matrix_orig[ii, jj]               
                             
                x_transformed = x_train3.dot(R_matrix)
                
                model = DecisionTreeClassifier(criterion='entropy', max_depth = 3, min_samples_leaf = 4)
                model.fit(x_transformed,y_train3)
                models.append(model)
                r_matrices.append(R_matrix)
        
            predicted_y_all = []
            for i,model in enumerate(models):             
                x_test_new = x_test3.dot(r_matrices[i])    
                predicted_y = model.predict(x_test_new)
                predicted_y_all.append(predicted_y)
                predicted_matrix = np.asmatrix(predicted_y_all)   
                final_prediction = []   
                for i in range(len(y_test3)):
                    pred_from_all_models = np.ravel(predicted_matrix[:,i])                 
                    is_one = sum(pred_from_all_models)/len(models) 
                    final_prediction.append(is_one)         
            y_predict3 = np.array(final_prediction)    
                
            fpr, tpr, thr = metrics.roc_curve(y_test3,y_predict3)
            auc3 = metrics.auc(fpr, tpr)     
            all_auc3.append(auc3)
            
            precision3, recall3, pr_thresholds = metrics.precision_recall_curve(y_test3, y_predict3)
            area3 = metrics.auc(recall3,precision3)
            all_aupr3.append(area3) 

        roc_auc3 = sum(all_auc3)/5  
        AUC3.append(roc_auc3)
 
        PR_auc3 = sum(all_aupr3)/5  
        AUPR3.append(PR_auc3)
        
        print('--------------------------1') 
        end3 = datetime.now()
        print((end3-start3))
    AUC_final_3 = np.average(AUC3)
    print(AUC_final_3)
    
    AUPR_final_3 = np.average(AUPR3)
    print(AUPR_final_3)
    
    return AUC_final_3, AUPR_final_3


################################################### RFEM
def RFEM1226(x, y, times):

    AUC3 = []
    AUPR3 = []
    
    for i3 in range(1,times):
        print('the-',i3,'-group')   
        
        start3 = datetime.now()
        all_auc3 = [] 
        all_aupr3 = [] 
        
        mice_x3 = x.copy()
        mice_y3 = y.copy()    
        index3 = [ii for ii in range(len(mice_x3))]
        random.shuffle(index3) 
        # 5-fold 
        for a3 in range(5):  
            print('the',a3+1,'times')
            
            if a3 == 3:
                x_train3 = np.delete(mice_x3, index3[36*a3:36*a3+37], axis=0)  
                x_test3 = mice_x3[index3[36*a3:36*a3+37]]
                y_train3 = np.delete(mice_y3, index3[36*a3:36*a3+37], axis=0)     
                y_test3 = mice_y3[index3[36*a3:36*a3+37]]  
            elif a3 == 4:
                x_train3 = np.delete(mice_x3, index3[36*a3+1:36*a3+38], axis=0)  
                x_test3 = mice_x3[index3[36*a3+1:36*a3+38]]
                y_train3 = np.delete(mice_y3, index3[36*a3+1:36*a3+38], axis=0)     
                y_test3 = mice_y3[index3[36*a3+1:36*a3+38]]  
            else:
                x_train3 = np.delete(mice_x3, index3[36*a3:36*a3+36], axis=0)  
                x_test3 = mice_x3[index3[36*a3:36*a3+36]]
                y_train3 = np.delete(mice_y3, index3[36*a3:36*a3+36], axis=0)     
                y_test3 = mice_y3[index3[36*a3:36*a3+36]] 
                
            models = [] 
            r_matrices = []  
            feature_subsets =[] 
            d = 500       
            for i_1 in range(d):
                feature_index = [i for i in range(mice_x3.shape[1])]
                np.random.shuffle(feature_index)  
                
                subset1 = feature_index[0:76]
                subset2 = feature_index[76:152]  
                subset3 = feature_index[152:228]
                subset4 = feature_index[228:304]  
                subset5 = feature_index[304:380]
                subset6 = feature_index[380:456]      
                subset7 = feature_index[456:533]
                subset8 = feature_index[533:610]  
                subset9 = feature_index[610:687]
                subset10 = feature_index[687:764]  
                subset11 = feature_index[764:841]  
                subset12 = feature_index[841:918]
                subset13 = feature_index[918:995]      
                subset14 = feature_index[995:1072]
                subset15 = feature_index[1072:1149]  
                subset16 = feature_index[1149:1226]
                feature_subsets = [subset1,subset2,subset3,subset4,subset5,subset6,subset7,subset8,
                                    subset9,subset10,subset11,subset12,subset13,subset14,subset15,subset16]  
                
                R_matrix = np.zeros((mice_x3.shape[1],mice_x3.shape[1]),dtype=float)  
                x_transformed = []
                for i_2 in range(16):
                    each_subset = feature_subsets[i_2]         
                    # sample 75% training data 
                    index0 = [i for i in range(x_train3.shape[0])]
                    np.random.shuffle(index0)    
                    x_subset0 = x_train3[:,each_subset]     
                    x_subset1 = x_subset0[index0]
                    x_subset = x_subset1[:int(len(x_train3)*0.75)]   
                    
                    # PCA
                    pca = PCA()
                    pca.fit(x_subset)
                    Com_pca = pca.components_.T
                    
                    R_matrix_orig = np.zeros((len(pca.components_),len(pca.components_)),dtype=float)  
                    
                                        
                    if i_2 <= 5:      
                        for ii in range(len(pca.components_)):
                            for jj in range(len(pca.components_)):
                                R_matrix_orig[ii, jj] = Com_pca[ii, jj]
                                R_matrix[ii + i_2 * len(pca.components_), each_subset[jj]] = R_matrix_orig[ii, jj] 
                                
                    if i_2 >= 6:             
                        for ii in range(len(pca.components_)):
                            for jj in range(len(pca.components_)):
                                R_matrix_orig[ii, jj] = Com_pca[ii, jj]
                                R_matrix[ii + (i_2-6) * len(pca.components_) + 456, each_subset[jj]] = R_matrix_orig[ii, jj]               
               
                x_transformed = x_train3.dot(R_matrix)
                
                model = DecisionTreeClassifier(criterion='entropy', max_depth = 3, min_samples_leaf = 4)
                model.fit(x_transformed,y_train3)
                models.append(model)
                r_matrices.append(R_matrix)
        
            predicted_y_all = []
            for i,model in enumerate(models):             
                x_test_new = x_test3.dot(r_matrices[i])    
                predicted_y = model.predict(x_test_new)
                predicted_y_all.append(predicted_y)
                predicted_matrix = np.asmatrix(predicted_y_all)   
                final_prediction = []   
                for i in range(len(y_test3)):
                    pred_from_all_models = np.ravel(predicted_matrix[:,i])                 
                    is_one = sum(pred_from_all_models)/len(models) 
                    final_prediction.append(is_one)         
            y_predict3 = np.array(final_prediction)
                
            fpr, tpr, thr = metrics.roc_curve(y_test3,y_predict3)
            auc3 = metrics.auc(fpr, tpr)     
            all_auc3.append(auc3)
            
            precision3, recall3, pr_thresholds = metrics.precision_recall_curve(y_test3, y_predict3)
            area3 = metrics.auc(recall3,precision3)
            all_aupr3.append(area3) 
        
        roc_auc3 = sum(all_auc3)/5  
        AUC3.append(roc_auc3)

        PR_auc3 = sum(all_aupr3)/5  
        AUPR3.append(PR_auc3)

        print('---------------------------------------2') 
        end3 = datetime.now()
        print((end3-start3))
    AUC_final_3 = np.average(AUC3)
    print(AUC_final_3)
    
    AUPR_final_3 = np.average(AUPR3)
    print(AUPR_final_3)
    
    return AUC_final_3, AUPR_final_3

################################################### RFEM
def RFEM1264(x, y, times):

    AUC3 = []
    AUPR3 = []
    
    for i3 in range(1,times):
        print('the-',i3,'-group')   
        
        start3 = datetime.now()
        all_auc3 = [] 
        all_aupr3 = [] 
        
        mice_x3 = x.copy()
        mice_y3 = y.copy()    
        index3 = [ii for ii in range(len(mice_x3))]
        random.shuffle(index3) 
        # 5-fold 
        for a3 in range(5):  
            print('the',a3+1,'times')
            
            if a3 == 3:
                x_train3 = np.delete(mice_x3, index3[36*a3:36*a3+37], axis=0)  
                x_test3 = mice_x3[index3[36*a3:36*a3+37]]
                y_train3 = np.delete(mice_y3, index3[36*a3:36*a3+37], axis=0)     
                y_test3 = mice_y3[index3[36*a3:36*a3+37]]  
            elif a3 == 4:
                x_train3 = np.delete(mice_x3, index3[36*a3+1:36*a3+38], axis=0)  
                x_test3 = mice_x3[index3[36*a3+1:36*a3+38]]
                y_train3 = np.delete(mice_y3, index3[36*a3+1:36*a3+38], axis=0)     
                y_test3 = mice_y3[index3[36*a3+1:36*a3+38]]  
            else:
                x_train3 = np.delete(mice_x3, index3[36*a3:36*a3+36], axis=0)  
                x_test3 = mice_x3[index3[36*a3:36*a3+36]]
                y_train3 = np.delete(mice_y3, index3[36*a3:36*a3+36], axis=0)     
                y_test3 = mice_y3[index3[36*a3:36*a3+36]] 
                
            models = [] 
            r_matrices = []  
            feature_subsets =[] 
            d = 500       
            for i_1 in range(d):
                feature_index = [i for i in range(mice_x3.shape[1])]
                np.random.shuffle(feature_index)  
                
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
                
                R_matrix = np.zeros((mice_x3.shape[1],mice_x3.shape[1]),dtype=float)  
                x_transformed = []
                for i_2 in range(16):
                    each_subset = feature_subsets[i_2]         
                    # sample 75% training data 
                    index0 = [i for i in range(x_train3.shape[0])]
                    np.random.shuffle(index0)    
                    x_subset0 = x_train3[:,each_subset]     
                    x_subset1 = x_subset0[index0]
                    x_subset = x_subset1[:int(len(x_train3)*0.75)]   
                    
                    # PCA
                    pca = PCA()
                    pca.fit(x_subset)
                    Com_pca = pca.components_.T
                    
                    R_matrix_orig = np.zeros((len(pca.components_),len(pca.components_)),dtype=float)  
                    for ii in range(len(pca.components_)):
                        for jj in range(len(pca.components_)):
                            R_matrix_orig[ii, jj] = Com_pca[ii, jj]
                            R_matrix[ii + i_2 * len(pca.components_), each_subset[jj]] = R_matrix_orig[ii, jj]   
                             
                x_transformed = x_train3.dot(R_matrix)
                
                model = DecisionTreeClassifier(criterion='entropy', max_depth = 3, min_samples_leaf = 4)
                model.fit(x_transformed,y_train3)
                models.append(model)
                r_matrices.append(R_matrix)
        
            predicted_y_all = []
            for i,model in enumerate(models):             
                x_test_new = x_test3.dot(r_matrices[i])    
                predicted_y = model.predict(x_test_new)
                predicted_y_all.append(predicted_y)
                predicted_matrix = np.asmatrix(predicted_y_all)   
                final_prediction = []   
                for i in range(len(y_test3)):
                    pred_from_all_models = np.ravel(predicted_matrix[:,i])                 
                    is_one = sum(pred_from_all_models)/len(models) 
                    final_prediction.append(is_one)         
            y_predict3 = np.array(final_prediction)    
                
            fpr, tpr, thr = metrics.roc_curve(y_test3,y_predict3)
            auc3 = metrics.auc(fpr, tpr)     
            all_auc3.append(auc3)
            
            precision3, recall3, pr_thresholds = metrics.precision_recall_curve(y_test3, y_predict3)
            area3 = metrics.auc(recall3,precision3)
            all_aupr3.append(area3) 
  
        roc_auc3 = sum(all_auc3)/5  
        AUC3.append(roc_auc3)

        PR_auc3 = sum(all_aupr3)/5  
        AUPR3.append(PR_auc3)

        print('--------------------------------------------------------------3') 
        end3 = datetime.now()
        print((end3-start3))
    AUC_final_3 = np.average(AUC3)
    print(AUC_final_3)
    
    AUPR_final_3 = np.average(AUPR3)
    print(AUPR_final_3)
    
    return AUC_final_3, AUPR_final_3

if __name__ == '__main__':
    start = datetime.now()
    # data1
    dataset_x1 = np.loadtxt('dataset38_x.txt',dtype=float)  
    dataset_y1 = np.loadtxt('dataset38_y.txt',dtype=float) 
    x1 = mat(dataset_x1)
    y1 = dataset_y1
    # data2
    dataset_x2 = np.loadtxt('dataset1226_x.txt',dtype=float)  
    dataset_y2 = np.loadtxt('dataset1226_y.txt',dtype=float) 
    x2 = mat(dataset_x2)
    y2 = dataset_y2
    # data3
    dataset_x3 = np.loadtxt('dataset38+1226_x.txt',dtype=float)  
    dataset_y3 = np.loadtxt('dataset38+1226_y.txt',dtype=float) 
    x3 = mat(dataset_x3)
    y3 = dataset_y3

    times = 51
#### Ablation experiment
    AUC_final_1, AUPR_final_1 = RFEM38(x1, y1, times)
    AUC_final_2, AUPR_final_2 = RFEM1226(x2, y2, times)
    AUC_final_3, AUPR_final_3 = RFEM1264(x3, y3, times)
    
    end = datetime.now()
    print((end-start))
    
    print(AUC_final_1)
    print(AUPR_final_1)
    print(AUC_final_2)
    print(AUPR_final_2)
    print(AUC_final_3)
    print(AUPR_final_3)




