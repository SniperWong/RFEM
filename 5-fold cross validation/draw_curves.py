import random
import numpy as np
from sklearn import metrics
from numpy import *
from sklearn.decomposition import PCA      
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt
from datetime import datetime
start = datetime.now()
# 1 
from sklearn.linear_model import LogisticRegression
# 2
from xgboost import XGBClassifier
# 3
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings("ignore")

################################################### miES
def miES(x, y, times):
    
    AUC1 = []
    TPR_1 = []
    FPR_1 = []
    
    Precision_1 = []
    Recall_1 = []
    AUPR1 = []
    
    for i1 in range(1,times):
        print('the-',i1,'-group')
        mice_x1 = x.copy()
        mice_y1 = y.copy()
        start1 = datetime.now()
        TPR1 = np.zeros((5,100000))
        FPR1 = np.zeros((5,100000))
        
        Precision1 = np.zeros((5,100000))
        Recall1 = np.zeros((5,100000))
        
        model1 = LogisticRegression(solver='liblinear',random_state=int(i1))   
        all_auc1 = [] 
        
        all_aupr1 = [] 
        
        index1 = [a for a in range(len(mice_x1))]
        random.shuffle(index1)   
        for a1 in range(5):  
            print('the-',a1+1,'-times')
            if a1 == 3:
                X_train1 = np.delete(mice_x1, index1[36*a1:36*a1+37], axis=0)  
                X_test1 = mice_x1[index1[36*a1:36*a1+37]]
                y_train1 = np.delete(mice_y1, index1[36*a1:36*a1+37], axis=0)     
                y_test1 = mice_y1[index1[36*a1:36*a1+37]]  
            elif a1 == 4:
                X_train1 = np.delete(mice_x1, index1[36*a1+1:36*a1+38], axis=0)  
                X_test1 = mice_x1[index1[36*a1+1:36*a1+38]]
                y_train1 = np.delete(mice_y1, index1[36*a1+1:36*a1+38], axis=0)     
                y_test1 = mice_y1[index1[36*a1+1:36*a1+38]]  
            else:
                X_train1 = np.delete(mice_x1, index1[36*a1:36*a1+36], axis=0)  
                X_test1 = mice_x1[index1[36*a1:36*a1+36]]
                y_train1 = np.delete(mice_y1, index1[36*a1:36*a1+36], axis=0)     
                y_test1 = mice_y1[index1[36*a1:36*a1+36]] 
                
            model1.fit(X_train1,y_train1)
            pp1 = model1.predict_proba(X_test1)    
            pp1.tolist()
            y_predict1 = pp1[:,1]
            
            for b1 in range(100000):
                p0 = (b1+1)/100000         
                p1 = np.zeros(len(y_predict1)) 
                for k1 in range(len(y_predict1)):
                    if y_predict1[k1] > p0:
                        p1[k1] = 1
                    else:
                        p1[k1] = 0
                np.array(p1)
                np.array(y_test1)
                confusion1 = confusion_matrix(y_test1, p1)    
                
                TPR1[a1,b1] = (confusion1[1,1]/(confusion1[1,0]+confusion1[1,1]))
                FPR1[a1,b1] = (confusion1[0,1]/(confusion1[0,1]+confusion1[0,0]))   
                
                if (confusion1[0,1]+confusion1[1,1]) != 0 and (confusion1[1,1]+confusion1[1,0]) != 0:
                    Precision1[a1,b1] = (confusion1[1,1]/(confusion1[0,1]+confusion1[1,1]))
                    Recall1[a1,b1] = (confusion1[1,1]/(confusion1[1,1]+confusion1[1,0]))   
                    
            fpr, tpr, thr = metrics.roc_curve(y_test1,y_predict1)
            auc1 = metrics.auc(fpr, tpr) 
            all_auc1.append(auc1)      
            
            precision1, recall1, pr_thresholds1 = metrics.precision_recall_curve(y_test1, y_predict1)
            area1 = metrics.auc(recall1,precision1)
            all_aupr1.append(area1)  
               
        fpr1, tpr1 = np.mean(FPR1, axis=0), np.mean(TPR1, axis=0)
        roc_auc1 = sum(all_auc1)/5
        AUC1.append(roc_auc1)
        TPR_1.append(tpr1)
        FPR_1.append(fpr1)
     
        precision11, recall11 = np.mean(Precision1, axis=0),np.mean(Recall1, axis=0)
              
        PR_auc1 = sum(all_aupr1)/5  
        AUPR1.append(PR_auc1)
        Precision_1.append(precision11)
        Recall_1.append(recall11) 
        print('-------------------------1') 
        end1 = datetime.now()
        print((end1-start1))
    AUC_final_1 = np.average(AUC1)
    fpr_1, tpr_1 = np.mean(FPR_1, axis=0),np.mean(TPR_1, axis=0)
    print(AUC_final_1)
    
    AUPR_final_1 = np.average(AUPR1)
    precision_1, recall_1 = np.mean(Precision_1, axis=0),np.mean(Recall_1, axis=0)
    print(AUPR_final_1)
    
    return  fpr_1, tpr_1, AUC_final_1, precision_1, recall_1, AUPR_final_1
   
################################################### PESM
def PESM(x, y, times):
    
    AUC2 = []
    TPR_2 = []
    FPR_2 = []
    
    Precision_2 = []
    Recall_2 = []
    AUPR2 = []
    
    for i2 in range(1,times):
        print('the-',i2,'-group')
        start2 = datetime.now()
        mice_x2 = x.copy()
        mice_y2 = y.copy()
        TPR2 = np.zeros((5,100000))
        FPR2 = np.zeros((5,100000))
        
        Precision2 = np.zeros((5,100000))
        Recall2 = np.zeros((5,100000))
        
        model2 = XGBClassifier(max_depth=6, learning_rate=0.01, n_estimators=1000)
        index2 = [aa for aa in range(len(mice_x2))]
        random.shuffle(index2)  
        all_auc2 = [] 
        
        all_aupr2 = [] 
        
        for a2 in range(5):  
            print('the',a2+1,'times')
            if a2 == 3:
                x_train2 = np.delete(mice_x2, index2[36*a2:36*a2+37], axis=0)  
                x_test2 = mice_x2[index2[36*a2:36*a2+37]]
                y_train2 = np.delete(mice_y2, index2[36*a2:36*a2+37], axis=0)     
                y_test2 = mice_y2[index2[36*a2:36*a2+37]]  
            elif a2 == 4:
                x_train2 = np.delete(mice_x2, index2[36*a2+1:36*a2+38], axis=0)  
                x_test2 = mice_x2[index2[36*a2+1:36*a2+38]]
                y_train2 = np.delete(mice_y2, index2[36*a2+1:36*a2+38], axis=0)     
                y_test2 = mice_y2[index2[36*a2+1:36*a2+38]]  
            else:
                x_train2 = np.delete(mice_x2, index2[36*a2:36*a2+36], axis=0)  
                x_test2 = mice_x2[index2[36*a2:36*a2+36]]
                y_train2 = np.delete(mice_y2, index2[36*a2:36*a2+36], axis=0)     
                y_test2 = mice_y2[index2[36*a2:36*a2+36]] 
            model2.fit(x_train2, y_train2)
            pp2 = model2.predict_proba(x_test2)
            pp2.tolist()
            y_predict2 = pp2[:,1]
            for b2 in range(100000):
                p0 = (b2+1)/100000         
                p2 = np.zeros(len(y_predict2)) 
                for k2 in range(len(y_predict2)):
                    if y_predict2[k2] > p0:
                        p2[k2] = 1
                    else:
                        p2[k2] = 0
                np.array(p2)
                np.array(y_test2)
                confusion2 = confusion_matrix(y_test2, p2)       
                TPR2[a2,b2] = (confusion2[1,1]/(confusion2[1,0]+confusion2[1,1]))
                FPR2[a2,b2] = (confusion2[0,1]/(confusion2[0,1]+confusion2[0,0]))     
                
                if (confusion2[0,1]+confusion2[1,1]) != 0 and (confusion2[1,1]+confusion2[1,0]) != 0:
                    Precision2[a2,b2] = (confusion2[1,1]/(confusion2[0,1]+confusion2[1,1]))
                    Recall2[a2,b2] = (confusion2[1,1]/(confusion2[1,1]+confusion2[1,0]))  
            fpr, tpr, thr = metrics.roc_curve(y_test2,y_predict2)
            auc2 = metrics.auc(fpr, tpr) 
            all_auc2.append(auc2)
            
            precision2, recall2, pr_thresholds2 = metrics.precision_recall_curve(y_test2, y_predict2)
            area2 = metrics.auc(recall2,precision2)
            all_aupr2.append(area2) 
            
        fpr2,tpr2 = np.mean(FPR2, axis=0),np.mean(TPR2, axis=0)  
        roc_auc2 = sum(all_auc2)/5  
        AUC2.append(roc_auc2)
        TPR_2.append(tpr2)
        FPR_2.append(fpr2)   
    
        precision22, recall22 = np.mean(Precision2, axis=0),np.mean(Recall2, axis=0)      
        PR_auc2 = sum(all_aupr2)/5  
        AUPR2.append(PR_auc2)
        Precision_2.append(precision22)
        Recall_2.append(recall22)   
    
        end2 = datetime.now()
        print((end2-start2))
        print('-----------------------------------------2') 
    AUC_final_2 = np.average(AUC2)
    fpr_2, tpr_2 = np.mean(FPR_2, axis=0),np.mean(TPR_2, axis=0)
    print(AUC_final_2)
    
    AUPR_final_2 = np.average(AUPR2)
    precision_2, recall_2 = np.mean(Precision_2, axis=0),np.mean(Recall_2, axis=0)
    print(AUPR_final_2)
    return  fpr_2, tpr_2, AUC_final_2, precision_2, recall_2, AUPR_final_2

################################################### RFEM
def RFEM(x, y, times):

    AUC3 = []
    TPR_3 = []
    FPR_3 = []
    
    Precision_3 = []
    Recall_3 = []
    AUPR3 = []
    
    for i3 in range(1,times):
        print('the-',i3,'-group')   
        TPR3 = np.zeros((5,100000))
        FPR3 = np.zeros((5,100000))
        
        Recall3 = np.zeros((5,100000))
        Precision3 = np.zeros((5,100000))
        
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
            for b3 in range(100000):
                p0 = (b3+1)/100000       
                p3 = np.zeros(len(y_predict3)) 
                for k3 in range(len(y_predict3)):
                    if y_predict3[k3] > p0:
                        p3[k3] = 1
                    else:
                        p3[k3] = 0
                np.array(p3)
                np.array(y_test3)
                confusion3 = confusion_matrix(y_test3, p3)       
                TPR3[a3,b3] = (confusion3[1,1]/(confusion3[1,0]+confusion3[1,1]))
                FPR3[a3,b3] = (confusion3[0,1]/(confusion3[0,1]+confusion3[0,0]))   
                
                if (confusion3[0,1]+confusion3[1,1]) != 0 and (confusion3[1,1]+confusion3[1,0]) != 0:
                    Precision3[a3,b3] = (confusion3[1,1]/(confusion3[0,1]+confusion3[1,1]))
                    Recall3[a3,b3] = (confusion3[1,1]/(confusion3[1,1]+confusion3[1,0])) 
                
            fpr, tpr, thr = metrics.roc_curve(y_test3,y_predict3)
            auc3 = metrics.auc(fpr, tpr)     
            all_auc3.append(auc3)
            
            precision3, recall3, pr_thresholds = metrics.precision_recall_curve(y_test3, y_predict3)
            area3 = metrics.auc(recall3,precision3)
            all_aupr3.append(area3) 
            
        fpr3, tpr3 = np.mean(FPR3, axis=0),np.mean(TPR3, axis=0) 
        
        roc_auc3 = sum(all_auc3)/5  
        AUC3.append(roc_auc3)
        TPR_3.append(tpr3)
        FPR_3.append(fpr3)    
        
        precision33, recall33 = np.mean(Precision3, axis=0),np.mean(Recall3, axis=0)
       
        PR_auc3 = sum(all_aupr3)/5  
        AUPR3.append(PR_auc3)
        Precision_3.append(precision33)
        Recall_3.append(recall33) 
        
        print('--------------------------------------------------------------3') 
        end3 = datetime.now()
        print((end3-start3))
    AUC_final_3 = np.average(AUC3)
    fpr_3, tpr_3 = np.mean(FPR_3, axis=0),np.mean(TPR_3, axis=0)
    print(AUC_final_3)
    
    AUPR_final_3 = np.average(AUPR3)
    precision_3, recall_3 = np.mean(Precision_3, axis=0),np.mean(Recall_3, axis=0)
    print(AUPR_final_3)
    
    return  fpr_3, tpr_3, AUC_final_3, precision_3, recall_3, AUPR_final_3

if __name__ == '__main__':

    # data1
    dataset_x1 = np.loadtxt('miES_x.txt',dtype=float)  
    dataset_y1 = np.loadtxt('miES_y.txt',dtype=float) 
    x1 = mat(dataset_x1)
    y1 = dataset_y1
    # data2
    dataset_x2 = np.loadtxt('PESM_x.txt',dtype=float)  
    dataset_y2 = np.loadtxt('PESM_y.txt',dtype=float) 
    x2 = mat(dataset_x2)
    y2 = dataset_y2
    # data3
    dataset_x3 = np.loadtxt('dataset_x.txt',dtype=float)  
    dataset_y3 = np.loadtxt('dataset_y.txt',dtype=float) 
    x3 = mat(dataset_x3)
    y3 = dataset_y3

    times = 51
#### this research
    fpr_1, tpr_1, AUC_final_1, precision_1, recall_1, AUPR_final_1 = miES(x3, y3, times)
    fpr_2, tpr_2, AUC_final_2, precision_2, recall_2, AUPR_final_2 = PESM(x3, y3, times)
    fpr_3, tpr_3, AUC_final_3, precision_3, recall_3, AUPR_final_3 = RFEM(x3, y3, times)
    
#### respective research
    fpr_1_resp, tpr_1_resp, AUC_final_1_resp, precision_1_resp, recall_1_resp, AUPR_final_1_resp = miES(x1, y1, times)
    fpr_2_resp, tpr_2_resp, AUC_final_2_resp, precision_2_resp, recall_2_resp, AUPR_final_2_resp = PESM(x2, y2, times)
    fpr_3_resp, tpr_3_resp, AUC_final_3_resp, precision_3_resp, recall_3_resp, AUPR_final_3_resp = fpr_3, tpr_3, AUC_final_3, precision_3, recall_3, AUPR_final_3

#### this research

    # draw ROC curve
    Font={'size':12, 'family':'Times New Roman'}
    plt.plot(fpr_1, tpr_1, 'b', label = 'miES (AUC = %0.3f)' % AUC_final_1, color='Red')
    plt.plot(fpr_2, tpr_2, 'b', label = 'PESM (AUC = %0.3f)' % AUC_final_2, color='k')
    plt.plot(fpr_3, tpr_3, 'b', label = 'RFEM (AUC = %0.3f)' % AUC_final_3, color='RoyalBlue')
    plt.legend(loc = 'lower right', prop=Font)
    plt.ylabel('True Positive Rate', Font)
    plt.xlabel('False Positive Rate', Font)
    plt.ylim([0.0, 1.05])    
    plt.xlim([0.0, 1.0])   
    plt.tick_params(labelsize=12)
    plt.savefig('ROC curve(this research).tiff', dpi=600, bbox_inches = 'tight')
    plt.show()
    print('-----------------------------------------')   
    end = datetime.now()
    print((end-start))
    
    # draw PR curve
    Font={'size':12, 'family':'Times New Roman'} 
    plt.plot(recall_1, precision_1, 'b', label = 'miES (AUPR = %0.3f)' % AUPR_final_1, color='Red')
    plt.plot(recall_2, precision_2, 'b', label = 'PESM (AUPR = %0.3f)' % AUPR_final_2, color='k')
    plt.plot(recall_3, precision_3, 'b', label = 'RFEM (AUPR = %0.3f)' % AUPR_final_3, color='RoyalBlue')
    plt.legend(loc = 'lower right', prop=Font)
    plt.xlabel('Recall', Font)    
    plt.ylabel('Precision', Font)       
    plt.ylim([0.0, 1.05])    
    plt.xlim([0.0, 1.0])    
    plt.tick_params(labelsize=12)     
    plt.savefig('PR curve(this research).tiff', dpi=600, bbox_inches = 'tight')
    plt.show()
    print('--------------------------------------------------------------')   
    end = datetime.now()
    print((end-start))
   
    
#### respective research

    # draw ROC curve
    Font={'size':12, 'family':'Times New Roman'}
    plt.plot(fpr_1_resp, tpr_1_resp, 'b', label = 'miES (AUC = %0.3f)' % AUC_final_1_resp, color='Red')
    plt.plot(fpr_2_resp, tpr_2_resp, 'b', label = 'PESM (AUC = %0.3f)' % AUC_final_2_resp, color='k')
    plt.plot(fpr_3_resp, tpr_3_resp, 'b', label = 'RFEM (AUC = %0.3f)' % AUC_final_3_resp, color='RoyalBlue')
    plt.legend(loc = 'lower right', prop=Font)
    plt.ylabel('True Positive Rate', Font)
    plt.xlabel('False Positive Rate', Font)
    plt.ylim([0.0, 1.05])    
    plt.xlim([0.0, 1.0])    
    plt.tick_params(labelsize=12)
    plt.savefig('ROC curve(respective research).tiff', dpi=600, bbox_inches = 'tight')
    plt.show()
    print('-----------------------------------------')   
    end = datetime.now()
    print((end-start))
    
    # draw PR curves
    Font={'size':12, 'family':'Times New Roman'}  
    plt.plot(recall_1_resp, precision_1_resp, 'b', label = 'miES (AUPR = %0.3f)' % AUPR_final_1_resp, color='Red')
    plt.plot(recall_2_resp, precision_2_resp, 'b', label = 'PESM (AUPR = %0.3f)' % AUPR_final_2_resp, color='k')
    plt.plot(recall_3_resp, precision_3_resp, 'b', label = 'RFEM (AUPR = %0.3f)' % AUPR_final_3_resp, color='RoyalBlue')
    plt.legend(loc = 'lower right', prop=Font)  
    plt.xlabel('Recall', Font)    
    plt.ylabel('Precision', Font)       
    plt.ylim([0.0, 1.05])    
    plt.xlim([0.0, 1.0])     
    plt.tick_params(labelsize=12)     
    plt.savefig('PR curve(respective research).tiff', dpi=600, bbox_inches = 'tight')
    plt.show()
    print('--------------------------------------------------------------')   
    end = datetime.now()
    print((end-start))






