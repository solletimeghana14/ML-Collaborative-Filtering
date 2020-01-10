import os
import numpy as np
import string
import pandas as  pd
import math
import random
import sys

def Correlation(Mean_active_user,Mean_database_user,data_correlate):
    sum1=0
    sum2=0
    sum3=0
    weight=0
    for j in data_correlate.columns:
            va=t.at[data_correlate.index[0],j]-Mean_active_user
            vc=t.at[data_correlate.index[1],j]-Mean_database_user
            sum1=sum1+(va*vc)
            sum2=sum2+(va*va)
            sum3=sum3+(vc*vc)
            
    numerator=sum1
    denominator=math.sqrt(sum2*sum3)
    if numerator==0 or denominator==0:
        weight=1
    else:
        weight=float(numerator)/denominator                
    return weight

def Prediction(train,unique_test_user_id,train_data_mean,test_movie_id,test_user_id,test_rating_id):
    MAE=0
    MSE=0
    Weights_Matrix=pd.DataFrame(columns=unique_test_user_id,index=train.index)
    for i in range(len(test_movie_id)):
        if i==0:
            database_users=train[train[test_movie_id[i]].notnull()]
        else:    
            if i!=0 and test_movie_id[i]!=test_movie_id[i-1]:
                database_users=train[train[test_movie_id[i]].notnull()]
            
        predict=train_data_mean[test_user_id[i]]
        sum_f=0
        k=0
        c=0
        d=0
        for j in database_users.index:
            data_correlate=database_users.loc[[test_user_id[i],j]].dropna(axis=1)
            if str(Weights_Matrix.at[j,test_user_id[i]])=='nan':
                weight=Correlation(train_data_mean[test_user_id[i]],train_data_mean[j],data_correlate)
                Weights_Matrix.at[j,test_user_id[i]]=weight
                c=c+1
            else:
                weight=Weights_Matrix.at[j,test_user_id[i]]
                d=d+1
               
            sum_f=sum_f+weight*(database_users.at[j,test_movie_id[i]]-train_data_mean[j])
            k=k+weight    
        P=predict+float(1/k)*sum_f
        MAE=MAE+abs(P-test_rating_id[i])
        MSE=MSE+((P-test_rating_id[i])*(P-test_rating_id[i]))
    M=float(MAE)/len(test_movie_id)
    E=math.sqrt(MSE/len(test_movie_id))
    return M,E

if __name__ == '__main__':
     arg_list = sys.argv
    
    path1 =str(arg_list[1])
    path2 = str(arg_list[2])
    train1=pd.read_csv(path1,header=None)
    header=['MovieIds','UserIds','Rating']
    train1.columns=header
    train=train1.pivot(index ="UserIds",columns ="MovieIds",values ="Rating") 
    test=pd.read_csv(path2,header=None)
    header=['MovieIds','UserIds','Rating']
    test.columns=header
    test_movie_id=np.array(test['MovieIds'])
    test_user_id=np.array(test['UserIds'])
    test_rating_id=np.array(test['Rating'])
    unique_test_user_id=np.unique(test_user_id)
    train_data_mean=train.mean(axis=1)
    Mean_Absolute_Error,Mean_Square_Error=Prediction(train,unique_test_user_id,train_data_mean,test_movie_id,test_user_id,test_rating_id)
    print('Mean Absolute Error:',Mean_Absolute_Error)
    print('Mean Square Error:',Mean_Square_Error)
