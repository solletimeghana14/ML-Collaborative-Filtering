import os
import numpy as np
import string
import pandas as  pd
import math
import random
import sys


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
    train_data_mean=train.mean(axis=1)
    train_arr=train.values
    train_unique_user_id=np.array(train.index)
    data_train_mean=np.nanmean(train_arr,axis=1,dtype=float)
    train_active_arr=np.zeros(train_arr.shape,dtype=float)
    train_other_arr=np.zeros(train_arr.shape,dtype=float)
    for i in range(len(train_unique_user_id)):
        x=train_arr[i,:]-data_train_mean[i]
        train_active_arr[i,:]=np.nan_to_num(x)
        train_other_arr[i,:]=np.nan_to_num(x)
    numerator=np.dot(train_active_arr,train_other_arr.T) 
    x2=np.square(train_active_arr)
    y2=np.square(train_other_arr)
    W=np.sum(x2,axis=1)
    Z=np.sum(y2,axis=1)
    l=W.reshape(28978,1)
    m=Z.reshape(1,28978)
    denominator=np.sqrt(np.dot(l,m))
    D=np.divide(numerator,denominator,out=np.ones_like(numerator),where=denominator!=0)
    C=np.sum(D,axis=1)
    K=C.reshape(28978,1)
    V=np.dot(D,train_active_arr)
    Mean=data_train_mean.reshape(28978,1)
    R=np.divide(V,K,out=np.ones_like(V),where=K!=0)
    P=np.add(R,Mean)
    Predict=pd.DataFrame(P,index=train.index,columns=train.columns)
    MAE=0
    MSE=0
    for i in range(len(test_movie_id)):
        error=Predict.at[test_user_id[i],test_movie_id[i]]-test_rating_id[i]
        
       
        MAE=MAE+abs(error)
        MSE=MSE+(error*error)
        
    M=float(MAE)/len(test_movie_id)
    E=math.sqrt(float(MSE)/len(test_movie_id))
    print('Mean Absolute Error:',M)
    print('Root Mean Square Error:',E)
        
