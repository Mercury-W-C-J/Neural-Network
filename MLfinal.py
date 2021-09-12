from tqdm import trange
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import itertools
import random
import math
import pandas as pd
import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
import itertools
#Importing tqdm for the progress bar
from tqdm import tnrange, tqdm_notebook

random.seed(5)
def sigmoid(x, derive=False): 
    if derive: 
        return sigmoid(x) * (1 - sigmoid(x)) 
    return 1 / (1 + np.exp(-x)) 

def relu(x, derive=False): 
    if derive: 
        return np.where(x > 0, 1, 0) 
    return np.where(x > 0, x, 0) 

def tanh(x, derive=False): 
    if derive: 
        return 1 - np.power(tanh(x), 2) 
    return 2 * sigmoid(np.multiply(2, x)) - 1 

     
class NeuralNetwork(object):
    def __init__(self,activation):
        self.activation=activation
        self.character=pd.DataFrame({'nodes':[],'activation':[]})
        self.weight=[]
        self.before_act=[]
        self.after_act=[]
        self.imput=[]
        self.y_true=0
        self.y_pred=0
        self.bias_weight=[]
        self.LR=0
        self.xini=0
        self.result=pd.DataFrame({'churning':[]})
        self.lamda=0
        self.lastone=0
        self.optimal=0
        self.optimal_weight=[]
        self.count=0
        self.loop=pd.DataFrame({'Accuracy':[],'Precision':[], 'Recall':[], 'TNR':[]})
        self.test=pd.DataFrame({'Accuracy':[],'Precision':[], 'Recall':[], 'TNR':[]})
        self.train_error=pd.DataFrame({'Accuracy':[],'Precision':[], 'Recall':[], 'TNR':[]})
        self.tolerated_decrease=0

     
    def add_hidden_layer(self,nodes,activation_function):
        o=pd.DataFrame({'nodes':[nodes],'activation':[activation_function]})
        self.character=self.character.append(o)
        
    def first(self):
        xini=self.xini                                           #need to genralize
        for i in range(len(self.character)+1):
            if i==0:
                self.weight.append(0.1*np.random.random((int(self.character.iloc[i,0]),xini)))
                self.bias_weight.append(0.1*np.random.uniform(-1,1,int(self.character.iloc[i,0])))
            else:
                if i==len(self.character):
                    self.weight.append(0.1*np.random.random((1,int(self.character.iloc[i-1,0]))))
                    self.bias_weight.append(0.1*np.random.uniform(-1,1,1))
                else:
                    self.weight.append(0.1*np.random.random((int(self.character.iloc[i,0]),int(self.character.iloc[i-1,0]))))
                    self.bias_weight.append(0.1*np.random.uniform(-1,1,int(self.character.iloc[i,0])))
                    
    def forward(self,x,y):

        self.y_true=y
        self.input=x
        self.before_act=[]
        self.after_act=[]
        
        for i in range(len(self.weight)):
            before=[]
            after=[]
            if i==0:
                for j in range(len(self.weight[i])):
                    sum=np.dot(x,self.weight[i][j])+self.bias_weight[i][j]
                    before.append(sum)
                    if self.character.iloc[i,1]=='sigmoid':
                        aftersum=sigmoid(sum)
                    if self.character.iloc[i,1]=='relu':  
                        aftersum=relu(sum)
                    if self.character.iloc[i,1]=='tanh':  
                        aftersum=tanh(sum)
                    after.append(aftersum)
                self.before_act.append(before)
                self.after_act.append(after)
            elif i==len(self.weight)-1:
                for j in range(len(self.weight[i])):
                    sum=np.dot(self.after_act[i-1],self.weight[i][j])+self.bias_weight[i][j]
                    before.append(sum)
                    if self.activation=='sigmoid':
                        aftersum=sigmoid(sum)
                    if self.activation=='relu':  
                        aftersum=relu(sum)
                    if self.activation=='tanh':  
                        aftersum=tanh(sum)
                    after.append(aftersum)
                self.before_act.append(before)
                self.after_act.append(after)
            else:
                for j in range(len(self.weight[i])):
                    sum=np.dot(self.after_act[i-1],self.weight[i][j])+self.bias_weight[i][j]
                    before.append(sum)
                    if self.character.iloc[i,1]=='sigmoid':
                        aftersum=sigmoid(sum)
                    if self.character.iloc[i,1]=='relu':  
                        aftersum=relu(sum)
                    if self.character.iloc[i,1]=='tanh':  
                        aftersum=tanh(sum)
                    after.append(aftersum)   
                self.before_act.append(before)
                self.after_act.append(after)
        self.y_pred=self.after_act[-1][0]

    def bp(self):
        LR=self.LR
        error=[]
        e=0
        error_y=(self.y_true-self.y_pred)/self.y_pred*(1-self.y_pred) #cross entropy
        #self.y_true-self.y_pred
        #error_y=1/2*(self.y_true-self.y_pred)*(self.y_true-self.y_pred)  square loss
        copy_weight=self.weight
        copy_bias_weight=self.bias_weight

        for i in range(len(self.weight)):
            if i==0:
                for j in range(len(self.weight[-1][0])):
                    if self.activation == 'sigmoid': 
                        e=sigmoid(self.before_act[-1][0],True)*(error_y)
                        #print(copy_weight)
                        copy_weight[-1][0][j]=copy_weight[-1][0][j]+LR*e*self.after_act[-2][j]
                    elif self.activation == 'relu': 
                        e=relu(self.before_act[-1][0],True)*(error_y)
                        copy_weight[-1][0][j]=copy_weight[-1][0][j]+LR*e*self.after_act[-2][j]
                    elif self.activation == 'tanh': 
                        e=tanh(self.before_act[-1][0],True)*(error_y)
                        copy_weight[-1][0][j]=copy_weight[-1][0][j]+LR*e*self.after_act[-2][j]
                    else:
                        print('Plz enter the proper activation function!')
                    error.append(e)
                    copy_bias_weight[-i-1][0]+=LR*e
            
            elif i==(len(self.weight)-1):
                weighted_error=[]
                total_weighted_error=[]
                for k in range(len(self.weight[-i-1])):     # k是本轮调整权值输出节点个数 计算第k个加权值 未乘导数
                    for l in range(len(self.weight[-i])):      # l 是本轮调整权重下一层输出节点个数
                        weighted_error.append(error[l]*self.weight[-i][l][k])
                    total_weighted_error.append(sum(weighted_error))
                    weighted_error=[]
                error=[]
                
                for k in range(len(self.weight[-i-1])):    #计算本轮调整权值输出节点error
                    if self.character.iloc[-i][1] == 'sigmoid': 
                        error.append(sigmoid(self.before_act[-i-1][k],True)*total_weighted_error[k])
                    elif self.character.iloc[-i][1] == 'relu': 
                        error.append(relu(self.before_act[-i-1][k],True)*total_weighted_error[k])
                    elif self.character.iloc[-i][1] == 'tanh': 
                        error.append(tanh(self.before_act[-i-1][k],True)*total_weighted_error[k])
                    else:
                        print('Plz enter the proper activation function!')
                
                for j in range(len(self.weight[-i-1])):    #j是调整权值矩阵行数 即输出节点个数
                    for k in range(len(self.weight[-i-1][0])): #k是调整权值矩阵列数 即输入节点个数
                        copy_weight[-i-1][j][k]+=LR*self.input[k]*error[j]
                for k in range(len(self.weight[-i-1])):
                    copy_bias_weight[-i-1][k]+=LR*error[k]
            else:
                weighted_error=[]
                total_weighted_error=[]
                for k in range(len(self.weight[-i-1])):     # k是本轮调整权值输出节点个数 计算第k个加权值 未乘导数
                    for l in range(len(self.weight[-i])):      # l 是本轮调整权重下一层输出节点个数
                        weighted_error.append(error[l]*self.weight[-i][l][k])
                    total_weighted_error.append(sum(weighted_error))
                    weighted_error=[]
                error=[]
                
                for k in range(len(self.weight[-i-1])):    #计算本轮调整权值输出节点error
                    if self.character.iloc[-i][1] == 'sigmoid': 
                        error.append(sigmoid(self.before_act[-i-1][k],True)*total_weighted_error[k])
                    elif self.character.iloc[-i][1] == 'relu': 
                        error.append(relu(self.before_act[-i-1][k],True)*total_weighted_error[k])
                    elif self.character.iloc[-i][1] == 'tanh': 
                        error.append(tanh(self.before_act[-i-1][k],True)*total_weighted_error[k])
                    else:
                        print('Plz enter the proper activation function!')
                
                for j in range(len(self.weight[-i-1])):    #j是调整权值矩阵行数 即输出节点个数
                    for k in range(len(self.weight[-i-1][0])): #k是调整权值矩阵列数 即输入节点个数
                        copy_weight[-i-1][j][k]+=LR*self.after_act[-i-2][k]*error[j]
                for k in range(len(self.weight[-i-1])):    #更新bias_weight
                    copy_bias_weight[-i-1][k]+=LR*error[k]
            self.weight=copy_weight
            self.bias_weight=copy_bias_weight
            
    def train(self,x_train,y_train,LR,iteration):
        # x_train=x_train.to_numpy()
        y_train=y_train.to_numpy()
        self.weight=[]
        self.bias_weight=[]
        self.LR=LR
        self.xini=x_train.shape[1]
        self.first()
        for i in range(iteration):
            for j in range(len(x_train)):
                self.forward(x_train[j],y_train[j])
                self.bp()
                
    def predict(self,x,y):
        # x=x.to_numpy()
        y=y.to_numpy()
        self.result=pd.DataFrame({'churning':[]})
        for j in range(len(x)):
            self.forward(x[j],y[j])
            tem=pd.DataFrame({'churning':[self.y_pred]})
            self.result=self.result.append(tem,ignore_index=True)
            
    def score(self,threshold,y):
        pred_threshold=pd.DataFrame({'pred_threshold':[]})
        TP=0
        TN=0
        for i in range (len(y)):
            if self.result.iloc[i,0]>=threshold:
                tem=pd.DataFrame({'pred_threshold':[1]})
                pred_threshold=pred_threshold.append(tem,ignore_index=True)
            else:
                tem=pd.DataFrame({'pred_threshold':[0]})
                pred_threshold=pred_threshold.append(tem,ignore_index=True)
        for j in range (len(y)):
            if pred_threshold.iloc[j,0]==y.iloc[j,0]:
                if y.iloc[j,0]==1:
                    TP+=1
                else :
                    TN+=1
        True_churn=int(len(y)-y.sum())
        True_exist=int(y.sum())
        Pred_True_churn=int(len(pred_threshold)-pred_threshold.sum()) 
        Pred_True_exist=int(pred_threshold.sum())  
        accuracy=(TP+TN)/len(y)
        precision=TP/Pred_True_exist
        recall=TP/True_exist
        TNR=TN/True_churn
        Score=pd.DataFrame({'Accuracy':[accuracy],'Precision':[precision],'Recall':[recall],'TNR':[TNR]})
        return Score
    
    def train_loop(self,x_train_valid,y_train_valid,x_valid,y_valid):
        self.count+=1
        tolerated_decrease=0.02
        for j in trange(len(x_train_valid)):
            self.forward(x_train_valid[j],y_train_valid[j])
            self.bp()
        self.predict2(x_valid,y_valid)
        validation_error=self.score2(0.5,y_valid)
        self.loop=self.loop.append(validation_error,ignore_index=True)
        print(validation_error)
        if self.optimal < validation_error.iloc[0,3]:
            self.optimal=validation_error.iloc[0,3]
            self.optimal_weight=self.weight
        if validation_error.iloc[0,3]<self.optimal-tolerated_decrease:
            self.weight=self.optimal_weight
            return 0
        else:
            self.train_loop(x_train_valid,y_train_valid,x_valid,y_valid)          
            
    def train_early_stop(self,x_train,y_train,LR,tolerated_decrease):
        self.tolerated_decrease=tolerated_decrease
        if type(x_train) !=np.ndarray: 
            x_train=x_train.to_numpy()
        if type(y_train) !=np.ndarray:         
            y_train=y_train.to_numpy()
        self.weight=[]
        self.bias_weight=[]
        self.LR=LR
        self.xini=x_train.shape[1]
        self.first()
        X_train_valid, X_valid, y_train_valid, y_valid = train_test_split(x_train,y_train, test_size=0.15)
        self.train_loop(X_train_valid, y_train_valid, X_valid, y_valid)
        for i in range(len (x_train)):
            self.forward(x_train[i],y_train[i])
            self.bp()
            
    def predict2(self,x,y):
        # x=x.to_numpy()         
        if type(y)!=np.ndarray:
            y=y.to_numpy()
        self.result=pd.DataFrame({'churning':[]})
        for j in range(len(x)):
            self.forward(x[j],y[j])
            tem=pd.DataFrame({'churning':[self.y_pred]})
            self.result=self.result.append(tem,ignore_index=True)
    
    def score2(self,threshold,y):
        pred_threshold=pd.DataFrame({'pred_threshold':[]})
        TP=0
        TN=0
        for i in range (len(y)):
            if self.result.iloc[i,0]>=threshold:
                tem=pd.DataFrame({'pred_threshold':[1]})
                pred_threshold=pred_threshold.append(tem,ignore_index=True)
            else:
                tem=pd.DataFrame({'pred_threshold':[0]})
                pred_threshold=pred_threshold.append(tem,ignore_index=True)
        for j in range (len(y)):
            if pred_threshold.iloc[j,0]==y[j]:
                if y[j]==1:
                    TP+=1
                else :
                    TN+=1
        True_churn=int(len(y)-y.sum())
        True_exist=int(y.sum())
        Pred_True_churn=int(len(pred_threshold)-pred_threshold.sum()) 
        Pred_True_exist=int(pred_threshold.sum())  
        accuracy=(TP+TN)/len(y)
        precision=TP/Pred_True_exist
        recall=TP/True_exist
        TNR=TN/True_churn
        Score=pd.DataFrame({'Accuracy':[accuracy],'Precision':[precision],'Recall':[recall],'TNR':[TNR]})
        return Score        
    
    
    


class randomforest(object):
    def __init__(self):
        self.predict_sum=pd.DataFrame({'':[]})
        self.predict=pd.DataFrame({'':[]})

    def random(self,X,y,X_test,y_test,feature,number,node):
        X=pd.DataFrame(X)
        
        X.index=range(len(X))
        y.index=range(len(y))
        Xy=pd.concat([X,y],axis=1)
        Xy_boot=Xy.iloc[np.random.randint(len(Xy), size=number*len(Xy))]
        Xy_boot.index=range(len(Xy_boot))
        X_train=Xy_boot.iloc[:,:-1]
        y_train=Xy_boot.iloc[:,-1]
        y_train=pd.DataFrame(y_train)
        
        
        X_test=pd.DataFrame(X_test)
        n=0
        for combo in itertools.combinations(X.columns,feature):
            X_c = X_train[list(combo)].sample(len(Xy)) 
            y_train_train=y_train.iloc[X_c.index,:]
            y_train_train.index=range(len(y_train_train))
            X_c_test = X_test[list(combo)] 
            X_c.index=range(len(X_c))
            X_c_test.index=range(len(X_c_test))
            X_c=X_c.to_numpy()
            X_c_test=X_c_test.to_numpy()
            b=NeuralNetwork('sigmoid')
            b.add_hidden_layer(node, 'relu')
            b.train(X_c,y_train_train,0.01,10)
            b.predict(X_c_test,y_test)
            
            n=n+1
            if n==1:
                self.predict=b.result
            elif n>number:
                break
            else :
                self.predict=pd.concat([b.result,self.predict],axis=1)
        self.predict_sum=self.predict.mean(axis=1)
    
    
            
    def randomscore(self,threshold,y):
        pred_threshold=pd.DataFrame({'pred_threshold':[]})
        TP=0
        TN=0
        for i in range (len(y)):
            if self.predict_sum[i]>=threshold:
                tem=pd.DataFrame({'pred_threshold':[1]})
                pred_threshold=pred_threshold.append(tem,ignore_index=True)
            else:
                tem=pd.DataFrame({'pred_threshold':[0]})
                pred_threshold=pred_threshold.append(tem,ignore_index=True)
        for j in range (len(y)):
            if pred_threshold.iloc[j,0]==y.iloc[j,0]:
                if y.iloc[j,0]==1:
                    TP+=1
                else :
                    TN+=1
        True_churn=int(len(y)-y.sum())
        True_exist=int(y.sum())
        Pred_True_churn=int(len(pred_threshold)-pred_threshold.sum()) 
        Pred_True_exist=int(pred_threshold.sum())  
        accuracy=(TP+TN)/len(y)
        precision=TP/Pred_True_exist
        recall=TP/True_exist
        TNR=TN/True_churn
        RandomScore=pd.DataFrame({'Accuracy':[accuracy],'Precision':[precision],'Recall':[recall],'TNR':[TNR]})
        return RandomScore

df= pd.read_csv("BankChurners.csv")


df_dummy = pd.get_dummies(df, columns = ['Attrition_Flag','Gender','Education_Level','Marital_Status','Income_Category','Card_Category'],drop_first = True)

df_exist=df_dummy.loc[df_dummy['Attrition_Flag_Existing Customer']==1]
df_attrited=df_dummy.loc[df_dummy['Attrition_Flag_Existing Customer']==0]
df_attrited.index=range(len(df_attrited))
df_boot=df_attrited.iloc[np.random.randint(len(df_attrited), size=len(df_exist))]
df_experiment=pd.concat([df_boot,df_exist])

df=df_experiment

y= pd.DataFrame(df['Attrition_Flag_Existing Customer'], columns=["Attrition_Flag_Existing Customer"])
x=df.drop(['Attrition_Flag_Existing Customer','CLIENTNUM','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'],axis=1)
x_standard=x[['Customer_Age','Dependent_count','Months_on_book','Total_Relationship_Count','Months_Inactive_12_mon','Contacts_Count_12_mon','Credit_Limit','Total_Revolving_Bal','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1','Total_Trans_Amt','Total_Trans_Ct','Total_Ct_Chng_Q4_Q1','Avg_Utilization_Ratio']]
scaler_x, scaler_y = StandardScaler(), StandardScaler()
scaler_x.fit(x_standard)
x_standard=scaler_x.transform(x_standard)
x_dummy=x.drop(['Customer_Age','Dependent_count','Months_on_book','Total_Relationship_Count','Months_Inactive_12_mon','Contacts_Count_12_mon','Credit_Limit','Total_Revolving_Bal','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1','Total_Trans_Amt','Total_Trans_Ct','Total_Ct_Chng_Q4_Q1','Avg_Utilization_Ratio'],axis=1)
x_dummy=x_dummy.to_numpy()
x=np.hstack([x_standard,x_dummy])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
# l=randomforest()
# l.random(X_train, y_train, X_test_data, y_test_data, 10, 10,3)
# hahah_random=l.randomscore(0.5,y_test)

a=NeuralNetwork('sigmoid')
a.add_hidden_layer(3, 'relu')
a.add_hidden_layer(4,'relu')


a.train(X_train,y_train,0.01,5)
a.predict(X_test,y_test)
print(a.score(0.5,y_test))

a.train_early_stop(X_train,y_train,0.01,0.05)
a.predict(X_test,y_test)
print(a.score(0.5,y_test))

l=randomforest()
l.random(X_train, y_train, X_test, y_test, 10, 10,3)
hahah_random=l.randomscore(0.5,y_test)