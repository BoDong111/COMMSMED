from xgboost import XGBClassifier
import shap
import pandas
import sklearn
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
#from sklearn.model_selection import train_test_split

#X_train,X_test,y_train,y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
#print("X_train1")
#print(X_train)
#print(y_train)
#print(X_test)

#mydf=pandas.read_csv('full_data_12_19_new_CHKK.csv')
mydf=pandas.read_csv('data_0930_chkk_noncluster.csv')
#mydf=pandas.read_csv('covid_final_data1104.csv')
mydf=mydf.fillna(0)

df1 = mydf.iloc[:,1:-1]
y = mydf.iloc[:,-1]
#print('df')
#print(df)
#ri
X_train, X_test, y_train, y_test = train_test_split(df1, y, test_size=0.2)

###array=df.values
###X_train=array[0:120000,0:-1]
###y_train=array[0:120000,-1]
###X_test=array[120001:149939,0:-1]
###y_test=array[120001:149939,-1]
#shap.initjs()
#X,y = shap.datasets.boston()
#print(X)
#model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
#model.fit(X, y)
# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
#explainer = shap.KernelExplainer(y,X, link="logit")

#shap_values = explainer.shap_values(X)
#shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
#shap.force_plot(explainer.expected_value, shap_values, X)
#print(shap.__version__)

#X = np.array([1, 2, 3]).reshape(-1, 1)
#y = [0, 1, 1]

#rf = MLPRegressor()
#rf.fit(X_train, y_train)
#explainer = shap.KernelExplainer(rf.predict, X_train)
#print(explainer.shap_values(X_train))
##X_train=X_train.values
##y_train=y_train.values
##X_test=X_test.values
print('xgboost')
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
print(precision_recall_fscore_support(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(balanced_accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

'''
decision tree model
'''

print('decision tree')
clf = tree.DecisionTreeClassifier()
dtmodel= clf.fit(X_train, y_train)
y_pred_dt = dtmodel.predict(X_test)
print(precision_recall_fscore_support(y_test, y_pred_dt, average='macro'))
print(accuracy_score(y_test, y_pred_dt))
print(balanced_accuracy_score(y_test, y_pred_dt))

'''
svm model
'''

print('svm')
clf = svm.SVC()
svmmodel= clf.fit(X_train, y_train)
y_pred_svm = svmmodel.predict(X_test)
print(precision_recall_fscore_support(y_test, y_pred_svm, average='macro', zero_division=0))
print(accuracy_score(y_test, y_pred_svm))
print(balanced_accuracy_score(y_test, y_pred_svm))
'''
knn model
'''

print('knn')
clf = KNeighborsClassifier(n_neighbors=3)
knnmodel= clf.fit(X_train, y_train)
y_pred_knn = knnmodel.predict(X_test)
print(precision_recall_fscore_support(y_test, y_pred_knn, average='macro'))
print(accuracy_score(y_test, y_pred_knn))
print(balanced_accuracy_score(y_test, y_pred_knn))
'''
random forest model
'''

print('random forest')
clf = RandomForestClassifier(n_estimators=6)
rfmodel= clf.fit(X_train, y_train)
y_pred_rf = rfmodel.predict(X_test)
print(precision_recall_fscore_support(y_test, y_pred_rf, average='macro', zero_division=0))
print(accuracy_score(y_test, y_pred_rf))
print(balanced_accuracy_score(y_test, y_pred_rf))
'''
nn model
'''

print('nn')
clf = mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000)
nnmodel= clf.fit(X_train, y_train)
y_pred_nn = nnmodel.predict(X_test)
print(accuracy_score(y_test, y_pred_nn))
print(balanced_accuracy_score(y_test, y_pred_nn))
print(precision_recall_fscore_support(y_test, y_pred_nn, average='macro', zero_division=0))


#y_pred_svm = svm.predict(X_test)
#print(precision_recall_fscore_support(y_test, y_pred_svm, average='macro'))

# use Kernel SHAP to explain test set predictions

###explainer = shap.KernelExplainer(model.predict_proba, X_train, link="logit")
###shap_values = explainer.shap_values(X_test, nsamples=100)
###print(shap_values)
#newpd=pandas.DataFrame(shap_values)
#newpd.to_csv('shapvalues.csv', index=False, header=False)
#X=X.DataFrame(X)
# plot the SHAP values for the Setosa output of the first instance
#shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_test[0], link="logit")
#plt.savefig('test.png')
# visualize the training set predictions
#shap.force_plot(explainer.expected_value[0], shap_values[0], X_test, link="logit")
#plt.savefig('trainingset.png')

##data = pandas.DataFrame(X_train, columns=['Illiteracy','Population without health services',
##                                           'Houses with floor of dirt','Houses without toilet','Houses without water pipelines',
##                                           'Houses without sewage','Houses without electricity', 'Index of Social backwardness',
##                                           'Mean max temp','Mean min temp','Mean temp',
##                                           'Mean mm/day','Max mm/day','Min mm/day','Altitude'])
##print('data')
##print(data)
##shap.summary_plot(shap_values, data)
#newpd=pandas.DataFrame(shap_values)
##plt.savefig('summary09101.png')