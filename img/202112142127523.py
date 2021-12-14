import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split

RawData = pd.read_csv("http://gstorage.acbs.top/img/202112142119723.csv")
y = RawData["target"].to_numpy()
y.shape

xData = RawData.copy()
del xData["target"]
X = xData.to_numpy()
X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree, svm, naive_bayes, neighbors
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier


clfs = []
# add SVM （12s）
for ckernal in ["linear", "poly", "rbf"]:
    for cC in [10**(i-4) for i in range(8)]:
        name = f"SVM kernal={str(ckernal)} c={str(cC)}"
        clfs.append([name, "SVM", svm.SVC(kernel=ckernal, C=cC), 0])

# add 决策树
for cCriterion in ["gini", "entropy"]:
    for cmax_depth in range(5,21):
        for csplitter in ["best", "random"]:
            name = f"SVM criterion={cCriterion}, max_depth={cmax_depth}"
            clfs.append([name, "decision_tree", tree.DecisionTreeClassifier(criterion=cCriterion,
                                                                        max_depth=cmax_depth,
                                                                        splitter=csplitter,
                                                                        ), 0])

# add beiyesi
for cvarsmooth in [10**(i-9) for i in range(10)]:
    name = f"naive_gaussian varsmooth={str(cvarsmooth)}"
    clfs.append([name, "naive_gaussian", naive_bayes.GaussianNB(var_smoothing=cvarsmooth), 0])

# add
for calpha in np.arange(0, 1 ,0.05):
    name = f"naive_mul alpha={str(calpha)}"
    clfs.append([name, "naive_mul", naive_bayes.MultinomialNB(alpha=calpha), 0])

# add k近邻
for cnabigor in range(1,10):
    for cweight in ["uniform", "distance", "distance"]:
        if cweight == "distance":
            for cp in range(1,7):
                name = f"knn weight={cweight} nabor={str(cnabigor)} p={str(cp)}"
                clfs.append([name, "K_neighbor", neighbors.KNeighborsClassifier(weights=cweight, n_neighbors=cnabigor, p=cp), 0])
        else:
            name = f"knn weight={cweight} nabor={str(cnabigor)}"
            clfs.append([name, "K_neighbor", neighbors.KNeighborsClassifier(weights=cweight, n_neighbors=cnabigor), 0])

# add 随机森林
for cmax_depth in range(10, 100):
    for cminsample in range(2,5):
        for cminsampleleaf in range(1,3):
            name = f"随机森林 maxdepth={cmax_depth} min_samples_split={cminsample} min_samples_leaf={cminsampleleaf}"
            clfs.append([name, "random_forest", RandomForestClassifier(
                                        n_estimators=50,
                                        max_depth=cmax_depth,
                                        min_samples_split=cminsample,
                                        min_samples_leaf=cminsampleleaf,
                                    ), 0])

# add adaboost
for calgorithm in ["SAMME", "SAMME.R"]:
    for cn_estimators in range(20, 70, 4):
        for clr in [0.01, 0.1, 0.2, 0.5, 0.9, 1, 1.2]:
            name = f"adaboost calgorithm={calgorithm} estimators={str(cn_estimators)} lr={clr}"
            clfs.append([name, "adaboost", AdaBoostClassifier(
                            algorithm=calgorithm,
                            n_estimators=cn_estimators,
                            learning_rate=clr
                        ),0])

# train and clear
def scoreit(clist):
    clist[2].fit(X_train,y_train.ravel())
    clist[3] = clist[2].score(X_test,y_test.ravel())
[scoreit(c) for c in clfs]
clfs.sort(reverse=True, key=lambda x:x[3])
print(clfs)
pdCLFS = pd.DataFrame(clfs)
clfs = []

import random
def random_color():
     colors1 = '0123456789ABCDEF'
     num = "#"
     for i in range(6):
         num += random.choice(colors1)
     return num

    

matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号

modelGroup = set(pdCLFS[1])
colorDict = {i:random_color() for i in modelGroup}
colorList = [colorDict[i] for i in pdCLFS[1]]
plt.figure(figsize=(50,5))
plt.bar(range(len(pdCLFS)), pdCLFS[3]-0.4, width=0.5, color=colorList)
plt.xticks(range(len(pdCLFS)),pdCLFS[0], rotation=90, size=3)
plt.gcf().subplots_adjust(bottom=0.3)
plt.savefig("fig2.jpg",dpi=1000)