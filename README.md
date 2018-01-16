# vehicle-detection-and-tracking

tips to finish the project:  
Parameter tuning for svm. use gridSearchCV  
example: 
``` python
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(iris.data, iris.target)
```
to access the paremeters, use: `clf.best_params_`

