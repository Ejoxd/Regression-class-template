import pandas as pd
import numpy as np


class Regressions :
    def __init__ (self, x, y): 
        self.x = x.values
        self.y = y.values
        self.y = self.y.reshape(-1,1)
        self.x_train = 0
        self.y_train = 0
        self.x_test = 0
        self.y_test = 0
        self.y_pred = 0
        self.check = False

    def split_datas(self, test_size) :
        from sklearn.model_selection import train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size = test_size, random_state=0)
        return self.x_train, self.x_test, self.y_train, self.y_test

    def MLR(self, test_size) : 
        # IMPORT AND SPLIT
        from sklearn.linear_model import LinearRegression
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_datas(test_size)
        # FIT DATA
        lr = LinearRegression()
        lr.fit(self.x_train, self.y_train)
        # PREDICT
        self.y_pred = lr.predict(self.x_test)
        self.check = True

    def PLR(self, test_size) : 
        # CHECK ANY REGRESSION BE IMPLEMENTED BEFORE 
        self.check_implemantation()
        # IMPORT PLR
        from sklearn.preprocessing import PolynomialFeatures
        # DATA SPLITTING
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_datas(test_size)
        # FIT DATA 
        plr = PolynomialFeatures(degree=2)
        x_poly = plr.fit_transform(self.x_train)
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(x_poly, self.y_train)
        # PREDICT
        self.y_pred = lr.predict(plr.fit_transform(self.x_test))

    
    def SVR(self, test_size) :
        # CHECK ANY REGRESSION BE IMPLEMENTED BEFORE 
        self.check_implemantation()
        # IMPORT SVR
        from sklearn.svm import SVR        
        # SCALING
        from sklearn.preprocessing import StandardScaler
        sc_x = StandardScaler()
        sc_y = StandardScaler() 
      
        # SPLITTIN DATA
        self.y = self.y.reshape(len(self.y), 1)
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_datas(test_size)
        
        self.x_train = sc_x.fit_transform(self.x_train)
        self.y_train = sc_y.fit_transform(self.y_train)
        
        # SVR
        svr = SVR(kernel='rbf')
        svr.fit(self.x_train, self.y_train)

        # BE IMPLEMENTED CHECK
        self.check = True

        # INVERSE SCALED DATA TO READ
        self.y_pred = sc_y.inverse_transform(svr.predict(sc_x.fit_transform(self.x_test)))

    def DTR(self, test_size) : 
        # CHECK ANY REGRESSION BE IMPLEMENTED BEFORE 
        self.check_implemantation()
        # IMPORT DECISION TREE
        from sklearn.tree import DecisionTreeRegressor
        dtr = DecisionTreeRegressor(random_state=0)
        # SPLIT DATAS 
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_datas(test_size)
        # FIT DATA
        dtr.fit(self.x_train, self.y_train)
        # CHECK CONTROL
        self.check = True
        # PREDICT
        self.y_pred = dtr.predict(self.x_test) 

    def RFR(self, test_size): 
        # CHECK ANY REGRESSION BE IMPLEMENTED BEFORE 
        self.check_implemantation()
        # IMPORT RANDOM FOREST
        from sklearn.ensemble import RandomForestRegressor
        rfr = RandomForestRegressor(n_estimators=20, random_state=0)
        # SPLIT DATAS 
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_datas(test_size)
        # FIT DATAS
        rfr.fit(self.x_train, self.y_train)
        # CHECK CONTROL 
        self.check = True
        # PREDICT
        self.y_pred = rfr.predict(self.x_test)        
   
    def score(self) :
        # R2 SCORE 
        from sklearn.metrics import r2_score
        score = r2_score(self.y_test, self.y_pred)
        return score

    def check_implemantation(self) : 
        if(self.check != False) :
            self.x_train = 0
            self.y_train = 0
            self.x_test = 0
            self.y_test = 0
            self.y_pred = 0
        else :
            pass
# READ DATA                
data = pd.read_csv("ENTER_THE_DATASET_NAME.csv")
x = data.iloc[:,:3]
y = data.iloc[:,-1]

# CREATE MODEL
regression_model = Regressions(x,y)

# MLR
regression_model.MLR(0.2)
score = regression_model.score()
print("MLR score", score)

# PLR
regression_model.PLR(0.2)
score = regression_model.score()
print("PLR score", score)

#SVR
regression_model.SVR(0.2)
score = regression_model.score()
print("SVR score", score)

# DTR
regression_model.DTR(0.2)
score = regression_model.score()
print("DTR score", score)

# RFR
regression_model.RFR(0.2)
score = regression_model.score()
print("RFR score", score)