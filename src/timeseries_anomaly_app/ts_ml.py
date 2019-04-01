mport os
from itertools import product                    # some useful functions
from dateutil.relativedelta import relativedelta # working with dates with style
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')
import configparser
import datetime
import smtplib
import sys


#Statistics and data manuplation
import numpy as np
import pandas as pd


#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, median_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.externals import joblib


def mean_absolute_percentage_error(target, prediction): 
        return np.mean(np.abs((target - prediction) / target)) * 100


def send_email(config_path, source, date):
    """
    Parameters:
    -----------
    config: str
         Configuration file
    
    Methods
    -------
    
    """

    if os.path.exists(config_path):
        cfg = configparser.ConfigParser()
        cfg.read(config_path)
    else:
        print("Config not found! Exiting!")
        sys.exit(1)
        
    
    host = cfg.get('email', "server")
    from_addr = cfg.get('email', "from_addr")
    to_addr_1 = cfg.get('email', "to_addr_1")
    to_addr_2 = cfg.get('email', "to_addr_2")
    username = cfg.get('email', "username")
    password = cfg.get('email', "password")
    
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%s")
    
    body_text= "There are anomaly in the folowing dates below on the {}\n{}".format(source, date)

    subject = 'Anomaly detected in {}'.format(source)
        
    to_addrs = [to_addr_1, to_addr_2]
    
    BODY = "\r\n".join((
           "Date: %s" % now,
           "From: %s" % from_addr,
           "To: %s" % to_addr_1,
            "CC: %s" %  to_addr_2,
           "Subject: %s" % subject,
           "",
           body_text
          ))
    try:
        server = smtplib.SMTP(host)
        server.login(username, password)
        server.sendmail(from_addr, to_addrs, BODY)
        server.quit()
        
    except SMTPException as e:
        print('we encountered error {}'.format(e))



def cat_average(data, interval_feature, target):
    '''
        Returns a dictionary where keys are unique categories of the interval_feature,
        and values are average over target variable
        Parameters:
        
        Results:
        -----------
        return dictionary 
    '''
    return dict(data.groupby(interval_feature)[target].mean())


def data_preparation(series, lag_start, lag_end, target_encoding=False, time_index='daily'):
    '''
        Parameters:
           ----------
            series: DataFrame
            dataframe with timeseries
           
            lag_start: int
                first step in time to slice target variable.
                For instance, lag_start is set to 1 which means that the model will see yesterday's values 
                to predict today if the dataframe has index of daily 
                
           
            lag_end: int
                Final step in time to splice target variable.
                For instance, lag_end is 4 which means that the model will see up to 4 days/hours back in time
                to predict today if the dataframe has index of daily
            
            target_encoding: boolean
            If True - add target averages to the dataset
                
    '''
    #make copy of the dataset/series
    time_data = pd.DataFrame(series.copy())
    time_data.columns = ['target']
    
    #create lags in time_data
    for item in range(lag_start, lag_end):
        time_data["lag_{}".format(item)]=time_data.target.shift(item)
    
    #Create datetime features
    #test_index = int(len(time_data.dropna())*(1-test_size))
        
    if time_index=='hour':
        time_data["hour"] = time_data.index.hour
        if target_encoding:
                time_data["hour_average"] = list(map(cat_average(time_data, 'hour', "target").get, time_data.hour))
            
        
    time_data["weekday"] = time_data.index.weekday
    if target_encoding:
        time_data['weekday_average'] = list(map(cat_average(time_data, 'weekday', "target").get, time_data.weekday))
            
       
    #train-test split the dataset
    target = time_data.dropna().target
    features = time_data.dropna().drop(['target'], axis=1)
        
    return  features, target


class TimeSeriesMl:
    '''
    Parameters:
    ----------
       series: pandas Series
       model: machine learning saved model
       lag_start: int
                first step in time to slice target variable.
                For instance, lag_start is set to 1 which means that the model will see yesterday's values 
                to predict today if the dataframe has index of daily 
        lag_end: int
                Final step in time to splice target variable.
                For instance, lag_end is 4 which means that the model will see up to 4 days/hours back in time
                to predict today if the dataframe has index of daily
        scale: float
        test_size: float
        target_encoding: boolan
        time_index: str
    '''

    def __init__(self, series, modelfile, lag_start, lag_end,  scale=1.96, test_size=0.3, target_encoding=False, time_index='daily'):
        self.series = series
        self.model = joblib.load(modelfile)
        #self.ts_cv = TimeSeriesSplit(n_splits=cv)
        self.lag_start = lag_start
        self.lag_end=lag_end
        #self.error_score=error_score   error_score="neg_mean_squared_error"
        self.scale=scale
        self.test_size=test_size
        self.target_encoding=target_encoding
        self.time_index=time_index


    def ts_train_test_split(self):
        """
        Return train-test split of the time series dataset
        """
        features, target = data_preparation(series=self.series, 
                                            lag_start=self.lag_start,
                                            lag_end=self.lag_end,
                                           target_encoding=self.target_encoding, 
                                           time_index=self.time_index)

        #create the index after which test set starts
        test_index = int(len(features)*(1-self.test_size))
        
        feature_train = features.iloc[:test_index]
        target_train = target.iloc[:test_index]
        feature_test = features.iloc[test_index:]
        target_test = target.iloc[test_index:]
        
        
        return feature_train, feature_test, target_train, target_test 
   
    
    
    def standardise(self, feature):
        '''
         Return Standardising the dataset
        '''
        
        scaler = StandardScaler()
        
        feature_scaled = scaler.fit_transform(feature)

        return feature_scaled
    
    
    def apply_model(self):
        '''
        
        Return dataframe that contains  anomaly values if  it exists and send alert.
        '''

        features, target = data_preparation(series=self.series, 
                                            lag_start=self.lag_start,
                                            lag_end=self.lag_end,
                                           target_encoding=self.target_encoding, 
                                           time_index=self.time_index)
        
        feature_scaled=self.standardise(features)
        
        #cv = cross_val_score(self.model, feature_scaled, target, cv=self.ts_cv, scoring=self.error_score)
        
        #deviation = np.sqrt(cv.std())
        
    
        prediction = self.model.predict(feature_scaled)
        deviation = np.std(target- prediction)
        lower = prediction - (self.scale * deviation)
        upper = prediction + (self.scale * deviation)
    
        anomalies = np.array([np.NaN]*len(target))
        anomalies[target<lower] = target[target<lower]
        anomalies[target>upper] = target[target>upper]

    
        result = pd.DataFrame(target.index, columns=['target'])
        result['target'] = target
        result['prediction']=prediction
        result['anomalies']=anomalies

        result['lower']=lower
        result['upper']=upper

         #Check if there are anomaly data in the dataframe then trigger email
        col_name= anomalies.columns[0]
        
        if result['anomalies'].sum()>0:
            date = []
            for index, row in anomalies.iterrows():
                if row[col_name] is not None:
                        date.append(str(index))
           
            send_email(self.config_path, self.source, date)
    
        return result

    
    def test_plot_model(self):
        
        result = self.apply_model()

        plt.figure(figsize=(15,5))
        plt.plot(result['lower'], "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(result['upper'], "r--", alpha=0.5)
        plt.fill_between(x = result.index, y1=result['upper'], y2=result['lower'], alpha=0.2, color="grey")

        plt.plot(result['anomalies'], "o", markersize=10, label = "Anomalies")
        
        error = mean_absolute_percentage_error(result['prediction'], result['target'])
        plt.title("Mean absolute percentage error {0:.2f}%".format(error))
        #plt.legend(loc="best")
        #plt.tight_layout()
        #plt.grid(True)
        plt.plot(result['target'], "g", label="Actual values")
        plt.plot(result['prediction'], color="y", label="Prediction")
        plt.legend(loc="upper left")
        plt.grid(True)    
        
    
    def plot_coefficients(self):
       """
           Plots sorted coefficient values of the model
       """
       feature_train, feature_test, target_train, target_test = self.ts_train_test_split()
    
       coefs = pd.DataFrame(self.model.coef_, feature_train.columns)
       coefs.columns = ["coef"]
       coefs["abs"] = coefs.coef.apply(np.abs)
       coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
    
       plt.figure(figsize=(15, 7))
       coefs.coef.plot(kind='bar')
       plt.grid(True, axis='y')
       plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');
        
