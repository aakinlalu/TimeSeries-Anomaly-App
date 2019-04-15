======================
ML framework for Timeseries Anomaly Application
======================

Anomaly_app
|-anomalies/
|                |-anomalydashboard.py                                        # see is believing
|                |-model_training.py                             # Train your learner and save it for deploy
|                |-IBModel.sav                                                     #Models
|                |-NabcModel.sav
|                |-TeradataModel.sav
|                |-ESGModel.sav
|
|-config/
|           |-config.ini          #authentication details for storage version and smtp(email)  server
|
|-docs/
|
|-notebook/
|                  |-anomaly_test.ipynb                                         #testing notebook
|                  |-deploy_template.ipynb                                  #deployment notebook
|       
|-src/anomaly_app/
|                                |- connect.py
|                                |                    |- log 
|                                |                    |-connector                    #Connect to data storage 
|                                |                    |-create_data_ts           #transform data source to time-Series 
|                                |                    |-send_email                 #to send emails
|                                |
|                                |- detection_model.py
|                                |                                    |- mean_absolute_percentage_error
|                                |                                    |-send_email
|                                |                                    |-AnomalyTimeSeries                    #class
|                                |                                    |-TimeSeriesML                               #class
|                                                     
|-test/
|        |- dataset to build your model.
|
|-environment.lock.yaml                                                                                        # Reproducibility
|
|-requirements.txt                                                                                                  # List of packages
|
|-setup.py                                                                                                   # python setup.py develop




Description
===========

A longer description of your project goes here...


Note
====
