import psycopg2
import configparser
import sys
import os
import pandas as pd 
import datetime


sql_template = """
               

"""

class Connector:
    def __init__(self, config_path, system_name, template=sql_template):
        '''
    parameters
    ----------
    config_path: str
        configuration file that include database authentication
        [Redshift]
        user=########   Login Userid
        password=###### Redshift password
        host=####### Hostname or USER
        port=#### Port number
        dbname=##### Database name
    dbm: str
       it is database management system name as in configuration file
    filename: files
        The database authentication config file
        
    Methods
    -------
    conn = connector(config_path, dbm)
    Redshift function connects you to Redshift database. First, create cedential file that contains
    hostname, username, password, database name and port. For example config.ini
    >>conn = connector('config.ini')
      Connection is successful
    '''
        self.config_path = config_path
        self.system_name = system_name
        self.template = template 


    def connector(self):
        
        #parser = configparser.ConfigParser()
        try:
            if os.path.exists(self.config_path):
                parser = configparser.ConfigParser()
            else:
                print("Config not found! Exiting!")
                sys.exit(1)
            parser.read(self.config_path)
            self.host = parser.get(self.system_name, 'host')
            self.user =parser.get(self.system_name,'user')
            self.password=parser.get(self.system_name,'password')
            self.dbname=parser.get(self.system_name,'dbname')
            self.port=parser.get(self.system_name,'port')
            #if dbm.startswith('Redshift'):
            conn = psycopg2.connect(dbname=self.dbname, host=self.host, port=int(self.port), user=self.user, password=self.password)
            print(conn)
            return conn
        except OSError as e:
            print('we encountered error {}'.format(e))


    def create_data_ts(self):
        conn = self.connector(self.config_path, self.system_name)
        df = pd.read_sql(self.template, conn)
        col_name = list(df.columns)
        if len(col_name) < 3:
            df[col_name[0]] = pd.to_datetime(df[col_name[0]], format="%Y-%m-%d %H:%M:%S")
            df=df.set_index(col_name[0])
            return df
        else:
            df[col_name[1]] = pd.to_datetime(df[col_name[1]], format="%Y-%m-%d %H:%M:%S")
            df=df.set_index(col_name[1])
            return df
