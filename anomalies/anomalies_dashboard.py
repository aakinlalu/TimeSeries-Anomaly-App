rom os.path import join, dirname 
import numpy as np
import pandas as pd 
from scipy.optimize import minimize  

from bokeh.io import output_file, show, curdoc
from bokeh.layouts import column, row, widgetbox, gridplot
from bokeh.models import Tabs, Panel,ColumnDataSource, Div, HoverTool
from bokeh.models.widgets import RadioGroup, Slider, Select
from bokeh.plotting import figure

from anomaly_app.detection_model import AnomalyTimeSeries
from anomaly_app.detection_model import HoltWinterAnomaly
from anomaly_app.detection_model import ts_cv_score
from anomaly_app.connect import connector



#--------Generate data -----------------
#conn = connector('../config/config.ini', 'Redshift_prod')

template = """
            select date_trunc('hour', occurs_at) as Time, 
            count(1) as event_num
             from discovery.fact_events
            where source_system='IB'
           and date(occurs_at) between '2018-12-01' and '2018-12-31'
            group by 1
             order by 1
"""

#------------ETL process-------------------------------------------------
#df =pd.read_sql(template, conn)
df = pd.read_csv('tests/error.csv')
df['time'] =pd.to_datetime(df.time, format="%Y-%m-%d %H:%M:%S")
df =df.set_index('time')


output_file("anomaly.html")

#-----------------------Load the data into the model----------------------------------


window = Slider(start=0, end=24, value=6,title="window", step=6)
scale = Slider(start=0, end=3, value=1.96,title="scale factor", step=0.03)


#-------------------Rolling mean----------------------------
anomaly = AnomalyTimeSeries(window=window.value, scale=scale.value)
model, series, lower_band, upper_band, anomalies = anomaly.rolling_mean_model(df, '../config/config.ini', 'email')



#model, series, lower_band, upper_band, anomalies = anomaly.rolling_mean_model(df, 'email')

#---------------Start dashboarding----------------------------------------


select_widget = Select(options=["Moving Average","Holt Winter", "Linear Regression", "Ridge Cv", "Lasso Cv", "XGBoost"],
                         value="Moving Average", title='Models',width=200)

desc = Div(text=open("html_dir/description.html").read(), width=1200)
#turning = Div(text=open("../html_dir/parameter.html").read(), width=200)
#model_hd = Div(text=open("../html_dir/model_header.html").read(),width=200)


data_point = ColumnDataSource(data = {'timestamp': series.index,
                                      'Actual_Value': series['event_num'],
                                      'Model': model['event_num'],
                                      'Anomaly':anomalies['event_num'],
                                      })

hv = HoverTool(
   tooltips = [
    ("Actual Value","@Actual_Value"),
    ("Model","@Model"),
    ("Timestamp", "@timestamp{%Y-%m-%d %H:%M:%S}"),
    ("Day of week","@timestamp{%A}"),
    ("Anomaly", "@Anomaly")
    ],
    formatters={
        'timestamp': 'datetime'}
)

#------------------First Plot---------------------------------
plot = figure(x_axis_type='datetime', x_axis_label='Timestamp', y_axis_label='Volume', plot_width=1000, plot_height=400, title ="Internet Banking")   #tooltips=tooltips)
plot.add_tools(hv)

plot.line(x='timestamp', y='Actual_Value', source=data_point, legend="Actual Value")

#anomaly
if anomalies['event_num'].sum()>0:
    plot.circle(x='timestamp',y='Anomaly', source=data_point, color='red', size=9, alpha=0.8, legend="Anomalies")
    plot.line(x='timestamp',y='Anomaly', source=data_point, line_color="orange", legend="Anomalies")

plot.line(x='timestamp', y='Model', source=data_point, line_color="green", legend="Rolling mean")

y = np.append(lower_band.event_num, upper_band.event_num[::-1])
x = np.append(lower_band.index, lower_band.index[::-1])

data_patch = ColumnDataSource(data={'x': x,
                                    'y': y,})

plot.patch(x='x', y='y', source=data_patch, color='#7570B3',line_dash=[4,4], line_color='red', fill_alpha=0.2, legend='Upper bond/lower bond')

plot.legend.location = "top_left"
plot.legend.click_policy="hide"


#-------------------------Second Plot--------------------------
plot2 = figure(plot_width=400, plot_height=300, tools="pan,hover")

#--------------------------Third Plot
plot3 = figure(plot_width=800, plot_height=300, tools="pan,hover",toolbar_location="below", title='NABC')



#------------------------------------Update-----------------------------------
def select():
    new_window = window.value
    new_scale = scale.value
    new_anomaly = AnomalyTimeSeries(window=new_window, scale=new_scale)

    return new_anomaly
    

def update():
   
    anomaly_class=select()

    if select_widget.value =="Moving Average":
        new_model, new_series, new_lower_band, new_upper_band, new_anomalies = anomaly_class.rolling_mean_model(df, '../config/config.ini', 'email')

        new_y = np.append(new_lower_band.event_num, new_upper_band.event_num[::-1])
        new_x = np.append(new_lower_band.index, new_lower_band.index[::-1])

        data_point.data ={'timestamp': new_series.index,
                      'Actual_Value': new_series['event_num'],
                      'Model': new_model['event_num'],
                      'Anomaly':new_anomalies['event_num'],
                      }
        data_patch.data ={'x':new_x,
                          'y': new_y,}


#---------------------deploy------------------------------------
controls = [window, scale, select_widget]  
for control in controls:
   control.on_change('value', lambda attr, old, new: update())

slider_layout = widgetbox(*controls, width=200)


upper_part = row(slider_layout,plot)
lower_part = row(plot2, plot3)
layout = gridplot([desc,upper_part, lower_part], ncols=1)



curdoc().add_root(layout)
#show(layout)
