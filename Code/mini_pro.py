import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
plt.style.use('fivethirtyeight')


confirmed_cases=pd.read_csv('time_series_covid19_recovered_global.csv')
death_cases=pd.read_csv('time_series_covid19_deaths_global.csv')
recovered_cases=pd.read_csv('time_series_covid19_recovered_global.csv')
cols=confirmed_cases.keys()
confirmed=confirmed_cases.loc[:,cols[4]:cols[-1]]
deaths=death_cases.loc[:,cols[4]:cols[-1]]
recovered=recovered_cases.loc[:,cols[4]:cols[-1]]

dates=confirmed.keys()
total_cases=[]
total_deaths=[]
morality_rate=[]
recovery_rate=[]
total_recovered=[]
total_active=[]

india_cases=[]
china_cases=[]
us_cases=[]
italy_cases=[]
france_cases=[]

india_deaths=[]
china_deaths=[]
us_deaths=[]
italy_deaths=[]
france_deaths=[]

india_recoveries=[]
china_recoveries=[]
us_recoveries=[]
italy_recoveries=[]
france_recoveries=[]


for i in dates:
    confirmed_total=confirmed[i].sum()
    deaths_total=deaths[i].sum()
    recovered_total=recovered[i].sum()

    total_cases.append(confirmed_total)
    total_deaths.append(deaths_total)
    total_recovered.append(recovered_total)
    total_active.append(confirmed_total-(deaths_total+recovered_total))

    morality_rate=(deaths_total/confirmed_total)
    recovery_rate=(recovered_total/confirmed_total)

    india_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='India'][i].sum())
    china_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='China'][i].sum())
    us_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='US'][i].sum())
    italy_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='Italy'][i].sum())
    france_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='France'][i].sum())

    india_deaths.append(death_cases[death_cases['Country/Region']=='India'][i].sum())
    china_deaths.append(death_cases[death_cases['Country/Region']=='China'][i].sum())
    us_deaths.append(death_cases[death_cases['Country/Region']=='US'][i].sum())
    italy_deaths.append(death_cases[death_cases['Country/Region']=='Italy'][i].sum())
    france_deaths.append(death_cases[death_cases['Country/Region']=='France'][i].sum())

    india_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='India'][i].sum())
    china_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='China'][i].sum())
    us_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='US'][i].sum())
    italy_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='Italy'][i].sum())
    france_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='France'][i].sum())

def daily_increase(data):
    d=[]
    for i in range (len(data)):
        if i == 0:
            d.append(data[0])
        else :
            d.append(data[i]-data[i-1])
    return d

#confirmed_cases_daily_increase
world_daily_cases=daily_increase(total_cases)
india_daily_cases=daily_increase(india_cases)
china_daily_cases=daily_increase(china_cases)
us_daily_cases=daily_increase(us_cases)
italy_daily_cases=daily_increase(italy_cases)
france_daily_cases=daily_increase(france_cases)

#death_cases_daily_increase
world_daily_death=daily_increase(total_deaths)
india_daily_death=daily_increase(india_deaths)
china_daily_death=daily_increase(china_deaths)
us_daily_death=daily_increase(us_deaths)
italy_daily_death=daily_increase(italy_deaths)
france_daily_death=daily_increase(france_deaths)

#confirmed_cases_daily_increase
world_daily_recovery=daily_increase(total_recovered)
india_daily_recovery=daily_increase(india_recoveries)
china_daily_recovery=daily_increase(china_recoveries)
us_daily_recovery=daily_increase(us_recoveries)
italy_daily_recovery=daily_increase(italy_recoveries)
france_daily_recovery=daily_increase(france_recoveries)

days_since_1_22=np.array([i for i in range(len(dates))]).reshape(-1,1)
total_cases=np.array(total_cases).reshape(-1,1)
total_deaths=np.array(total_deaths).reshape(-1,1)
total_recovered=np.array(total_recovered).reshape(-1,1)

days_in_future=20
future_forecast=np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1,1)
adjusted_dates=future_forecast[:-20]

start='1/22/20'
start_date=datetime.datetime.strptime(start, '%m/%d/%y')
future_forecast_dates = []
for i in range(len(future_forecast)):
    future_forecast_dates.append((start_date+datetime.timedelta(days=i)).strftime('%m/%d/%y'))
'''
TOTAL CORONA CASES
'''
x_train_confirmed,x_test_confirmed,y_train_confirmed,y_test_confirmed=train_test_split(days_since_1_22,total_cases,test_size=0.25,shuffle=False)

#transform data for polynomial regression
poly=PolynomialFeatures(degree=3)
poly_x_train_confirmed = poly.fit_transform(x_train_confirmed)
poly_x_test_confirmed = poly.fit_transform(x_test_confirmed)
poly_future_forecast = poly.fit_transform(future_forecast)

#polynomial regression
linear_model = LinearRegression(normalize=True,fit_intercept=False)
linear_model.fit(poly_x_train_confirmed, y_train_confirmed)
test_linear_pred=linear_model.predict(poly_x_test_confirmed)
linear_pred = linear_model.predict(poly_future_forecast)
print('MAE : ', mean_absolute_error(test_linear_pred,y_test_confirmed))
print('MSE : ', mean_squared_error(test_linear_pred,y_test_confirmed))

plt.plot(y_test_confirmed)
plt.plot(test_linear_pred)
plt.title('World total cases')
plt.legend(['Test data','Polynomial Prediction data'])
plt.show()

#future prediction of total cases using polynomial regression
plt.figure(figsize=(10,6))
plt.plot(adjusted_dates,total_cases)
plt.plot(future_forecast,linear_pred ,linestyle='dashed',color='red')
plt.title('Prediction of number of corona virus cases over time',size=15)
plt.xlabel('Days since 1/22/2020',size=15)
plt.ylabel('Number of cases',size=15)
plt.legend(['Confirmed cases','polynomial regression prediction'],prop={'size':10})
plt.xticks(size=10)
plt.yticks(size=10)
plt.show()

linear_pred=linear_pred[-20:].reshape(1,-1)[0]
k=np.array(future_forecast_dates[-20:])
a={'Date': k, 'predicted number of confirmed cases worldwide': np.round(linear_pred)}
poly_df=pd.DataFrame.from_dict(a)
print(poly_df)


'''
TOTAL DEATH CASES
'''
x_train_confirmed,x_test_confirmed,y_train_confirmed,y_test_confirmed=train_test_split(days_since_1_22,total_deaths,test_size=0.25,shuffle=False)

#transform data for polynomial regression
poly=PolynomialFeatures(degree=3)
poly_x_train_confirmed = poly.fit_transform(x_train_confirmed)
poly_x_test_confirmed = poly.fit_transform(x_test_confirmed)
poly_future_forecast = poly.fit_transform(future_forecast)

#polynomial regression
linear_model = LinearRegression(normalize=True,fit_intercept=False)
linear_model.fit(poly_x_train_confirmed, y_train_confirmed)
test_linear_pred=linear_model.predict(poly_x_test_confirmed)
linear_pred = linear_model.predict(poly_future_forecast)
print('MAE : ', mean_absolute_error(test_linear_pred,y_test_confirmed))
print('MSE : ', mean_squared_error(test_linear_pred,y_test_confirmed))

plt.plot(y_test_confirmed)
plt.plot(test_linear_pred)
plt.title('World total death cases')
plt.legend(['Test data','Polynomial Prediction data'])
plt.show()

#future prediction of total cases using polynomial regression
plt.figure(figsize=(10,6))
plt.plot(adjusted_dates,total_deaths)
plt.plot(future_forecast,linear_pred ,linestyle='dashed',color='red')
plt.title('Prediction of number of death cases over time',size=15)
plt.xlabel('Days since 1/22/2020',size=15)
plt.ylabel('Number of death cases',size=15)
plt.legend(['Confirmed death cases','polynomial regression prediction'],prop={'size':10})
plt.xticks(size=10)
plt.yticks(size=10)
plt.show()

linear_pred=linear_pred[-20:].reshape(1,-1)[0]
k=np.array(future_forecast_dates[-20:])
a={'Date': k, 'predicted number of confirmed death cases worldwide': np.round(linear_pred)}
poly_df=pd.DataFrame.from_dict(a)
print(poly_df)

'''
TOTAL RECOVERED CASES
'''
x_train_confirmed,x_test_confirmed,y_train_confirmed,y_test_confirmed=train_test_split(days_since_1_22,total_recovered,test_size=0.25,shuffle=False)

#transform data for polynomial regression
poly=PolynomialFeatures(degree=3)
poly_x_train_confirmed = poly.fit_transform(x_train_confirmed)
poly_x_test_confirmed = poly.fit_transform(x_test_confirmed)
poly_future_forecast = poly.fit_transform(future_forecast)

#polynomial regression
linear_model = LinearRegression(normalize=True,fit_intercept=False)
linear_model.fit(poly_x_train_confirmed, y_train_confirmed)
test_linear_pred=linear_model.predict(poly_x_test_confirmed)
linear_pred = linear_model.predict(poly_future_forecast)
print('MAE : ', mean_absolute_error(test_linear_pred,y_test_confirmed))
print('MSE : ', mean_squared_error(test_linear_pred,y_test_confirmed))

plt.plot(y_test_confirmed)
plt.plot(test_linear_pred)
plt.title('World total recovered cases')
plt.legend(['Test data','Polynomial Prediction data'])
plt.show()

#future prediction of total cases using polynomial regression
plt.figure(figsize=(10,6))
plt.plot(adjusted_dates,total_recovered)
plt.plot(future_forecast,linear_pred ,linestyle='dashed',color='red')
plt.title('Prediction of number recovered cases over time',size=15)
plt.xlabel('Days since 1/22/2020',size=15)
plt.ylabel('Number of recovered cases',size=15)
plt.legend(['Confirmed recovered cases','polynomial regression prediction'],prop={'size':10})
plt.xticks(size=10)
plt.yticks(size=10)
plt.show()

linear_pred=linear_pred[-20:].reshape(1,-1)[0]
k=np.array(future_forecast_dates[-20:])
a={'Date': k, 'predicted number of recovered cases worldwide': np.round(linear_pred)}
poly_df=pd.DataFrame.from_dict(a)
print(poly_df)

plt.figure(figsize=(10, 6))
plt.plot(adjusted_dates, india_cases)
plt.plot(adjusted_dates, china_cases)
plt.plot(adjusted_dates, us_cases)
plt.plot(adjusted_dates, italy_cases)
plt.plot(adjusted_dates, france_cases)
plt.title('Number of confirmed Cases', size=15)
plt.xlabel('Days Since 1/22/2020', size=15)
plt.ylabel('Number of Cases', size=15)
plt.legend(['India', 'China', 'Us', 'Italy', 'France'], prop={'size': 10})
plt.xticks(size=10)
plt.yticks(size=10)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(adjusted_dates, india_deaths)
plt.plot(adjusted_dates, china_deaths)
plt.plot(adjusted_dates, us_deaths)
plt.plot(adjusted_dates, italy_deaths)
plt.plot(adjusted_dates, france_deaths)
plt.title('Number of death Cases', size=15)
plt.xlabel('Days Since 1/22/2020', size=15)
plt.ylabel('Number of Cases', size=15)
plt.legend(['India', 'China', 'Us', 'Italy', 'France'], prop={'size': 10})
plt.xticks(size=10)
plt.yticks(size=10)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(adjusted_dates, india_recoveries)
plt.plot(adjusted_dates, china_recoveries)
plt.plot(adjusted_dates, us_recoveries)
plt.plot(adjusted_dates, italy_recoveries)
plt.plot(adjusted_dates, france_recoveries)
plt.title('Number of recovered Cases', size=15)
plt.xlabel('Days Since 1/22/2020', size=15)
plt.ylabel('Number of Cases', size=15)
plt.legend(['India', 'China', 'Us', 'Italy', 'France'], prop={'size': 10})
plt.xticks(size=10)
plt.yticks(size=10)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(adjusted_dates, total_deaths, color='r')
plt.plot(adjusted_dates, total_recovered, color='green')
plt.legend(['death', 'recoveries'], loc='best', fontsize=18)
plt.title('Number of Coronavirus Cases worldwide', size=15)
plt.xlabel('Days Since 1/22/2020', size=15)
plt.ylabel('Number of Cases', size=15)
plt.xticks(size=10)
plt.yticks(size=10)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(total_recovered, total_deaths)
plt.title('Number of Coronavirus Deaths vs. Number of Coronavirus Recoveries', size=15)
plt.xlabel('Number of Coronavirus Recoveries', size=15)
plt.ylabel('Number of Coronavirus Deaths', size=15)
plt.xticks(size=10)
plt.yticks(size=10)
plt.show()


def country_plot(x, y1, y2, y3, y4, country):
    plt. figure(figsize=(10, 6))
    plt.plot(x, y1)
    plt.title('{} Confirmed Cases'.format(country), size=15)
    plt.xlabel('Days Since 1/22/2020', size=15)
    plt.ylabel('Number of Cases', size=15)
    plt.xticks(size=10)
    plt.yticks(size=10)
    plt.show()

    plt. figure(figsize=(10, 6))
    plt.plot(x, y2)
    plt.title('{} Daily Increases in confirmed Cases'.format(country), size=15)
    plt.xlabel('Days Since 1/22/2020', size=15)
    plt.ylabel('Number of Cases', size=15)
    plt.xticks(size=10)
    plt.yticks(size=10)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x, y3)
    plt.title('{} Daily Increases in death Cases'.format(country), size=15)
    plt.xlabel('Days Since 1/22/2020', size=15)
    plt.ylabel('Number of Cases', size=15)
    plt.xticks(size=10)
    plt.yticks(size=10)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x, y4)
    plt.title('{} Daily Increases in recovery Cases'.format(country), size=15)
    plt.xlabel('Days Since 1/22/2020', size=15)
    plt.ylabel('Number of Cases', size=15)
    plt.xticks(size=10)
    plt.yticks(size=10)
    plt.show()


country_plot(adjusted_dates, india_cases, india_daily_cases, india_daily_death, india_daily_recovery, "India")
country_plot(adjusted_dates, china_cases, china_daily_cases, china_daily_death, china_daily_recovery, "China")
country_plot(adjusted_dates, us_cases, us_daily_cases, us_daily_death, us_daily_recovery, "US")
country_plot(adjusted_dates, italy_cases, italy_daily_cases, italy_daily_death, italy_daily_recovery, "Italy")
country_plot(adjusted_dates, france_cases, france_daily_cases, france_daily_death, france_daily_recovery, "France")


