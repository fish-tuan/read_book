#coding:utf-8
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from dateutil.parser import parse


#runfile('C:/Users/tuan/Documents/code/book/python_for_data_analysis/my_code/10_time_series.py', wdir='C:/Users/tuan/Documents/code/book/python_for_data_analysis/my_code')
if __name__ =='__main__':

    #1 普通的时间函数
    now = datetime.now()
    # print(now)
    delta = datetime(2011,1,7)-datetime(2011,12,7)
    add_time = datetime(2019,3,4)+timedelta(12)
    # print('add_time',add_time)
    # print(delta)

    #2. 时间的转换
    stamp = datetime(2011,1,3)
    print(stamp.strftime('%Y-%m-%d'))#时间转字符串

    value = '2011-3-4'
    value_time = datetime.strptime(value,'%Y-%m-%d')

    #使用包进行更加快
    a = parse('2011-3-4')
    b = pd.to_datetime(value)

    #3.=================================时间序列基础======================================================
    dates = [datetime(2011, 1, 2), datetime(2011, 1, 5),
             datetime(2011, 1, 7), datetime(2011, 1, 8),
             datetime(2011, 1, 10), datetime(2011, 1, 12)]

    ts = pd.Series(np.random.randn(6),index=dates)

    stamp = ts.index[2]
    print(ts[stamp])

    longer_ts = pd.Series(np.random.randn(1000),
                          index=pd.date_range('1/1/2000', periods=1000))

    ts.truncate(after='1/9/2011')

    dates = pd.date_range('1/1/2000', periods=100, freq='W-WED')
    long_df = pd.DataFrame(np.random.randn(100,4),
                           index=dates,
                           columns=['Colorado', 'Texas',
                                'New York', 'Ohio'])
    long_df.loc['5-2001']

    ### Time Series with Duplicate Indices
    dates = pd.DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000',
                              '1/2/2000', '1/3/2000'])
    dup_ts = pd.Series(np.arange(5), index=dates)

    dup_ts.is_unique

    grouped = dup_ts.groupby(level=0)



    ## Date Ranges, Frequencies, and Shifting
    resampler = ts.resample('D')

    a = pd.date_range(start='4/1/2012',periods=20)

    a = pd.date_range('2000-01-01', '2000-12-01', freq='BM')#'BM'代表月

    from pandas.tseries.offsets import Hour, Minute

    hour = Hour()

    a =pd.date_range('2000-01-01', '2000-01-03 23:59', freq='4h')
    a = pd.date_range('2000-01-01', periods=10, freq='1h30min')

    #### Shifting (Leading and Lagging) Data


    #4.========================### Periods and Period Arithmetic时期及其运算============================================
    p = pd.Period(2007, freq='A-DEC')

    rng = pd.period_range('2000-01-01', '2000-06-30', freq='M')
    a  = pd.Series(np.random.randn(6), index=rng)

    values = ['2001Q3', '2002Q2', '2003Q1']
    index = pd.PeriodIndex(values, freq='Q-DEC')

    ### Period Frequency Conversion
    p = pd.Period('2007',freq='A-DEC')
    p.asfreq('M', how='start')
    p.asfreq('M',how='end')

    ### Quarterly Period Frequencies

    p = pd.Period('2012Q4', freq='Q-JAN')

    ### Converting Timestamps to Periods (and Back)
    rng = pd.date_range('2000-01-01', periods=3, freq='M')
    ts = pd.Series(np.random.randn(3), index=rng)

    rng = pd.date_range('1/29/2000', periods=6, freq='D')
    ts2 = pd.Series(np.random.randn(6), index=rng)
    ts2
    # ts2.to_period('M')

   # index = pd.PeriodIndex(year=data.year, quarter=data.quarter, freq='Q-DEC')#合并时间，即把年月合并在一起

   #5.重采样及频率转换
    rng = pd.date_range('2000-01-01', periods=100, freq='D')
    ts = pd.Series(np.random.randn(len(rng)), index=rng)

    ts.resample('M').mean()
    # ts.resample('M', kind='period').mean()

    rng = pd.date_range('2000-01-01', periods=12, freq='T')
    ts = pd.Series(np.arange(12), index=rng)

    ts.resample('5min', closed='right').sum()