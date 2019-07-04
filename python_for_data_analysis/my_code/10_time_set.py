#coding:utf-8
import pandas as pd
from datetime import datetime
from datetime import timedelta
from dateutil.parser import parse


#runfile('C:/Users/tuan/Documents/code/book/python_for_data_analysis/my_code/10_time_set.py', wdir='C:/Users/tuan/Documents/code/book/python_for_data_analysis/my_code')
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