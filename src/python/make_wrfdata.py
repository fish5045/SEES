import numpy as np                                 #引入陣列功能
import os
import wrf                                         #引入處理wrf模式資料之功能(似NCL wfr_user系列函數)
import time as tm                                  #計算花費時間
import Read_settings as Rs
from Read_settings import Mkdir
from grids import  WRFDATA, FD2, rmse, Get_wrfout_time                                        #引入資料前處理函式
import plt_tool_for_paper as pt                              #引入繪圖的整合式函數
import exp                                         #引入實驗用初始場函數
import matplotlib.pyplot as plt
import json

import modelf as mf
if 'interp' not in dir(mf):                        #至此應該都要讀到fortran函式，否則無法繼續下面的運算
    print('Fatal Error: no fortran subroutines.\nCheck module "testmodulef" or "modelf" contains "interp" and "se_eqn_core" or not.')

'''
代辦事項

'''

t0 = tm.time()
wrf.omp_set_num_threads(16)
os.system('python settings.py')
# 讀取設定檔 setting.josn (包含資料網格、模式網格設定、共用設定)
with open('settings.json','r') as f:
    settings = json.load(f)

W = []   #WRF 資料網格
wrfdata_num = settings['main']['wrfdata_num']
yr = settings['common']['yr']
mo = settings['common']['mo']
day = settings['common']['day']
hr = settings['common']['hr']
mi = settings['common']['mi']
sc = settings['common']['sc']
Ndata = 2000
dt = 300
for t in range(Ndata):
    for i in range(wrfdata_num):  #資料網格數
        curr_time, currSec  = Get_wrfout_time(yr,mo,day,hr,mi,sc,t*dt)
        W.append(WRFDATA(settings['common'], settings['wrf'], i, curr_time))     #初始化模式網格
        W[i].work_dir = f'data/LEKIMA_{curr_time}'
        W[i].RW_DATASET()            #依設定載入資料
    W = []







