#This is the settings for SEES
import json
import numpy as np

d = dict()
d['wrf'] = dict()
d['exp'] = dict()
d['model'] = dict()
d['common'] = dict()
d['main'] = dict()
dwrf = d['wrf']
dexp = d['exp']
dmodel = d['model']
dcommon = d['common']
dmain = d['main']

#以'.'區別整數或浮點數
#整數者，=0 為關閉， =1 為開啟

# 模式總設定
dcommon['work_dir_prefix']         = f'LEKIMA_'                    #wrfdatas的儲存路徑，資料夾需位於主程式之下
dcommon['output_dir_prefix']         = f'LEKIMA_'                    #wrfdatas的儲存路徑，資料夾需位於主程式之下
dcommon['cpus']                    = 16     # cpu總數目

#理想化模式網格設定 (圓柱座標)
dcommon['zstart']      = 200              #理想化模式之頂層壓力
dcommon['zend']        = 23000             #理想化模式之底層壓力
dcommon['dz']               = 400              #理想化模式之壓力座標網格間距
dcommon['dtheta']           = 1               #理想化模式之切方向網格間距
dcommon['thetastart']       = 0              #理想化模式之切方向起始網格角度
dcommon['thetaend']         = 360              #理想化模式之切方向終止網格角度
dcommon['dr']               = 1000            #理想化模式之徑向網格間距
dmodel['rmax']              = 801000          #理想化模式之外邊界距中心距離

#資料網格設定 (將WRF之直角網格內插至圓柱座標) (WRFDATA)
#垂直網格、切向網格、徑向網格間距將與理想化模式網格一致，可更動資料的最大徑向範圍
#可設定多個資料，最後將依範圍資料慢慢填入模式網格

dwrf['wrf_dir']             = r'M:\Research_data\wrfout\LEKIMA4_WDM6_5min_PBL\d03'                #欲使用的wrf資料的資料夾名稱
dwrf['wrfout_prefix_1']     = 'wrfout'                #欲使用的wrf資料的資料夾名稱
dwrf['wrfout_prefix_2']     = 'additional'                #欲使用的wrf資料的資料夾名稱

dcommon['yr']               = 2019            #wrf資料 年
dcommon['mo']               = 8               #wrf資料 月
dcommon['day']              = 8               #wrf資料 日
dcommon['hr']               = 13             #wrf資料 時
dcommon['mi']               = 0               #wrf資料 分
dcommon['sc']               = 0              #wrf資料 秒
dwrf['dt']                  = 300             #wrf資料的時間間距(影響到溫度趨勢)


#以下需給定各資料的設定
dmain['wrfdata_num']     = 1                #要建立的WRF資料數目
dwrf['wrfdata_name']     = 'dx3000',     #各wrfdata名稱，將使用資料夾分別
dwrf['wrf_domain']       = 3,               #欲使用的wrf資料domain編號
dwrf['wrf_tracking']     = 1,               #該 wrf domain 是否是tracking，用於定義颱風中心
dwrf['dx']               = 3000,            #wrf資料 東西網格間距
dwrf['dy']               = 3000,            #wrf資料 南北網格間距
dwrf['data_rmax']        = 800000,          #要內插的wrf資料 徑方向範圍


#模式網格變更 (用於理想化實驗)
dexp['omega']            = 1.9945          #SOR factor
dexp['equation']         = 2
dexp['boundary']         = 'Zero'
dexp['balance_type']     = 'Holton'
dexp['diff']             = 1               #是否要使用微分乘法規則(應使用才正確，但會導致結果較為不連續)
dexp['smooth']           = 15               #是否要使用9宮格平滑
dexp['stablize_AC']      = 1               #是否強制慣性穩定度、靜力穩定度不穩定(<0)時設為中性(0)以增加數值穩定
dexp['stablize_BB']      = 0               #是否強制慣性穩定度、靜力穩定度不穩定(<0)時設為中性(0)以增加數值穩定
dexp['subfix']           = 'test'         #繪圖及psi結果(/[work_dir]/result/psi_[subfix].pybin)檔名之後綴
dexp['rerun']            = 1               #如果有psi結果，是否強制重新診斷

dexp['exp_B']            = 0               #是否要進行斜壓性實驗
dexp['B_factor']         = 1.              #斜壓性B的強度是wrf資料的幾倍
dexp['B_avg']            = 0               #斜壓性B是否要取250km平均

dexp['exp_hdy']          = 0               #是否要進行非絕熱加熱實驗
dexp['hdy_factor']       = 1.              #非絕熱加熱是wrf資料的幾倍
dexp['hdy_cut_idx']      = 50              #非絕熱加熱實驗，切分線的網格位置(以內 眼牆/內眼牆，以外 主雨帶/外眼牆)
dexp['hdy_fac_in']       = 1.0              #非絕熱加熱實驗，切分線以內的加熱倍率
dexp['hdy_fac_out']      = 0.5             #非絕熱加熱實驗，切分線以外的加熱倍率
dexp['hdy_n']            = 5               #非絕熱加熱實驗，切分線以內、以外n格進行三點平滑
dexp['hdy_d']            = 3               #非絕熱加熱實驗，切分線的平滑次數

dexp['exp_C']            = 0
dexp['C_factor']         = 1.              #非絕熱加熱是wrf資料的幾倍
dexp['C_cut_idx']        = 50              #非絕熱加熱實驗，切分線的網格位置(以內 眼牆/內眼牆，以外 主雨帶/外眼牆)
dexp['C_fac_in']         = 1.0             #非絕熱加熱實驗，切分線以內的加熱倍率
dexp['C_fac_out']        = 1.0             #非絕熱加熱實驗，切分線以外的加熱倍率
dexp['C_n']              = 5               #非絕熱加熱實驗，切分線以內、以外n格進行三點平滑
dexp['C_d']              = 3               #非絕熱加熱實驗，切分線的平滑次數


dexp['exp_A']            = 0
dexp['A_factor']         = 2.0              #A的強度是wrf資料的幾倍



with open('settings.json', 'w') as f:
    f.write(json.dumps(d,indent=4))



