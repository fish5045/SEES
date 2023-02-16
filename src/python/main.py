'''
The Sawyer-Eliassen Equation Solver (SEES)
Verison: 1.0
Author: Shang-En Li
Date: 2023/2/14
'''

import wrf  # 引入處理wrf模式資料之功能(似NCL wfr_user系列函數)
import numpy as np  # 引入陣列功能
from grids import gridsystem, computation_set, ModelDATA_rsl, WRFDATA, \
    FD2, rmse, Get_wrfout_time, Nine_pts_smooth  # 引入資料前處理函式
from define_CE_structure import divide, read_CE_structure
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import json
import os
os.system('python ../../settings/settings.py')


'''
代辦事項
將繪圖弄成另外一個檔案，視為後處理
output nc檔案 記載版本
main code弄成function
PEP-8
'''


def rect_select(M, rmin=np.nan, rmax=np.nan, zmin=np.nan, zmax=np.nan):
    if np.isnan(rmin):
        rmin = M.r[0]
    if np.isnan(rmax):
        rmax = M.r[-1]
    if np.isnan(zmin):
        zmin = M.z[0]
    if np.isnan(zmax):
        zmax = M.z[-1]

    z_select = np.logical_and(M.z >= zmin, M.z <= zmax).reshape(-1, 1)
    r_select = np.logical_and(M.r >= rmin, M.r <= rmax).reshape(1, -1)

    return np.logical_and(z_select, r_select)


def subploting(ax, var, varname, levels, rrange=300000, contour_var1=None, contour_var2=None, contour_levels=None):
    r_select = np.where(M.r <= rrange, True, False)
    if type(contour_var1) != type(None):
        ax.contour(M.r[r_select]/1000, M.z/1000, contour_var1[:,
                   r_select], levels=contour_levels, colors='k')
    if type(contour_var2) != type(None):
        ax.contour(M.r[r_select]/1000, M.z/1000, contour_var2[:,
                   r_select], levels=contour_levels, colors='g')
    c = ax.contourf(M.r[r_select]/1000, M.z/1000, var[:, r_select],
                    levels=levels, cmap='bwr', norm=PiecewiseNorm(levels), extend='both')
    fig.colorbar(c, ax=ax)
    ax.set_xlabel('Radius (km)', fontsize=12)
    ax.set_ylabel('Height (km)', fontsize=12)
    ax.set_title(varname, fontsize=14)


class PiecewiseNorm(Normalize):
    def __init__(self, levels, clip=False):
        # the input levels
        self._levels = np.sort(levels)
        # corresponding normalized values between 0 and 1
        self._normed = np.linspace(0, 1, len(levels))
        Normalize.__init__(self, None, None, clip)

    def __call__(self, value, clip=None):
        # linearly interpolate to get the normalized value
        return np.ma.masked_array(np.interp(value, self._levels, self._normed))


plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)


if __name__ == '__main__':
    os.makedirs('../../output', exist_ok=True)
    os.makedirs('../../data', exist_ok=True)

    # 讀取設定檔 setting.josn (包含資料網格、模式網格設定、共用設定)
    with open('../../settings/settings.json', 'r') as f:
        settings = json.load(f)
        dcommon = settings['common']
        dexp = settings['exp']
        dwrf = settings['wrf']

    wrf.omp_set_num_threads(dcommon['cpus'])

    # 準備模式網格物件

    class ModelDATA(gridsystem, computation_set[dexp['equation']-1], ModelDATA_rsl):
        pass

    # 主程式取得時間
    yr, mo, day, hr, mi, sc = dcommon['yr'], dcommon['mo'], dcommon['day'], dcommon['hr'], dcommon['mi'], dcommon['sc']
    for i in range(1):
        offset = i*settings['wrf']['dt']
        curr_time, currSec = Get_wrfout_time(yr, mo, day, hr, mi, sc, offset)

        W = []  # WRF 資料網格
        wrfdata_num = settings['main']['wrfdata_num']

        for i in range(wrfdata_num):  # 資料網格數
            # 初始化模式網格
            W.append(WRFDATA(settings['common'],
                     settings['wrf'], i, curr_time))
            W[i].RW_DATASET()  # 依設定載入資料

        # 時間平均
        time_range = range(-6, 7, 1)
        len_time_range = len(time_range)

        Wtmp = [[] for i in time_range]
        for idx, t in enumerate(time_range):
            if t == 0:
                continue
            for i in range(wrfdata_num):  # 資料網格數
                curr_time, currSec = Get_wrfout_time(
                    yr, mo, day, hr, mi, sc, offset + 300*t)
                Wtmp[idx].append(WRFDATA(settings['common'],
                                 settings['wrf'], i, curr_time))  # 初始化模式網格
                Wtmp[idx][i].RW_DATASET()  # 依設定載入資料

        for idx, t in enumerate(time_range):
            if t == 0:
                continue
            for i in range(wrfdata_num):
                W[i].uTR += Wtmp[idx][i].uTR
                W[i].vTR += Wtmp[idx][i].vTR
                W[i].fricvTR += Wtmp[idx][i].fricvTR
                W[i].TTR += Wtmp[idx][i].TTR
                W[i].ThTR += Wtmp[idx][i].ThTR
                W[i].pTR += Wtmp[idx][i].pTR
                W[i].hdyTR += Wtmp[idx][i].hdyTR
                W[i].wTR += Wtmp[idx][i].wTR
                W[i].fTR += Wtmp[idx][i].fTR
                W[i].PBLQTR += Wtmp[idx][i].PBLQTR
                W[i].RADQTR += Wtmp[idx][i].RADQTR

        count = len(time_range)
        for i in range(wrfdata_num):
            W[i].uTR /= len_time_range
            W[i].vTR /= len_time_range
            W[i].fricvTR /= len_time_range
            W[i].TTR /= len_time_range
            W[i].ThTR /= len_time_range
            W[i].pTR /= len_time_range
            W[i].hdyTR /= len_time_range
            W[i].wTR /= len_time_range
            W[i].fTR /= len_time_range
            W[i].PBLQTR /= len_time_range
            W[i].RADQTR /= len_time_range

        curr_time, currSec = Get_wrfout_time(yr, mo, day, hr, mi, sc, offset)

        # %%
        # 讀取理想化模式初始場設定
        M = ModelDATA(settings['common'],
                      settings['model'], settings['exp'], curr_time)
        M.wrfdata_merge(W, wrfdata_num)  # 資料網格合併到模式網格

        no_CE_settings = 0
        # 讀取雙眼牆結構劃分設定
        try:
            inner_r, inner_z, moat_r, moat_z, outer_r, outer_z = \
                read_CE_structure(
                    r'D:\Research_data\WRF_data\LEKIMA4_new_WDM6_5min\define_CE_structure30.txt', curr_time)

            CE_structure = np.zeros([M.Nz, M.Nr])
            CE_structure = divide(CE_structure, M.z, M.r, inner_r, inner_z)
            CE_structure = divide(CE_structure, M.z, M.r, moat_r, moat_z)
            CE_structure = divide(CE_structure, M.z, M.r, outer_r, outer_z)
        except:
            no_CE_settings = 1

        ulevels = np.arange(-17., 18., 2.)
        wlevels_u = np.arange(0, 1.01, 0.1)*0.8
        wlevels_d = np.arange(-1, 0.01, 0.1)*0.2
        wlevels = np.append(
            (wlevels_d[1:]+wlevels_d[:-1])/2, (wlevels_u[1:]+wlevels_u[:-1])/2)
        hdylevels = wlevels/200

        # 計算 A B C
        # 設定參數
        Rd = 287.  # J/kg*K
        Cp = 1004.  # J/kg*K
        g = 9.8  # m/s^2
        kappa = Rd/Cp
        subfix = M.subfix
        omega = M.omega

        ### 製作初始場 ###
        # A、dA_dr、Bt、Bv、dBt_dr、dBv_dr、C、dC_dr、dQ_dr為由T、u、v、hdy、計算後之網格位置
        # AA、dAA_dr、BBt、BBv、dBBt_dr、dBBv_dr、CC、dCC_dr、dQQdr為算術平均後至psi網格的位置，AA[0,0]與psi[1,1]為相同位置

        # 第一區 計算A、dA_dr、Bt、Bv、dBt_dr、dBv_dr、C、dC_dr、dQ_dr
        # 靜力穩定度 A 垂直邊界無A 水平交錯 [z-2,Nr-1]、A上下線性外插到邊界點

        zeta = FD2(M.r*M.avg_v, M.dr, axis=1)/M.r

        c = [None]*2
        r_idx = 200
        title_list = ['Latent heating (K/s)', 'Vertical velocity (m/s)']
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))

        # contourf
        #c[0] = ax[0].contourf(M.r[:r_idx]/1000, M.z/1000, ((M.avg_u**2+M.avg_v**2+M.avg_w**2)**0.5)[:,:r_idx], cmap='bwr', levels=np.arange(-1,1.01,0.1)*1e2,extend='both')
        #c[1] = ax[1].contourf(M.r[:r_idx]/1000, M.z/1000, M.avg_w[:,:r_idx], cmap='bwr', levels=wlevels,norm=PiecewiseNorm(wlevels),extend='both')
        c[0] = ax[0].contourf(M.r[:r_idx]/1000, M.z/1000, M.avg_hdy[:, :r_idx],
                              cmap='bwr', levels=hdylevels, norm=PiecewiseNorm(hdylevels), extend='both')
        c[1] = ax[1].contourf(M.r[:r_idx]/1000, M.z/1000, M.avg_w[:, :r_idx],
                              cmap='bwr', levels=wlevels, norm=PiecewiseNorm(wlevels), extend='both')
        #c[0] = ax[0].contourf(M.r[:r_idx]/1000, M.z/1000, zeta[:,:r_idx], cmap='bwr', levels=hdylevels/5,norm=PiecewiseNorm(hdylevels/5),extend='both')
        #c[1] = ax[1].contourf(M.r[:r_idx]/1000, M.z/1000, M.avg_u[:,:r_idx], cmap='bwr', levels=np.arange(-1,1.01,0.1)*5,extend='both')

        # contour
        ax[0].contour(M.r[:r_idx]/1000, M.z/1000, M.avg_w[:, :r_idx],
                      colors='#008888', levels=[-0.06, -0.01, 0.04, 0.12, 0.2], linewidths=0.7)
        ax[0].contour(M.r[:r_idx]/1000, M.z/1000, M.avg_v[:, :r_idx],
                      colors='#555555', levels=np.arange(30, 71, 5), linewidths=1)
        ax[1].contour(M.r[:r_idx]/1000, M.z/1000, M.avg_v[:, :r_idx],
                      colors='#555555', levels=np.arange(0, 71, 5), linewidths=1)
        for i in range(2):
            if no_CE_settings == 0:
                #ax[i].contour(M.r[:r_idx]/1000, M.z/1000, CE_structure[:,:r_idx], colors='k', levels=[0,1,2,3], linewidths=1.5)
                # for k in np.arange(1,6.01,0.5):
                #ax[i].plot(inner_r[:r_idx]/1000*k, inner_z/1000)
                # ax[i].plot(M.r[np.argmax(zeta,axis=1)+20]/1000*k,M.z/1000)
                #ax[i].plot(inner_r[:r_idx]/1000, inner_z/1000)
                ax[i].plot(moat_r[:r_idx]/1000, moat_z /
                           1000, linewidth=2, color='k')
                ax[i].plot(outer_r[:r_idx]/1000, outer_z /
                           1000, linewidth=2, color='k')
                pass

        for i in range(2):
            ax[i].set_ylabel('Height (km)', fontsize=18)
            ax[i].set_title(title_list[i], fontsize=18)
            ax[i].grid()
            ax[i].set_ylim(0, 23)
            ax[i].set_xlim(0, 200)
            fig.colorbar(c[i], ax=ax[i])
        ax[-1].set_xlabel('Radius (km)', fontsize=18)

        plt.suptitle(f'{curr_time}', fontsize=20)
        plt.tight_layout()
        plt.savefig(f'../../output/d04_{curr_time}.png', dpi=100)
        plt.show()

        # 0/0
        # 數值穩定
        # %%
        if True:

            def plt_var(var, M, varname, levels, cmap='bwr', rrange=800000, contour_var=M.avg_w, contour_levels=[-0.01, 0.15]):
                r_select = np.where(M.r <= rrange, True, False)
                fig, ax = plt.subplots(dpi=120)
                c = ax.contourf(M.r[r_select]/1000, M.z/1000, var[:, r_select],
                                cmap='bwr', extend='both', levels=levels, norm=PiecewiseNorm(levels))
                ax.contour(M.r[r_select]/1000, M.z/1000, contour_var[:, r_select],
                           extend='both', levels=contour_levels,
                           colors='#555555', linewidths=0.5)
                fig.colorbar(c, ax=ax)
                ax.set_xlabel('Radius (km)')
                ax.set_ylabel('Height (km)')
                ax.set_title(varname)
                plt.show()

            '''
            ### output_hdy_no_tsmooth_moat包含fila_外眼牆內外界手動
            Qtotal = np.where(CE_structure>=1, M.avg_hdy, 0)
            Ftotal = np.where(CE_structure>=1, M.avg_fricv, 0)
            Qin= np.where(CE_structure==3, M.avg_hdy,0)
            #Qout= np.where(np.logical_or(CE_structure==1,CE_structure==2), M.avg_hdy,0)
            Qout= np.where(CE_structure==1, M.avg_hdy,0)
            Qin = np.where(Qin>0, Qin, 0)
            Qout = np.where(Qout>3e-4, Qout, 0)
            Qmoat = Qtotal - Qin - Qout

            Fin = np.where(Qin != 0, M.avg_fricv,0)
            Fout = np.where(Qout != 0, M.avg_fricv,0)
            moat_mask_wrf = np.where(Qmoat==0,0,1)

            #Fout 要特別設定
            #在w>0.15 m/s的區域中會選不到外眼牆低層
            #設定成外眼牆涵蓋半徑中，從區域最高點做為最高點，底層作為F選取的最低點

            Fout_mask = np.where(Fout!=0,True,False)
            Fin_mask = np.where(Fin!=0,True,False)
            for i in range(M.Nr):
                Fout_mask[:np.argmax(Fout_mask[:,i]),i] = True
                Fin_mask[:np.argmax(Fin_mask[:,i]),i] = True
            Fout = Fout_mask*M.avg_fricv*np.where(CE_structure==1, 1,0)
            Fin = Fin_mask*M.avg_fricv*np.where(CE_structure==3, 1,0)
            Fmoat = Ftotal - Fin - Fout
            '''
            inner_region = np.where(CE_structure == 3, 1, 0)
            outer_region = np.where(np.logical_or(
                CE_structure == 2, CE_structure == 1), 1, 0)
            for k in range(M.Nz):
                idx = np.argmin(inner_region[k])
                inner_region[k, idx:idx+10] = 1
            # output_hdy_no_tsmooth_moat包含fila_外眼牆內外界手動
            Qtotal = np.where(CE_structure >= 1, M.avg_hdy, 0)
            Ftotal = np.where(CE_structure >= 1, M.avg_fricv, 0)
            Qin = np.where(inner_region, M.avg_hdy, 0)
            #Qout= np.where(np.logical_or(CE_structure==1,CE_structure==2), M.avg_hdy,0)
            Qout = np.where(CE_structure == 1, M.avg_hdy, 0)
            Qin = np.where(Qin > 0, Qin, 0)
            Qout = np.where(Qout > 3e-4, Qout, 0)
            Qmoat = Qtotal - Qin - Qout

            Fin = np.where(Qin != 0, M.avg_fricv, 0)
            Fout = np.where(Qout != 0, M.avg_fricv, 0)
            moat_mask_wrf = np.where(Qmoat == 0, 0, 1)

            # Fout 要特別設定
            # 在w>0.15 m/s的區域中會選不到外眼牆低層
            # 設定成外眼牆涵蓋半徑中，從區域最高點做為最高點，底層作為F選取的最低點

            Fout_mask = np.where(Fout != 0, True, False)
            Fin_mask = np.where(Fin != 0, True, False)
            for i in range(M.Nr):
                Fout_mask[:np.argmax(Fout_mask[:, i]), i] = True
                Fin_mask[:np.argmax(Fin_mask[:, i]), i] = True
            Fout = Fout_mask*M.avg_fricv*np.where(CE_structure == 1, 1, 0)
            Fin = Fin_mask*M.avg_fricv*np.where(CE_structure == 3, 1, 0)
            Fmoat = Ftotal - Fin - Fout

            # 平滑處理
            for i in range(M.smooth):
                M.smoothing()
                Qin = Nine_pts_smooth(Qin)
                Qmoat = Nine_pts_smooth(Qmoat)
                Qout = Nine_pts_smooth(Qout)
                Qtotal = Nine_pts_smooth(Qtotal)
                Fin = Nine_pts_smooth(Fin)
                Fmoat = Nine_pts_smooth(Fmoat)
                Fout = Nine_pts_smooth(Fout)
                Ftotal = Nine_pts_smooth(Ftotal)

            #Qmoat = Qtotal - Qin - Qout
            #Qmoat = np.where(rect_select(M,rmin=80000),0,Qmoat)
            '''
            plt_var(M.avg_w,M,'w (m/s)',wlevels,cmap='bwr',rrange=300000,contour_var=M.avg_hdy,contour_levels=wlevels*0.005)
            plt_var(M.avg_hdy,M,'Qall',wlevels*0.005,cmap='bwr',rrange=300000)
            plt_var(Qin,M,'Qin (K/s)',wlevels*0.005,cmap='bwr',rrange=300000)
            plt_var(Qout,M,'Qout (K/s)',wlevels*0.005,cmap='bwr',rrange=300000)
            plt_var(Qmoat,M,'Qmoat (K/s)',wlevels*0.005,cmap='bwr',rrange=300000)
            plt_var(Qtotal,M,'Qtotal (K/s)',wlevels*0.005,cmap='bwr',rrange=300000)
            plt_var(Fin,M,'Fin (m/s2)',wlevels*0.02,cmap='bwr',rrange=300000)
            plt_var(Fout,M,'Fout (m/s2)',wlevels*0.02,cmap='bwr',rrange=300000)
            plt_var(Fmoat,M,'Fmoat (m/s2)',wlevels*0.02,cmap='bwr',rrange=300000)
            plt_var(Ftotal*10,M,'10 x Tangential friction (m/s2)',np.arange(-1,1.01,0.1)*0.001,cmap='bwr',rrange=300000)
            '''

            # 繪製 forcing
            rrange = 200000
            varname = ['Azimuthal Averaged WRF w (m/s)',
                       '200 x Total Q (K/s)',
                       '50 x Total F (m/ss$^2$)',
                       '200 x Inner eyewall Q (K/s)',
                       '200 x Moat Q (K/s)',
                       '200 x Outer eyewall Q (K/s)',
                       '50 x Inner eyewall F (m/s$^2$)',
                       '50 x Moat F (m/ss$^2$)',
                       '50 x Outer eyewall F (m/ss$^2$)', ]

            varlist = [M.avg_w, Qtotal, Ftotal,
                       Qin, Qmoat, Qout, Fin, Fmoat, Fout]
            ffact = [1, 200, 50, 200, 200, 200, 50, 50, 50]

            fig, ax = plt.subplots(3, 3, figsize=(18, 12))
            ax = ax.reshape(-1)
            for i in range(9):
                subploting(ax[i], varlist[i]*ffact[i], varname[i], wlevels,
                           contour_var1=CE_structure, contour_levels=[0, 1], rrange=rrange)
            plt.suptitle(f'{curr_time}  Forcing', fontsize=16)
            plt.subplots_adjust(top=0.9, bottom=0.25, hspace=1, wspace=0.3)
            plt.tight_layout()
            plt.savefig(
                f'../../output/decompose_forcing_{curr_time}.png', dpi=150)
            plt.show()

        # %%

        if True:

            # main core
            ### 計算psi [Nz,Nr] ###
            # 輸出資料，準備給fortran運算

            rerun = M.rerun
            try:
                M.psi = np.fromfile(
                    f'{M.work_dir}/result/psi_{subfix}.pybin').reshape(M.Nz, M.Nr)
            except:
                rerun = 1

            M.add_task(Q=M.avg_hdy, F=M.avg_fricv)  # 完整的
            M.add_task(Q=Qtotal, F=Ftotal*0)  # 只有內外眼牆潛熱加熱
            M.add_task(Q=Qtotal*0, F=Ftotal)  # 只有內外眼牆摩擦
            M.add_task(Q=Qin, F=Ftotal*0)  # 只有內眼牆潛熱加熱
            M.add_task(Q=Qmoat, F=Ftotal*0)  # 只有moat潛熱加熱
            M.add_task(Q=Qout, F=Ftotal*0)  # 只有外眼牆潛熱加熱
            M.add_task(Q=Qtotal*0, F=Fin)  # 只有內眼牆摩擦
            M.add_task(Q=Qtotal*0, F=Fmoat)  # 只有moat摩擦
            M.add_task(Q=Qtotal*0, F=Fout)  # 只有外眼牆摩擦

            M.add_task(Q=M.avg_Th, F=Ftotal*0, task_name='Efficiency')
            M.add_task(Q=M.avg_hdy, F=M.avg_fricv*0)

            if rerun == 1:
                M.run_all()

            # 以下沉作為moat的範圍
            #moat_mask = np.where(M.w[0]<0,True,False)
            #moat_mask = np.where(rect_select(M,rmin=35000,rmax=85000,zmax=15000),moat_mask,0)

            # 以設定檔作為moat的範圍
            moat_mask = np.where(CE_structure == 2, 1, 0)

            M.w = np.stack(M.w, axis=0)
            M.u = np.stack(M.u, axis=0)
            M.set_coord_meta('task', 'task', np.arange(
                M.task_SE), unit='', description='Task index for every experiments')
            M.create_ncfile(f'{M.output_dir}_{curr_time}_{subfix}.nc', currSec)
            M.ncfile_addvar('avg_u', ['vertical', 'radial'], unit='m/s',
                            description='Azi-avg radial wind from WRF output')
            M.ncfile_addvar('avg_v', ['vertical', 'radial'], unit='m/s',
                            description='Azi-avg radial wind from WRF output')
            M.ncfile_addvar('avg_w', ['vertical', 'radial'], unit='m/s',
                            description='Azi-avg vertical velocity from WRF output')
            M.ncfile_addvar('avg_T', ['vertical', 'radial'], unit='m/s',
                            description='Azi-avg vertical velocity from WRF output')
            M.ncfile_addvar('avg_Th', ['vertical', 'radial'], unit='m/s',
                            description='Azi-avg vertical velocity from WRF output')
            M.ncfile_addvar('rho', ['vertical', 'radial'], unit='m/s',
                            description='Azi-avg vertical velocity from WRF output')
            M.ncfile_addvar('u', ['task', 'vertical', 'radial'],
                            unit='m/s', description='Diagnosed radial wind')
            M.ncfile_addvar('w', ['task', 'vertical', 'radial'],
                            unit='m/s', description='Diagnosed vertical velocity')
            M.ncfile_addvar('Q', ['task', 'vertical', 'radial'],
                            unit='K/s', description='Latent heating forcing')
            M.ncfile_addvar('F', ['task', 'vertical', 'radial'],
                            unit='m/s2', description='Momentum forcing')
            M.ncfile_addvar('effi_B', ['vertical', 'radial'], unit='K',
                            description='Baroclinicity forcing of efficiency (i.e. potential tenperature)')
            M.ncfile_addvar('effi_H', ['vertical', 'radial'],
                            unit='K/s', description='DEF of latent heating')
            M.ncfile_addvar(
                'effi_M', ['vertical', 'radial'], unit='m/s2', description='DEF of momentum')
            M.ncfile_addvar('effi_conv_H', ['vertical', 'radial'], unit='K/s',
                            description='Latent heating content in the integration of converting term')
            M.ncfile_addvar('effi_conv_M', ['vertical', 'radial'], unit='m/s2',
                            description='Momentum content in the integration of converting term')
            M.ncfile_addvar('chi', ['vertical', 'radial'], unit='K/s',
                            description='Solved variable of S-E equation')
            M.ncfile_addvar('psi', ['task', 'vertical', 'radial'],
                            unit='m/s2', description='Solved variable of DEF theory')

            # %%

            # 計算質量通量
            r2d = M.r.reshape(1, -1)
            z2d = M.z.reshape(-1, 1)
            rho2d = M.rho_ref.reshape(-1, 1)
            mass_flux = np.zeros([9])

            w_mask = np.where(M.w[1] < 0, 1, 0)
            w_mask_wrf = np.where(M.avg_w < 0, 1, 0)
            # 計算下沉質量通量
            for i in range(9):
                mass_flux[i] = np.sum(
                    r2d*rho2d*M.w[i]*M.dr*M.dz*moat_mask*w_mask)*2*np.pi
            mass_flux_ratio = mass_flux/mass_flux[1]

            wrf_mass_flux_ratio0 = mass_flux[0]/np.sum(
                r2d*rho2d*M.avg_w*M.dr*M.dz*moat_mask*w_mask_wrf)
            wrf_mass_flux_ratio1 = mass_flux[1]/np.sum(
                r2d*rho2d*M.avg_w*M.dr*M.dz*moat_mask*w_mask_wrf)

            # 繪製 u
            varname = ['Total Q and F',
                       'Total Q',
                       'Total F',
                       'Inner eyewall Q',
                       'Moat Q',
                       'Outer eyewall Q',
                       'Inner eyewall F',
                       'Moat F',
                       'Outer eyewall F', ]

            ufact = [1, 1, 1, 1, 1, 1, 1, 1, 1]
            wfact = [1, 1, 1, 1, 1, 1, 1, 1, 1]

            fig, ax = plt.subplots(3, 3, figsize=(18, 12))
            ax = ax.reshape(-1)
            for i in range(9):
                subploting(ax[i], M.u[i]*ufact[i],
                           varname[i], ulevels, rrange=rrange)
            plt.suptitle(f'{curr_time}  u (m/s)', fontsize=16)
            plt.subplots_adjust(top=0.9, bottom=0.25, hspace=1, wspace=0.3)
            plt.tight_layout()
            plt.savefig(f'../../output/decompose_u_{curr_time}.png', dpi=150)
            plt.show()

            # 繪製 w
            fig, ax = plt.subplots(3, 3, figsize=(18, 12))
            ax = ax.reshape(-1)
            for i in range(9):
                subploting(ax[i], M.w[i]*wfact[i], varname[i], wlevels, contour_var1=moat_mask,
                           contour_var2=w_mask*moat_mask, contour_levels=[0], rrange=rrange)
                ax[i].text(
                    20, 21, f'mass flux: {mass_flux[i]:.2e} kg*m/s', fontsize=14)
                ax[i].text(
                    20, 19, f'mass flux ratio: {mass_flux_ratio[i]*100:.2f}% ', fontsize=14)
            #ax[0].text(20,17,f'mass flux ratio (WRF): {wrf_mass_flux_ratio0*100:.2f}% ',fontsize=14)
            #ax[1].text(20,17,f'mass flux ratio (WRF): {wrf_mass_flux_ratio1*100:.2f}% ',fontsize=14)
            plt.suptitle(f'{curr_time}  w (m/s)', fontsize=16)
            plt.subplots_adjust(top=0.9, bottom=0.25, hspace=1, wspace=0.3)
            plt.tight_layout()
            plt.savefig(f'../../output/decompose_w_{curr_time}.png', dpi=150)
            plt.show()

            # %%
            # 校驗
            #dchidt = -M.chi**2*M.Q - M.u*FD2(M.chi,M.dr,axis=1) - M.w*FD2(M.chi,M.dz,axis=0)
            #dThdt = -dchidt*M.avg_Th**2*3600

            rmse_w = rmse(M.w[0], M.avg_w)
            rmse_u = rmse(M.u[0], M.avg_u)
            print('rmse w: ', rmse_w)
            print('rmse u: ', rmse_u)

            wlevels_u = np.arange(0, 1.01, 0.1)*1.2
            wlevels_d = np.arange(-1, 0.01, 0.1)*1.2
            wlevels = np.append(
                (wlevels_d[1:]+wlevels_d[:-1])/2, (wlevels_u[1:]+wlevels_u[:-1])/2)

            #tlevels = np.arange(-17.,18.,2.)

            r = M.r/1000
            z = M.z/1000
            r_idx = 200
            fig, ax = plt.subplots(3, 2, figsize=(16, 9))
            ax = ax.reshape(-1)
            c = [None]*9
            c[0] = ax[0].contourf(r[:r_idx], z, M.w[0][:, :r_idx], levels=wlevels,
                                  cmap='bwr', norm=PiecewiseNorm(wlevels), extend='both')
            c[1] = ax[1].contourf(
                r[:r_idx], z, M.u[0][:, :r_idx], levels=ulevels, cmap='bwr', extend='both')
            #c[2] = ax[2].contourf(r[:r_idx],z,dThdt[:,:r_idx],levels=tlevels,cmap='bwr')
            c[2] = ax[2].contourf(r[:r_idx], z, M.avg_w[:, :r_idx], levels=wlevels,
                                  cmap='bwr', norm=PiecewiseNorm(wlevels), extend='both')
            c[3] = ax[3].contourf(
                r[:r_idx], z, M.avg_u[:, :r_idx], levels=ulevels, cmap='bwr', extend='both')
            #c[5] = ax[5].contourf(r[:r_idx],z,M.avg_ThTend[:,:r_idx],levels=tlevels,cmap='bwr')
            c[4] = ax[4].contourf(r[:r_idx], z, M.w[0][:, :r_idx]-M.avg_w[:, :r_idx],
                                  levels=wlevels, cmap='bwr', norm=PiecewiseNorm(wlevels), extend='both')
            c[5] = ax[5].contourf(r[:r_idx], z, M.u[0][:, :r_idx] -
                                  M.avg_u[:, :r_idx], levels=ulevels, cmap='bwr', extend='both')
            #c[8] = ax[8].contourf(r[:r_idx],z,M.avg_ThTend[:,:r_idx]-dThdt[:,:r_idx],levels=tlevels,cmap='bwr')

            for i in range(6):
                fig.colorbar(c[i], ax=ax[i])

            calc_type = ['SE', 'WRF', 'S-W']
            var_names = ['w (m/s)', 'u (m/s)', r'd${\theta}$ (K/h)']
            ax = ax.reshape(3, 2)
            for j in range(3):
                for i in range(2):
                    ax[j, i].set_title(
                        f'{calc_type[j]} {var_names[i]}', fontsize=16)
            for i in range(2):
                ax[2, i].set_xlabel('Radius (km)', fontsize=14)
                ax[i, 0].set_ylabel('Height (km)', fontsize=14)
            plt.suptitle(f'{curr_time}')
            plt.tight_layout()
            plt.savefig(f'../../output/result_{curr_time}.png')
            plt.show()

            # %%  加熱效率

            a = M.effi_conv_H[0]
            b = M.effi_conv_M[0]

            eta_H = []
            eta_M = []
            for rmax in [50000, 100000, 200000]:
                C_H = M.conversion_heating(
                    rmin=1000, rmax=rmax, zmin=200, zmax=15000)
                C_M = M.conversion_momentum(
                    rmin=1000, rmax=rmax, zmin=200, zmax=15000)
                H = M.total_heating(rmin=1000, rmax=rmax, zmin=200, zmax=15000)
                F = M.total_momentum(rmin=1000, rmax=rmax,
                                     zmin=200, zmax=15000)

                eta_H.append(C_H/H)
                eta_M.append(C_M/F)

            r_idx = 200
            fig, ax = plt.subplots(1, 2, figsize=(9, 4))
            ax = ax.reshape(-1)
            c = [None]*2
            c[0] = ax[0].contourf(r[:r_idx], z, M.effi_H[0][:, :r_idx]*100,
                                  levels=np.arange(-30, 30.1, 3), cmap='bwr', extend='both')
            c[1] = ax[1].contourf(r[:r_idx], z, M.effi_M[0][:, :r_idx]*100,
                                  levels=np.arange(-100, 100.1, 10), cmap='bwr', extend='both')
            # ax[1].contour(r[:r_idx],z,M.avg_v[:,:r_idx],levels=[0,10,20,30],)

            for i in range(2):
                fig.colorbar(c[i], ax=ax[i])

            var_names = ['DEF of latent heating (%)', 'DEF of momentum (%)']
            for i in range(2):
                ax[i].set_title(f'{var_names[i]}', fontsize=16)
                ax[i].set_xlabel('Radius (km)', fontsize=14)
            ax[0].set_ylabel('Height (km)', fontsize=14)
            fig.text(0.28, 0.3, '${\eta}_{H50}$ : ' +
                     f'{eta_H[0]*100:.2f}%', fontsize=12)
            fig.text(0.28, 0.25, '${\eta}_{H100}$: ' +
                     f'{eta_H[1]*100:.2f}%', fontsize=12)
            fig.text(0.28, 0.2, '${\eta}_{H200}$: ' +
                     f'{eta_H[2]*100:.2f}%', fontsize=12)
            fig.text(0.738, 0.3, '${\eta}_{M50}$ : ' +
                     f'{eta_M[0]*100:.2f}%', fontsize=12)
            fig.text(0.738, 0.25, '${\eta}_{M100}$: ' +
                     f'{eta_M[1]*100:.2f}%', fontsize=12)
            fig.text(0.738, 0.2, '${\eta}_{M200}$: ' +
                     f'{eta_M[2]*100:.2f}%', fontsize=12)

            plt.suptitle(f'{curr_time}')
            plt.tight_layout()
            plt.savefig(f'../../output/DEF_{curr_time}.png', dpi=120)
            plt.show()

# %%
