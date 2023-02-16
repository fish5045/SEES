import numpy as np
from netCDF4 import Dataset                               #引入讀取nc檔功能
import wrf                                         #引入處理wrf模式資料之功能(似NCL wfr_user系列函數)
import time as tm
import multiprocessing as mp
from time import time
import os

from fastcompute import sawyer_eliassen, interp3d



wrf.omp_set_num_threads(16)

# Parameters
g = 9.8
R = 287.
Cp = 1004.
kappa = R/Cp

def Nine_pts_smooth(var,center_weight=1.0):
    svar = np.zeros([var.shape[0]+2,var.shape[1]+2])
    svar[1:-1,1:-1] = var
    svar[0,1:-1] = var[0,:]
    svar[-1,1:-1] = var[-1,:]
    svar[1:-1,0] = var[:,0]
    svar[1:-1,-1] = var[:,-1]
    out = (svar[:-2,1:-1]+svar[1:-1,:-2]+svar[1:-1,1:-1]*center_weight+svar[1:-1,2:]+svar[2:,1:-1])/(4+center_weight)
    return out


def timer(func):
    def func_wrapper(*args, **kwargs):
        t0 = time()
        result = func(*args, **kwargs)
        t1 = time()
        print(f'{func.__name__} cost time: {(t1-t0):.5f} (s).')
        return result
    return func_wrapper


def psi2uw(psi, dz, dr, rho_ref, r, equation):
    if equation == 1:
        w =  FD2(psi,dr,axis=1)/r/rho_ref.reshape(-1,1)
        u = -FD2(psi,dz,axis=0)/r/rho_ref.reshape(-1,1)
    elif equation == 2:
        w =  FD2(r*psi,dr,axis=1)/r/rho_ref.reshape(-1,1)
        u = -FD2(psi,dz,axis=0)/rho_ref.reshape(-1,1)
    return u, w


def solve_SEeqn(bdy, dz, dr, Nz, Nr, rmax, z, A, B, C, total_RHS, dA_dr, dB_dr, dB_dz, dC_dz):
    def RZVarInterp(var,s_rr,s_zz,Frame=False):
        #def RZVarInterp(var,s_rr,s_zz,Nz,Nr,Frame=False):
        '''
        在RZ平面上內插
        輸入: 要內插的資料                 var            [Nz,R]
             要內插的徑網格點              s_rr           [Z,R]
             要內插的垂直網格點            s_zz      [Z,R]
        輸出: 內插後的結果                 outvar         [Z,R]
        選項: 是否要補齊邊界點(A、B、C、Q)  Frame           False
             資料垂直網格數               Nz       Nz
             資料徑網格數                 Nr              Nr
        '''
        def Frame0array(a):
            if len(a.shape) == 1:
                tmp = np.zeros([a.shape[0]+2])
                tmp[1:-1] = a
                return tmp
            elif len(a.shape) == 2:
                tmp = np.zeros([a.shape[0]+2,a.shape[1]+2])
                tmp[1:-1,1:-1] = a
                return tmp

        rr = np.copy(s_rr)
        zz = np.copy(s_zz)
        rr -= np.ones(rr.shape)*rr[0,0]                       #對齊徑方向網格位置
        zz -= np.ones(zz.shape)*zz[0,0]        #對齊垂直方向網格位置
        if Frame == True:
            outvar = interp(dr,dz,rr,zz,Frame0array(var))
            return outvar[1:-1,1:-1]
        else:
            outvar = interp(dr,dz,rr,zz,var)
            return outvar

    
    grid_list = [Nr]
    grid_last = 0
    for grid in grid_list:
        s_r = np.linspace(1000,rmax,grid)
        s_z = z
        s_dr = s_r[1] - s_r[0]
        s_dz = s_z[1] - s_z[0]
        s_rr, s_zz = np.meshgrid(s_r,s_z)
        if grid == grid_list[0]:
            psi = np.zeros([Nz,grid_list[0]])
            psi[Nz//2,grid_list[0]//2] = 1000
        else:
            psi = RZVarInterp(psi,s_rr,s_zz,Nz=Nz,Nr=grid_last)
        s_A = RZVarInterp(A,s_rr,s_zz,Frame=False)
        s_B = RZVarInterp(B,s_rr,s_zz,Frame=False)
        s_C = RZVarInterp(C,s_rr,s_zz,Frame=False)
        s_dA_dr = RZVarInterp(dA_dr,s_rr,s_zz,Frame=False)
        s_dB_dr = RZVarInterp(dB_dr,s_rr,s_zz,Frame=False)
        s_dB_dz = RZVarInterp(dB_dz,s_rr,s_zz,Frame=False)
        s_dC_dz = RZVarInterp(dC_dz,s_rr,s_zz,Frame=False)
        s_RHS = RZVarInterp(total_RHS,s_rr,s_zz,Frame=False)
        psi = sawyer_eliassen(s_dz, s_dr, 1.942, 1e-0, 70000, bdy, 
                              s_RHS, psi, s_A, s_B, s_C, s_dA_dr, s_dB_dr, s_dB_dz, s_dC_dz)
        grid_last = grid
    return psi



def run(input1):
    #equation == 1 時 rho_ref使用rho_ref
    #equation == 2 時 rho_ref使用rho
    
    task, bdy, dz, dr, Nz, Nr, rmax, r, z, rho_ref, \
            A, B, C, total_RHS, dA_dr, dB_dr, dB_dz, dC_dz, equation = input1
    
    psi = solve_SEeqn(bdy, dz, dr, Nz, Nr, rmax, z,
                      A, B, C, total_RHS, dA_dr, dB_dr, dB_dz, dC_dz)
    if equation == 2:
        psi /= r.reshape(1,-1)
    
    u, w = psi2uw(psi, dz, dr, rho_ref, r, equation)
    return task, u, w, psi




def rmse(predictions, targets):
    return np.sqrt(np.nanmean((predictions - targets) ** 2))


def FD2(var,delta,axis,cyclic=False):
    '''
    以二階中差分法（邊界使用二階偏差分法）計算變數var的微分值。

    Parameters
    ----------
    var : array
        要計算差分的陣列。
    delta : float
        此維度的網格間距。
    axis : int
        要取微分的維度。
    cyclic : bool, optional
        此方向是否是循環而週期性的（如圓柱座標的切方向）
        預設為False。

    Returns
    -------
    output : array (dim same as var)
        二階差分法微分值。
    '''
    
    delta = float(delta)
    output = np.zeros(var.shape)
    
    # 此方向為週期性
    if cyclic == True:
        output = (np.roll(var,-1,axis)-np.roll(var,1,axis))/2/delta

    # 非週期性
    elif cyclic == False:
        if len(var.shape) == 3:
            if axis == 2:
                output[:,:,0] = (-3*var[:,:,0]+4*var[:,:,1]-var[:,:,2])/2/delta
                output[:,:,1:-1] = (var[:,:,2:]-var[:,:,:-2])/2/delta
                output[:,:,2:-2] = -(var[:,:,4:]-8*var[:,:,3:-1]+8*var[:,:,1:-3]-var[:,:,:-4])/12/delta
                output[:,:,-1] = (3*var[:,:,-1]-4*var[:,:,-2]+var[:,:,-3])/2/delta
                output[:,:,2:-2] = (-var[:,:,4:]+8*var[:,:,3:-1]-8*var[:,:,1:-3]+var[:,:,:-4])/12/delta
            elif axis == 1:
                output[:,0,:] = (-3*var[:,0,:]+4*var[:,1,:]-var[:,2,:])/2/delta
                output[:,1:-1,:] = (var[:,2:,:]-var[:,:-2,:])/2/delta
                output[:,2:-2,:] = -(var[:,4:,:]-8*var[:,3:-1,:]+8*var[:,1:-3,:]-var[:,:-4,:])/12/delta
                output[:,-1,:] = (3*var[:,-1,:]-4*var[:,-2,:]+var[:,-3,:])/2/delta
                output[:,2:-2,:] = (-var[:,4:,:]+8*var[:,3:-1,:]-8*var[:,1:-3,:]+var[:,:-4,:])/12/delta
            elif axis == 0:
                output[0,:,:] = (-3*var[0,:,:]+4*var[1,:,:]-var[2,:,:])/2/delta
                output[1:-1,:,:] = (var[2:,:,:]-var[:-2,:,:])/2/delta
                output[2:-2,:,:] = -(var[4:,:,:]-8*var[3:-1,:,:]+8*var[1:-3,:,:]-var[:-4,:,:])/12/delta
                output[-1,:,:] = (3*var[-1,:,:]-4*var[-2,:,:]+var[-3,:,:])/2/delta
                output[2:-2,:,:] = (-var[4:,:,:]+8*var[3:-1,:,:]-8*var[1:-3,:,:]+var[:-4,:,:])/12/delta
        elif len(var.shape) == 2:
            if axis == 1:
                output[:,0] = (-3*var[:,0]+4*var[:,1]-var[:,2])/2/delta
                output[:,1:-1] = (var[:,2:]-var[:,:-2])/2/delta
                output[:,2:-2] = -(var[:,4:]-8*var[:,3:-1]+8*var[:,1:-3]-var[:,:-4])/12/delta
                output[:,-1] = (3*var[:,-1]-4*var[:,-2]+var[:,-3])/2/delta
            elif axis == 0:
                output[0,:] = (-3*var[0,:]+4*var[1,:]-var[2,:])/2/delta
                output[1:-1,:] = (var[2:,:]-var[:-2,:])/2/delta
                output[2:-2,:] = -(var[4:,:]-8*var[3:-1,:]+8*var[1:-3,:]-var[:-4,:])/12/delta
                output[-1,:] = (3*var[-1,:]-4*var[-2,:]+var[-3,:])/2/delta
        elif len(var.shape) == 1:
            if axis == 0:
                output[0] = (-3*var[0]+4*var[1]-var[2])/2/delta
                output[1:-1] = (var[2:]-var[:-2])/2/delta
                output[2:-2] = -(var[4:]-8*var[3:-1]+8*var[1:-3]-var[:-4])/12/delta
                output[-1] = (3*var[-1]-4*var[-2]+var[-3])/2/delta
    return output

def Make_vertical_axis(zstart,zend,dz):
    '''
    建立垂直座標軸(等壓座標, hPa)
    輸入: 起始、終止、間距
    輸出: 非交錯垂直座標      z         [Nz]
         交錯垂直座標        zs        [Nz-1]
    '''
    z = np.arange(zstart,zend+0.01,dz,dtype='float64')
    zs = z[:-1]+dz/2 
    return z, zs
    
def Make_tangential_axis(thetastart,thetaend,dtheta):
    '''
    建立切向座標軸(數學角rad)
    輸入: 起始、終止、間距
    輸出: 非交錯切向座標        theta        [Nt]
         交錯切向座標          thetas       [Nt-1] or [Nt](完整一圈)
    '''
    theta = np.arange(thetastart,thetaend+0.00001,dtheta,dtype='float64')
    thetas = (theta[:-1] + theta[1:])/2
    if thetaend-thetastart == 2*np.pi:
        theta = theta[:-1]
    return theta, thetas
    
    
def Make_radial_axis(rmin,rmax,dr,):
    '''
    建立徑向座標軸(m)
    輸入: 起始、終止、間距
    輸出: 非交錯徑向座標      r         [Nr]
         交錯徑向座標        rs        [Nr-1]
    '''
    r = np.array(np.arange(rmin,rmax+0.0001,dr),dtype='float64') 
    rs = r[:-1] + dr/2
    return r, rs


def Get_wrfout_time(yr,mo,day,hr,mi,sc,shift):
    '''
    將wrf輸出設定檔(data_settings_file)所提供之時間訊息轉為wrfout檔名格式
    '''
    currSec = tm.mktime(tm.struct_time((yr,mo,day,hr,mi,sc,0,0,0)))
    return tm.strftime('%Y-%m-%d_%H_%M_%S',tm.localtime(currSec+shift)), currSec


class DimensionError(Exception):
    '''
    陣列維度錯誤
    '''
    pass


def Array_isthesame(a,b,tor=0):
    '''
    判定兩陣列的所有元素是否完全一樣(維度、大小、數值，不包含nan)
    輸入: 陣列一
         陣列二
    輸出: bool
    選項: 浮點數容許的round off error 範圍 (tor)
    '''
    
    if a.shape != b.shape:
        return False
    if a.dtype.name != b.dtype.name:
        return False
    if tor == 0 or a.dtype.name not in ['int32', 'int64', 'float32','float64']:
        TF = (a == b)
    else:
        TF = (np.abs(a-b)<tor)
    
    #numpy 中即使對應元素皆為nan，仍為False，因此取代為true
    TF = np.where(np.logical_and(np.isnan(a),np.isnan(b)), True, TF)
    if tor != 0 and a.dtype.name not in ['int32', 'int64', 'float32','float64']:
        print('Warning: Because two input arrays are not integer or float, tor is useless. Output True means all of the corresponding elements are the same.')
    if False in TF:
        return False
    else:
        return True

        
def interp(dx,dy,xLoc,yLoc,data):
    '''
    將直角網格座標線性內插至任意位置，來源資料可為三維(或二維)，但目標位置為二維，將內插到該位置的不同高度層。
    輸入: 來源資料的x方向網格大小          dx
         來源資料的x方向網格大小          dy
         要內插的位置x座標 (二維以下)     xLoc      [dim0,dim1]
         要內插的位置y座標 (二維以下)     yLoc      [dim0,dim1]
         來源資料，直角網格              data      [lenz,leny,lenx] 或 [leny,lenx]
    輸出: 內插後資料                    output    [lenz,dim0,dim1] 或 [dim0,dim1]
    '''
    if len(xLoc.shape) != len(yLoc.shape):  # 當目標位置為一維(軸)時，用meshgrid建立位置
        raise DimensionError("xLoc與yLoc維度需相同")
    elif len(xLoc.shape) == 1 and len(yLoc.shape) == 1:
        xLoc, yLoc = np.meshgrid(xLoc,yLoc)
        
    if len(data.shape) == 2:
        size0 = data.shape[0]
        size1 = data.shape[1]
        output = interp3d(dx,dy,xLoc,yLoc,data.reshape(1,size0,size1))
        return np.squeeze(output)
    
    elif len(data.shape) == 3:
        output = interp3d(dx,dy,xLoc,yLoc,data)
        return output
    else:
        raise DimensionError("來源資料(data)維度需為2或3")

    
def Make_cyclinder_coord(centerLocation,data_r,theta):
    '''
    以颱風中心為原點，建立圓柱座標的r theta對應的直角座標位置，水平交錯
    '''
    theta = np.array(theta)
    data_r = np.array(data_r)
    Nt = theta.shape[0]     #判斷給定的座標軸位置，座標軸可為交錯位置，因此Nt、data_Nr可能會少1
    data_Nr = data_r.shape[0]
    #建立圓柱座標系 r為要取樣的位置點
    
    #建立需要取樣的同心圓環狀位置(第1維:切向角度，第二維:中心距離)
    xTR = np.zeros([Nt,data_Nr])               #初始化水平交錯後要取樣的x位置 水平交錯就是 +1 /2 dr
    yTR = np.zeros([Nt,data_Nr])               #初始化水平交錯後要取樣的y位置
    
    ddata_r, ttheta = np.meshgrid(data_r, theta)
    xTR = ddata_r*np.cos(ttheta) + centerLocation[1]
    yTR = ddata_r*np.sin(ttheta) + centerLocation[0]

    return xTR, yTR


class gridsystem:
    '''
    網格系統的基底（陣列與網格資訊），包含metadata與從設定檔讀取metadata
    之後可以再衍生為直角網格、圓柱座標網格、wrf網格，可包含多個變數
    '''
    
    def set_meta(self,settings):
        for settingname in settings.keys():
            if type(settings[settingname]) == str:
                text = f'"{settings[settingname]}"'
                exec(f"self.{settingname} = {text}")
            else:
                exec(f"self.{settingname} = {settings[settingname]}")
        
    
    def meta_convert_to_float(self,*metalist):
        for meta in metalist:
            exec(f'self.{meta} = float(self.{meta})')
    
    
    def meta_convert_to_int(self,*metalist):
        for meta in metalist:
            check_code = f'''
if self.{meta} != int(self.{meta}):
    print('Warning: wrfout setting "{meta}" is not a integer and will ignore the decimal.')
'''
            tmp = exec(check_code)
            exec(f'self.{meta} = int(self.{meta})')
    
    
    def open_ncfile(self,filename):
        self.ncfilename = filename

        with Dataset(filename) as ncfile:
            for key, value in ncfile.dimensions.items():
                self.__readcoord(ncfile,key)
            
            for key, value in ncfile.__dict__.items():
                self.meta_list.append(key)
                try:
                    exec(f'self.{key} = {value}')
                except:
                    exec(f'self.{key} = {"value"}')


    def set_var_meta(self, varname, **metadata):
        if self.var_meta.get(varname) == None:
            self.var_meta[varname] = dict()
        
        for key in metadata:
            exec(f"self.var_meta['{varname}']['{key}'] = metadata['{key}']")

    
    def set_coord_meta(self, name, ncname, coord, **metadata):
        self.coords[name] = coord
        self.coord_ncname[name] = ncname
        self.set_var_meta(name, **metadata)


    def __addvar(self, ncfile, varname, data, coord, dtype, metadata):
        var = ncfile.createVariable(varname, dtype, coord)
        for key in metadata:
            exec(f"var.{key} = metadata['{key}']")
        var[:] = data
    
    def __readcoord(self, ncfile, coordncname):
        try:
            coordname = self.ncname_convert[coordncname]
        except:
            coordname = coordncname
        tmp, metadata = self.__readvar(ncfile, coordncname)
        exec(f'self.{coordname} = tmp')
        self.set_coord_meta(coordname, coordncname, tmp, **metadata)
    
    def __readvar(self, ncfile, varname):
        var = ncfile.variables[varname]
        metadata = var.__dict__
        return np.array(var), metadata


    def ncfile_readvar(self, varname):
        with Dataset(self.ncfilename,) as ncfile:
            tmp, metadata = self.__readvar(ncfile, varname)
            exec(f'self.{varname} = tmp')
            self.var_meta[varname] = metadata


    def ncfile_addvar(self, varname, coord, dtype='f4', **metadata):
        self.set_var_meta(varname, **metadata)
        data = eval(f'self.{varname}')
        
        try:
            with Dataset(self.ncfilename,'a') as output:
                self.__addvar(output, varname, data, coord, dtype, self.var_meta[varname])
        except:
            raise RuntimeError('The nc file has not been set yet. Please run the method "create_ncfile" first.')

    
    def ncfile_addcoord(self, ncfile, coordname, coord, dtype='f4'):
        try:
            ncfile.createDimension(self.coord_ncname[coordname], len(coord))
        except:
            ncfile.createDimension(self.coord_ncname[coordname], 1)

        self.__addvar(ncfile, self.coord_ncname[coordname],
                           coord,
                           [self.coord_ncname[coordname]],'f8',self.var_meta[coordname])


    def create_ncfile(self,filename,time=0):
        self.ncfilename = filename
        self.time = time
        self.Ntime = 1
        if self.time == 0:
            self.set_coord_meta('time', 'time', self.time, unit='', description='No time information')
        else:
            self.set_coord_meta('time', 'time', self.time, unit='s', description='Seconds from 1997/1/1 00:00:00')
        
        with Dataset(self.ncfilename,'w',format='NETCDF4') as output:            
            for settingname in self.meta_list:
                exec(f'output.{settingname} = self.{settingname}')
            
            for coordname, coord in self.coords.items():
                self.ncfile_addcoord(output, coordname, coord) 
                
                
    def Nine_pts_smooth(self,var,center_weight=1.0):
        svar = np.zeros([var.shape[0]+2,var.shape[1]+2])
        svar[1:-1,1:-1] = var
        svar[0,1:-1] = var[0,:]
        svar[-1,1:-1] = var[-1,:]
        svar[1:-1,0] = var[:,0]
        svar[1:-1,-1] = var[:,-1]
        out = (svar[:-2,1:-1]+svar[1:-1,:-2]+svar[1:-1,1:-1]*center_weight+svar[1:-1,2:]+svar[2:,1:-1])/(4+center_weight)
        return out
    
    def smoothing(self):
        self.avg_hdy = self.Nine_pts_smooth(self.avg_hdy)
        self.avg_v = self.Nine_pts_smooth(self.avg_v)
        self.avg_T = self.Nine_pts_smooth(self.avg_T)
        self.avg_Th = self.Nine_pts_smooth(self.avg_Th)
        self.avg_fricv = self.Nine_pts_smooth(self.avg_fricv)
        self.avg_p = self.Nine_pts_smooth(self.avg_p)
        self.avg_w = self.Nine_pts_smooth(self.avg_w)
        self.avg_u = self.Nine_pts_smooth(self.avg_u)
        self.avg_PBLQ = self.Nine_pts_smooth(self.avg_PBLQ)
        self.avg_RADQ = self.Nine_pts_smooth(self.avg_RADQ)



class computation1:
    def calc_comm_vars(self):
        v = self.avg_v
        f = self.avg_f
        r = self.r.reshape(1,-1)
        
        self.avg_rho = self.avg_p/R/self.avg_T
        self.rho_ref = np.nanmean(self.avg_rho,axis=(1))
        rho_ref = self.rho_ref.reshape(-1,1)

        self.chi = 1/self.avg_Th
        self.E = v**2/r + f*v
        #self.E = -FD2(self.avg_p, self.dr, axis=1)/self.rho_ref.reshape(-1,1) + self.avg_fricu
        self.xi = 2*v/r + f
        self.avg_zeta = v/r + FD2(v,self.dr,axis=1)
        self.r_rho_ref = r*rho_ref

    
    def calc_A(self, task=0):
        self.A[task] = -g*FD2(self.chi, self.dz, axis=0)/(self.r_rho_ref)
        self.dA_dr[task] = FD2(self.A[task], self.dr, axis=1)
                                               
        
    def calc_B(self, task=0):
        self.B[task] = -FD2(self.chi*self.E, self.dz, axis=0)/(self.r_rho_ref)
        self.dB_dr[task] = FD2(self.B[task], self.dr, axis=1)
        self.dB_dz[task] = FD2(self.B[task], self.dz, axis=0)
    

    def calc_C(self, task=0):
        f = self.avg_f
        self.C[task] = (self.xi*self.chi*(self.avg_zeta + f) + self.E*FD2(self.chi, self.dr, axis=1))/(self.r_rho_ref)
        self.dC_dz[task] = FD2(self.C[task], self.dz, axis=0)
        
    def set_Q(self, Qinput, task=0):
        self.Q[task] = Qinput
        self.calc_RHS1(task)
        self.calc_RHS2(task)
    
    def set_F(self, Finput, task=0):
        self.F[task] = Finput
        self.calc_RHS3(task)
        
    def calc_nonlinear_Qforcing(self, task=0):
        self.Q[task] = self.avg_hdy + self.avg_PBLQ*0# + self.avg_RADQ# - self.Q1 - self.Q2 - self.Q3

    def calc_nonlinear_Fforcing(self, task=0):
        self.F[task] = self.avg_fricv# - self.F1 - self.F2 - self.F3

    
    def calc_RHS1(self, task=0):
        try:
            self.RHS1[task] = g*FD2(self.chi**2*self.Q[task], self.dr, axis=1,)
        except:
            self.calc_nonlinear_Qforcing(task=0)
            self.RHS1[task] = g*FD2(self.chi**2*self.Q[task], self.dr, axis=1,)

    
    def calc_RHS2(self, task=0):
        try:
            self.RHS2[task] = FD2(self.E*self.chi**2*self.Q[task], self.dz, axis=0,)
        except:
            self.calc_nonlinear_Qforcing(task=0)
            self.RHS2[task] = FD2(self.E*self.chi**2*self.Q[task], self.dz, axis=0,)

        
        
    def calc_RHS3(self, task=0):
        try:
            self.RHS3[task] = -FD2(self.chi*self.xi*self.F[task], self.dz, axis=0,)
        except:
            self.calc_nonlinear_Fforcing(task=0)
            self.RHS3[task] = -FD2(self.chi*self.xi*self.F[task], self.dz, axis=0,)




    def calc_total_RHS(self, task=0, RHS1_fac=1, RHS2_fac=1, RHS3_fac=1):
        self.total_RHS[task] = self.RHS1[task]*RHS1_fac + \
                              self.RHS2[task]*RHS2_fac + \
                              self.RHS3[task]*RHS3_fac

class computation2:
    def calc_comm_vars(self):
        # 定義簡短變數
        v = self.avg_v
        f = self.avg_f
        r = self.r.reshape(1,-1)
        self.r2d = r
        
        #AAM
        self.aam = r*v + f*r**2/2
        
        #熱力 reference state
        self.avg_rho = self.avg_p/R/self.avg_T
        self.rho_ref = np.nanmean(self.avg_rho,axis=1)

        # pseudo height reference state
        self.theta_ref = self.calc_RZaverage(self.avg_Th).reshape(-1,1)
        #self.theta_ref = 536.
        #H = (1 - (g*self.z/Cp/self.theta_ref))**(1/kappa)
        #self.rho = (self.rho_ref*H**(1/kappa - 1)).reshape(-1,1)
        self.rho = self.rho_ref.reshape(-1,1)
    
    def calc_A(self, task=0):
        self.A[task] = g/(self.rho*self.theta_ref*self.r2d) * \
                       FD2(self.avg_Th, self.dz, axis=0)
        self.dA_dr[task] = FD2(self.A[task], self.dr, axis=1)
                                               
        
    def calc_B(self, task=0):
        self.B[task] = -g/(self.rho*self.theta_ref*self.r2d) * \
                       FD2(self.avg_Th, self.dr, axis=1)
        self.dB_dr[task] = FD2(self.B[task], self.dr, axis=1)
        self.dB_dz[task] = FD2(self.B[task], self.dz, axis=0)
    

    def calc_C(self, task=0):
        self.C[task] = 1/(self.rho*self.r2d**4) * \
                       FD2(self.aam**2, self.dr, axis=1)
        self.dC_dz[task] = FD2(self.C[task], self.dz, axis=0)
        
    def set_Q(self, Qinput, task=0):
        self.Q[task] = Qinput
        self.calc_RHS1(task)
        self.calc_RHS2(task)
    
    def set_F(self, Finput, task=0):
        self.F[task] = Finput
        self.calc_RHS3(task)
        
    def calc_nonlinear_Qforcing(self, task=0):
        self.Q[task] = self.avg_hdy + self.avg_PBLQ*0# + self.avg_RADQ# - self.Q1 - self.Q2 - self.Q3

    def calc_nonlinear_Fforcing(self, task=0):
        self.F[task] = self.avg_fricv# - self.F1 - self.F2 - self.F3

    
    def calc_RHS1(self, task=0):
        try:
            self.RHS1[task] = g/self.theta_ref * FD2(self.Q[task], self.dr, axis=1,)
        except:
            self.calc_nonlinear_Qforcing(task=0)
            self.RHS1[task] = g/self.theta_ref * FD2(self.Q[task], self.dr, axis=1,)

    
    def calc_RHS2(self, task=0):
        self.RHS2[task] = np.zeros([self.Nz, self.Nr])

        
        
    def calc_RHS3(self, task=0):
        try:
            self.RHS3[task] = -1/self.r2d**2 * FD2(2*self.aam*self.F[task], self.dz, axis=0,)
        except:
            self.calc_nonlinear_Fforcing(task=0)
            self.RHS3[task] = -1/self.r2d**2 * FD2(2*self.aam*self.F[task], self.dz, axis=0,)




    def calc_total_RHS(self, task=0, RHS1_fac=1, RHS2_fac=1, RHS3_fac=1):
        self.total_RHS[task] = self.RHS1[task]*RHS1_fac + \
                              self.RHS2[task]*RHS2_fac + \
                              self.RHS3[task]*RHS3_fac


computation_set = [computation1, computation2]


class ModelDATA_rsl:
    def select_boundary(self,bdy):
        if bdy == 'Neumann':
            return 'N'
        elif bdy == 'Zero':
            return 'D'

    
    def __init__(self, dcommon, dmodel, dexp, curr_time, open_file=None):
        self.meta_list = []
        self.var_meta = dict()
        self.coords = dict()
        self.coord_ncname = dict()
        self.ncname_convert = dict(vertical='z',tangential='theta',radial='r')
        
        if open_file != None:
            self.open_ncfile(open_file)
        
        else:

            #讀取設定
            self.set_meta(dcommon)
            self.set_meta(dmodel)
            self.set_meta(dexp)
    
            #型態轉換
            self.meta_convert_to_float('zstart','zend','dz',
                                       'dtheta','thetastart','thetaend',
                                       'rmax','dr','omega',
                                       'B_factor',
                                       'hdy_factor','hdy_fac_in','hdy_fac_out',
                                       'C_factor','C_fac_in','C_fac_out',
                                       'A_factor')
            self.meta_convert_to_int('diff','smooth','stablize_AC','stablize_BB','rerun',
                                     'exp_B','B_avg',
                                     'exp_hdy','hdy_cut_idx','hdy_n','hdy_d',
                                     'exp_C','C_cut_idx','C_cut_idx','C_n','C_d',
                                     'exp_A')
    
            self.dtheta = np.deg2rad(self.dtheta)
            self.thetastart = np.deg2rad(self.thetastart)
            self.thetaend = np.deg2rad(self.thetaend)
    
    
            #額外可以由設定計算出來的延伸設定 （如網格數）
            self.Nz = int((self.zend-self.zstart+0.1)/self.dz)+1    #WRF模式內插至等壓座標的垂直層數
            self.Nr = int((self.rmax-1000)/self.dr)+1  
            
            if (self.thetaend-self.thetastart) == 2*np.pi:
                self.Nt = int(np.round((self.thetaend-self.thetastart)/self.dtheta))
                self.cyclic = True
            else:
                self.Nt = int(np.round((self.thetaend-self.thetastart)/self.dtheta))+1
                self.cyclic = False
    
            #建立座標軸
            self.init_coord()
            
            #建立網格
            self.init_grid()   # 此段交由後續不同種網格做不同處理
            
            self.curr_time = curr_time
            self.work_dir = f'../../data/{self.work_dir_prefix}{self.curr_time}'
            self.output_dir = f'../../output/{self.output_dir_prefix}'

    def init_coord(self):
        #設定座標軸
        self.z, _ = Make_vertical_axis(self.zstart,self.zend,self.dz)
        self.theta, _ = Make_tangential_axis(self.thetastart,self.thetaend,self.dtheta)
        self.r, _ = Make_radial_axis(1000,self.rmax,self.dr)
        

        self.set_coord_meta('z', 'vertical', self.z, unit='m', description='Vertical coordinate')
        self.set_coord_meta('r', 'radial', self.r, unit='m', description='Radial coordinate')


    def init_grid(self):
        Nz = self.Nz
        Nt = self.Nt
        Nr = self.Nr
        # read model settings in settings.txt
        self.avg_u = np.zeros([Nz,Nr])
        self.avg_v = np.zeros([Nz,Nr])
        self.avg_fricv = np.zeros([Nz,Nr])
        self.avg_T = np.zeros([Nz,Nr])
        self.avg_Th = np.zeros([Nz,Nr])
        self.avg_p = np.zeros([Nz,Nr])
        self.avg_hdy = np.zeros([Nz,Nr])            
        self.avg_w = np.zeros([Nz,Nr])
        self.avg_f = np.zeros([1])
        self.avg_PBLQ = np.zeros([Nz,Nr])
        self.avg_RADQ = np.zeros([Nz,Nr])
        self.avg_rho = np.zeros([Nz,Nr])
        self.fTR = np.zeros([Nt,Nr])

        
        #S-E 環境場
        self.A = []
        self.B = []
        self.C = []
        self.dA_dr = []
        self.dB_dr = []
        self.dB_dz = []
        self.dC_dz = []
        self.RHS1 = []
        self.RHS2 = []
        self.RHS3 = []
        self.total_RHS = []
        self.BBAC = []
        self.Q = []
        self.F = []
        self.effi_B = []
        self.bdy = []

        
        #模式結果
        self.psi = []
        self.u = []
        self.w = []
        self.chi = []
        self.effi_H = []
        self.effi_M = []
        self.effi_conv_H = []
        self.effi_conv_M = []
        
        self.Ntask = 0
        self.task_type = []
        self.task_vars = [self.A, self.B, self.C, self.Q, self.F, 
                          self.dA_dr, self.dB_dr, self.dB_dz, self.dC_dz,
                          self.RHS1, self.RHS2, self.RHS3, 
                          self.total_RHS, self.BBAC,
                          self.psi, self.u, self.w, self.task_type, self.bdy,
                          self.effi_B, self.chi, self.effi_H, self.effi_M, self.effi_conv_M, self.effi_conv_H]
    
    
    def wrfdata_merge(self,W,wrfdata_num):
        Nz = self.Nz
        Nt = self.Nt
        Nr = self.Nr

        #the grid size of all DATASET are the same
        #5. fill the fill_value to the range between largest to outer boundary
        #6. take tangential avg
        
        
        
        #將資料網格放入模式網格
        for i in range(wrfdata_num):
            data_Nr = W[i].data_Nr
            self.avg_u[:,:data_Nr] = W[i].uTR
            self.avg_v[:,:data_Nr] = W[i].vTR
            self.avg_fricv[:,:data_Nr] = W[i].fricvTR
            self.avg_T[:,:data_Nr] = W[i].TTR
            self.avg_Th[:,:data_Nr] = W[i].ThTR
            self.avg_p[:,:data_Nr] = W[i].pTR
            self.avg_hdy[:,:data_Nr] = W[i].hdyTR
            self.avg_w[:,:data_Nr] = W[i].wTR
            self.fTR[:,:data_Nr] = W[i].fTR
            self.avg_PBLQ[:,:data_Nr] = W[i].PBLQTR
            self.avg_RADQ[:,:data_Nr] = W[i].RADQTR

        #fill values for the range between largest domain to outer boundary
        max_data_Nr = W[0].data_Nr
        self.avg_u[:,max_data_Nr:] = W[0].uTR[:,max_data_Nr-1].reshape(Nz,1)
        self.avg_v[:,max_data_Nr:] = W[0].vTR[:,max_data_Nr-1].reshape(Nz,1)
        self.avg_fricv[:,max_data_Nr:] = W[0].fricvTR[:,max_data_Nr-1].reshape(Nz,1)
        self.avg_T[:,max_data_Nr:] = W[0].TTR[:,max_data_Nr-1].reshape(Nz,1)
        self.avg_Th[:,max_data_Nr:] = W[0].ThTR[:,max_data_Nr-1].reshape(Nz,1)
        self.avg_p[:,max_data_Nr:] = W[0].pTR[:,max_data_Nr-1].reshape(Nz,1)
        self.avg_hdy[:,max_data_Nr:] = W[0].hdyTR[:,max_data_Nr-1].reshape(Nz,1)
        self.avg_w[:,max_data_Nr:] = W[0].wTR[:,max_data_Nr-1].reshape(Nz,1)
        self.fTR[:,max_data_Nr:] = W[0].fTR[:,max_data_Nr-1].reshape(Nt,1)
        self.avg_PBLQ[:,max_data_Nr:] = W[0].PBLQTR[:,max_data_Nr-1].reshape(Nz,1)
        self.avg_RADQ[:,max_data_Nr:] = W[0].RADQTR[:,max_data_Nr-1].reshape(Nz,1)

        #取環形平均
        self.avg_f = np.nanmean(self.fTR[:,:int(np.argwhere(W[0].r>250000)[0])])


    def select_axis(self, axis, axis_min, axis_max, axisname):
        if np.min((axis[1:]-axis[:-1])<=0):
            raise ValueError(f"{axisname} axis must increase strictly.")
        if np.isnan(axis_min):
            axis_min = axis[0]
        if np.isnan(axis_max):
            axis_max = axis[-1]
        
        selected = np.logical_and(axis>=axis_min, axis<=axis_max)
        return selected
    
    
    def calc_RZaverage(self, var, rmin=np.nan, rmax=np.nan, zmin=np.nan, zmax=np.nan):
        z_selected = self.select_axis(self.z, zmin, zmax, 'z')
        r_selected = self.select_axis(self.r, rmin, rmax, 'r')
        selected = z_selected.reshape(-1,1)*r_selected.reshape(1,-1)
        
        return np.nansum(var*selected*self.r)/np.nansum(selected*self.r)
    
    
    def integral_RZ(self, var, rmin=np.nan, rmax=np.nan, zmin=np.nan, zmax=np.nan):
        z_selected = self.select_axis(self.z, zmin, zmax, 'z')
        r_selected = self.select_axis(self.r, rmin, rmax, 'r')
        selected = z_selected.reshape(-1,1)*r_selected.reshape(1,-1)
        
        return np.nansum(var*selected*self.r2d*self.dr*self.dz)


    def conversion_heating(self, task=0, rmin=np.nan, rmax=np.nan, zmin=np.nan, zmax=np.nan):
        return self.integral_RZ(self.effi_conv_H[task], rmin, rmax, zmin, zmax)
    
    
    def conversion_momentum(self, task=0, rmin=np.nan, rmax=np.nan, zmin=np.nan, zmax=np.nan):
        return self.integral_RZ(self.effi_conv_M[task], rmin, rmax, zmin, zmax)
    
    
    def total_heating(self, task=0, rmin=np.nan, rmax=np.nan, zmin=np.nan, zmax=np.nan):
        return self.integral_RZ(self.Q[task]*Cp*self.avg_T*self.rho/self.avg_Th, rmin, rmax, zmin, zmax)
    
    
    def total_momentum(self, task=0, rmin=np.nan, rmax=np.nan, zmin=np.nan, zmax=np.nan):
        return self.integral_RZ(self.F[task]*self.avg_v*self.rho, rmin, rmax, zmin, zmax)
    
    
        
    def add_task(self,A=None, B=None, C=None, Q=None, F=None,
                 dA_dr=None, dC_dz=None, dB_dr=None, dB_dz=None, task_name = 'SE'):
        
        task = self.Ntask
        for var in self.task_vars:
            var.append([])
            
        if self.Ntask == 0:
            self.task_type[self.Ntask] = task_name
            self.calc_comm_vars()
            self.calc_A()
            self.calc_B()
            self.calc_C()
            self.calc_RHS1()
            self.calc_RHS2()
            self.calc_RHS3()
            self.calc_total_RHS()
            if self.stablize_AC == 1:
                self.Stablize_AC()
                
        else:
            self.task_type[task] = task_name
            self.A[task] = A if type(A) != type(None) else self.A[0]
            self.B[task] = B if type(B) != type(None) else self.B[0]
            self.C[task] = C if type(C) != type(None) else self.C[0]
            self.Q[task] = Q if type(Q) != type(None) else self.Q[0]
            self.F[task] = F if type(F) != type(None) else self.F[0]
            self.dA_dr[task] = dA_dr if type(dA_dr) != type(None) else self.dA_dr[0]
            self.dB_dr[task] = dB_dr if type(dB_dr) != type(None) else self.dB_dr[0]
            self.dB_dz[task] = dB_dz if type(dB_dz) != type(None) else self.dB_dz[0]
            self.dC_dz[task] = dC_dz if type(dC_dz) != type(None) else self.dC_dz[0]
            self.Stablize_AC(task)
            self.set_Q(Q,task)
            self.set_F(F,task)
            self.calc_total_RHS(task)
            
        if self.task_type[task] == 'SE':
            self.bdy[task] = self.select_boundary(self.boundary)
        elif self.task_type[task] == 'Efficiency':
            self.bdy[task] = self.select_boundary('Zero')
            if self.boundary != 'Zero':
                print('For the requirement of the DEF theory, all boudary conditions are set to Zero.')
        self.Ntask += 1

    
    
    @timer
    def run_all(self):
        content = []
        for task in range(self.Ntask):
            if self.equation == 1:
                content.append([task, self.bdy[task], self.dz, self.dr, self.Nz, self.Nr, self.rmax, 
                                self.r, self.z, self.rho_ref,
                                self.A[task], self.B[task], self.C[task], self.total_RHS[task], 
                                self.dA_dr[task], self.dB_dr[task], self.dB_dz[task], self.dC_dz[task], self.equation])
            
            elif self.equation == 2:
                content.append([task, self.bdy[task], self.dz, self.dr, self.Nz, self.Nr, self.rmax, 
                                self.r, self.z, self.rho,
                                self.A[task], self.B[task], self.C[task], self.total_RHS[task], 
                                self.dA_dr[task], self.dB_dr[task], self.dB_dz[task], self.dC_dz[task], self.equation])
        
        pool = mp.Pool()  #建立任務池 (pool)，4表示開4個核心平行
        results = pool.imap(run, content)  #這個也可以用，同上一行，執行行為不一樣
        pool.close()   #指派任務結束，子執行緒抓取任務並執行 (1個執行緒(thread)=1個核心)
        pool.join()    #等待所有執行緒工作結束
        
        
        ####### 後處理
        # 處理 u, w 依照任務進行計算
        self.task_SE = 0
        self.task_Efficiency = 0
        for task, u, w, psi in results:
            if self.task_type[task] == 'SE':
                self.psi[task] = psi
                self.u[task] = u
                self.w[task] = w
                self.task_SE += 1
            
            elif self.task_type[task] == 'Efficiency':
                self.chi[task] = psi
                self.effi_M[task] = 2*self.aam*u / (self.avg_v*self.r2d**2)
                self.effi_H[task] = g*self.avg_Th*w / (Cp*self.avg_T*self.theta_ref)
                self.effi_B[task] = self.Q[task]
                self.effi_conv_H[task] = g*self.Q[0]*w*self.rho/self.theta_ref
                self.effi_conv_M[task] = 2*self.aam*self.F[0]*self.rho*u/self.r2d**2
                self.Q[task] = []
                self.F[task] = []
                self.task_Efficiency += 1
        
        
        #將空的list予以刪除整理，並給予各類行任務總數
        for i in range(self.Ntask-self.task_SE):
            self.u.remove([])
            self.w.remove([])
            self.Q.remove([])
            self.F.remove([])
            self.psi.remove([])
        
        for i in range(self.Ntask-self.task_Efficiency):
            self.effi_M.remove([])
            self.effi_H.remove([])
            self.effi_B.remove([])
            self.effi_conv_M.remove([])
            self.effi_conv_H.remove([])
            self.chi.remove([])
            
    
    def Nine_pts_smooth(self,var):
        svar = np.zeros([var.shape[0]+2,var.shape[1]+2])
        svar[1:-1,1:-1] = var
        svar[0,1:-1] = var[0,:]
        svar[-1,1:-1] = var[-1,:]
        svar[1:-1,0] = var[:,0]
        svar[1:-1,-1] = var[:,-1]
        svar[ 0, 0] = var[ 0, 0]
        svar[-1, 0] = var[-1, 0]
        svar[ 0,-1] = var[ 0,-1]
        svar[-1,-1] = var[-1,-1]

        var = (svar[:-2,:-2]+svar[:-2,1:-1]+svar[:-2,2:]+svar[1:-1,:-2]+svar[1:-1,1:-1]+svar[1:-1,2:]+svar[2:,:-2]+svar[2:,1:-1]+svar[2:,2:])/9.
        return var
    
    
    def Three_pts_smooth_H(self,var):
        svar = np.zeros([var.shape[0],var.shape[1]+2])
        svar[:,1:-1] = var
        svar[:,0] = var[:,0]
        svar[:,-1] = var[:,-1]
        var = (svar[:,:-2]+svar[:,1:-1]+svar[:,2:])/3.
        return var

    
    def data_smooth(self):
        self.AA = self.Nine_pts_smooth(self.AA)
        self.BBt = self.Nine_pts_smooth(self.BBt)
        self.BBv = self.Nine_pts_smooth(self.BBv)
        self.CC = self.Nine_pts_smooth(self.CC)
        self.dAA_dr = self.Nine_pts_smooth(self.dAA_dr)
        self.dBBt_dr = self.Nine_pts_smooth(self.dBBt_dr)
        self.dBBv_dz = self.Nine_pts_smooth(self.dBBv_dz)
        self.dCC_dz = self.Nine_pts_smooth(self.dCC_dz)
        self.dQQ_dr = self.Nine_pts_smooth(self.dQQ_dr)

    def Stablize_AC(self,task=0):
        self.dA_dr[task] = np.where(self.A[task]<0,0,self.dA_dr[task])
        self.dC_dz[task] = np.where(self.C[task]<0,0,self.dC_dz[task])
        self.A[task] = np.where(self.A[task]<0,0,self.A[task])
        self.C[task] = np.where(self.C[task]<0,0,self.C[task])
    
    def Stablize_BB(self):
        self.stablized_BBt = np.where(self.BBt>0,True,False)
        self.stablized_BBv = np.where(self.BBv>0,True,False)
        self.dBBt_dr = np.where(self.BBt>0,0,self.dBBt_dr)
        self.dBBv_dz = np.where(self.BBv>0,0,self.dBBv_dz)
        self.BBt = np.where(self.BBt>0,0,self.BBt)
        self.BBv = np.where(self.BBv>0,0,self.BBv)
        
    def Stablize(self):
        self.AA = np.where(self.AA<0,3e-5,self.AA)
        self.CC = np.where(self.CC<0,-0.001*self.CC,self.CC)
        self.dAA_dr = np.where(self.AA<0,0,self.dAA_dr)
        self.dCC_dz = np.where(self.CC<0,0,self.dCC_dz)
        symmetric_instability = np.where(self.AA*self.CC-self.BBt*self.BBv<0,True,False)
        self.BBt = np.where(symmetric_instability==True,0,self.BBt)
        self.BBv = np.where(symmetric_instability==True,0,self.BBv)
        self.dBBt_dr = np.where(symmetric_instability==True,0,self.dBBt_dr)
        self.dBBv_dz = np.where(symmetric_instability==True,0,self.dBBv_dz)
        
    def calc_BBAC(self):
        self.BBAC = self.BB*self.BB-self.AA*self.CC
        self.BBAC = np.where(self.BB_AC<0,0,self.BB_AC)
        
    def calc_symmetric(self):
        Nt = self.Nt
        self.uTR_symm = np.stack([np.nanmean(self.uTR,axis=1)]*Nt,axis=1)
        self.vTR_symm = np.stack([np.nanmean(self.vTR,axis=1)]*Nt,axis=1)
        self.TTR_symm = np.stack([np.nanmean(self.TTR,axis=1)]*Nt,axis=1)
        self.hdyTR_symm = np.stack([np.nanmean(self.hdyTR,axis=1)]*Nt,axis=1)
        self.wTR_symm = np.stack([np.nanmean(self.wTR,axis=1)]*Nt,axis=1)
        self.TempTendTR_symm = np.stack([np.nanmean(self.TempTendTR,axis=1)]*Nt,axis=1)
        self.dBZTR_symm = np.stack([np.nanmean(self.dBZTR,axis=1)]*Nt,axis=1)

    def calc_asymmetric(self):
        print('self.uTR_symm' not in dir(self))
        if 'self.uTR_symm' not in dir(self) or \
           'self.vTR_symm' not in dir(self) or \
           'self.TTR_symm' not in dir(self) or \
           'self.hdyTR_symm' not in dir(self) or \
           'self.wTR_symm' not in dir(self) or \
           'self.TempTendTR_symm' not in dir(self) or \
           'self.dBZTR_symm' not in dir(self) :
            self.calc_symmetric()
        self.uTR_asym = self.uTR - self.uTR_symm
        self.vTR_asym = self.vTR - self.vTR_symm
        self.TTR_asym = self.TTR - self.TTR_symm
        self.hdyTR_asym = self.hdyTR - self.hdyTR_symm
        self.wTR_asym = self.wTR - self.wTR_symm
        self.TempTendTR_asym = self.TempTendTR - self.TempTendTR_symm
        self.dBZTR_asym = self.dBZTR - self.dBZTR_symm
    
    



class WRFDATA(gridsystem):
    def __init__(self,dcommon,dwrf,index,curr_time):
        '''
        讀取網格設定
        '''
        self.index = index
        self.set_meta(dcommon)
        self.set_meta(dwrf)
        self.curr_time = curr_time
        
        #這是第幾個網格 讀取對應的資料網格設定
        self.wrfdata_name = self.wrfdata_name[self.index]
        self.wrf_domain = self.wrf_domain[self.index]
        self.wrf_tracking = bool(self.wrf_tracking[self.index])
        self.dx = self.dx[self.index]
        self.dy = self.dy[self.index]
        self.data_rmax = self.data_rmax[self.index]

        #型態轉換
        self.meta_convert_to_float('zstart','zend','dz',
                                   'dtheta','thetastart','thetaend',
                                   'data_rmax','dr','dx','dy')
        self.meta_convert_to_int('yr','mo','day','hr','mi','sc',
                                 'wrf_domain','wrf_tracking')


        #單位轉換
        self.dtheta = np.deg2rad(self.dtheta)
        self.thetastart = np.deg2rad(self.thetastart)
        self.thetaend = np.deg2rad(self.thetaend)
        self.wrf_domain = f"d{self.wrf_domain:02d}"

        #額外可以由設定計算出來的延伸設定 （如網格數）
        self.Nz = int((self.zend-self.zstart+0.1)/self.dz)+1    #WRF模式內插至等壓座標的垂直層數
        self.data_Nr = int((self.data_rmax-1000)/self.dr)+1                          #圓柱座標徑方向網格數
        
        if (self.thetaend-self.thetastart) == 2*np.pi:
            self.Nt = int(np.round((self.thetaend-self.thetastart)/self.dtheta))
            self.cyclic = True
        else:
            self.Nt = int(np.round((self.thetaend-self.thetastart)/self.dtheta))+1
            self.cyclic = False

        
        self.init_coord()
        self.init_grid()   # 此段交由後續不同種網格做不同處理
        self.work_dir = f'../../data/{self.work_dir_prefix}{self.curr_time}'


    def init_coord(self):

        self.z, _ = Make_vertical_axis(self.zstart,self.zend,self.dz)
        self.theta, _ = Make_tangential_axis(self.thetastart,self.thetaend,self.dtheta)
        self.r, _ = Make_radial_axis(1000,self.data_rmax,self.dr)

    
    def init_grid(self):
        #軸
        self.centerLocation = np.array([np.nan,np.nan])
        

    
    def __repr__(self):
        return f'''working directory: {self.work_dir}
wrf output directory: {self.wrf_dir}
wrf domain: {self.wrf_domain}
wrf time: {self.yr:04d}-{self.mo:02d}-{self.day:02d}_{self.hr:02d}_{self.mi:02d}_{self.sc:02d}
wrf dimension: (leny, lenx) = ({self.lenx},{self.leny})
grid size: (dy,dx) = ({self.dx},{self.dy})
tracking: {self.wrf_tracking}

wrfdata name:   {self.wrfdata_name}
wrfdata number: {self.index}
wrfdata dimension: 
    Vertical:   {self.Nz:3d}  [{self.zstart} ~ {self.zend} ({self.dz}) (hPa)]
    Tangential: {self.Nt:3d}  [{round(np.rad2deg(self.thetastart),1)} ~ {round(np.rad2deg(self.thetaend),1)} ({round(np.rad2deg(self.dtheta),1)}) (deg)]
    Radial:     {self.data_Nr:3d}  [1000 ~ {self.data_rmax} ({self.dr}) (m)]
    Cyclic:     {self.cyclic}
dt: {self.dt} (s)'''
    
    def Show_all_settings(self):
        print(f'''work_dir = {self.work_dir}
wrf_dir = {self.wrf_dir}

zstart = {self.zstart}
zend = {self.zend}
dz = {self.dz}
Nz = {self.Nz}
dtheta = {self.dtheta}
thetastart = {self.thetastart}
thetaend = {self.thetaend}
Nt = {self.Nt}
dr = {self.dr}
data_Nr = {self.data_Nr}


yr = {self.yr}
mo = {self.mo}
day = {self.day}
hr = {self.hr}
mi = {self.mi}
sc = {self.sc}
dt = {self.dt}

wrfdata_name = {self.wrfdata_name}
index = {self.index}
wrf_domain = {self.wrf_domain}
wrf_tracking = {self.wrf_tracking}
rmax = {self.data_rmax}
dx = {self.dx}
dy = {self.dy}

lenx = {self.lenx}
leny = {self.leny}
cyclic = {self.cyclic}''')


    def Find_center(self,filenc,z):
        '''
        以1500m最小風速定義颱風中心位置
        '''
        
        if self.wrf_tracking==True:   # WRF tracking domain
            return np.array([self.dy*self.leny/2, self.dx*self.lenx/2])    #centerLocation
        
        ua = np.array(wrf.getvar(filenc,'ua'))
        va = np.array(wrf.getvar(filenc,'va'))
        WEwind = np.array(wrf.interplevel(ua,z,1500,missing=np.nan))
        NSwind = np.array(wrf.interplevel(va,z,1500,missing=np.nan))
        wspd = np.sqrt(WEwind**2 + NSwind**2)
        
        #找颱風中心的範圍
        start = (280,330)             # 起始網格位置(NS,WE)
        end = (350,500)               # 結束網格位置(NS,WE)
        S = start[0]                        #東南西北邊界
        N = end[0]
        W = start[1]
        E = end[1]
        
        
        #初始化欲尋找最小風之範圍陣列(颱風中心位置)
        min_npwspd = np.zeros([N-S,E-W])
        centerLocationIndex = np.zeros([2],dtype='int')
        centerLocation = np.zeros([2],dtype='float')
        
        #找颱風中心
        min_npwspd = wspd[S:N,W:E]
        minIndex = np.nanargmin(min_npwspd[:,:])               #尋找颱風中心網格位置(找範圍內最小風位置)
        centerLocationIndex[:] = [(minIndex//(E-W))+S,(minIndex%(E-W))+W]
        centerLocation[:] = [centerLocationIndex[0]*self.dy,centerLocationIndex[1]*self.dx]
        return centerLocation

    def Get_wrf_data_cyclinder(self,filenc,varname,pre,interpHeight,xinterpLoc=np.array([np.nan]),yinterpLoc=[np.nan],interp2cylinder=True,wrfvar=[]):
        '''
        讀取WRF資料(並內插至圓柱座標,optional)
        輸入: WRF output nc檔案          filenc           nc_file
             要讀取的WRF變數(文字)        varname          str
             該WRF檔案的壓力座標          pre              wrf_var
             要內插到的壓力層             interpHeight     [Nz]
        輸出: wrf變數                    wrfvar           wrf_var
             讀取的資料                  var...      
        選項: 
             要內插到的圓柱座標x位置       xinterpLoc       [Nt,Nr]    預設 np.array([np.nan])
             要內插到的圓柱座標y位置       yinterpLoc       [Nt,Nr]    預設 np.array([np.nan])
             是否使用內插                interp2cylinder             預設 True
             使用使用給定的wrf變數        wrfvar           wrf_var    預設 []
        注意: 若使用內插，xinterpLo與yinterpLoc需給定，此時回傳圓柱座標資料    [Nz,Nt,Nr]
             不使用內插，回傳等壓直角座標                                   [Nz,leny,lenx]
        '''
        if interp2cylinder==True and (Array_isthesame(xinterpLoc,np.array([np.nan])) or Array_isthesame(xinterpLoc,np.array([np.nan]))):
            raise RuntimeError("Fatal Error: Interpolation Function is on but given not enough locations information (xinterpLoc, yinterpLoc).")
    
        if wrfvar == []:        # 讀取WRF變數
            try:
                wrfvar = np.array(wrf.getvar(filenc,varname))
            except IndexError:
                try:
                    wrfvar = np.array(wrf.getvar(filenc,varname))
                except IndexError:
                    wrfvar = np.array(wrf.getvar(filenc,varname))
                    print("xf, xt, yf, yt are invaild and use the whole grids.")
            except ValueError:
                wrfvar = np.squeeze(np.array(filenc.variables[varname][:]))
                print('Warning: cannot use wrf.getvar, use ncfile.variables directly.')
        
        dim = len(wrfvar.shape) # 依照資料的不同維度，有不同功能
        
        if dim < 2:             # 資料小於2維，直接回傳
            return wrfvar
        
        elif dim == 2:          # 資料為2維 (單層)，可選擇內插到圓柱座標
            if interp2cylinder == False:
                return wrfvar, np.array(wrfvar)
            else:
                return wrfvar, interp(self.dx,self.dy,xinterpLoc,yinterpLoc,np.array(wrfvar))
        
        elif dim == 3:         # 資料為3維，可選擇內插到哪幾層高度，可選擇內插到圓柱座標
            var_pre = wrf.interplevel(wrfvar,pre,interpHeight,missing=np.nan)
            if interp2cylinder == False:
                return wrfvar, var_pre
            else:
                return wrfvar, interp(self.dx,self.dy,xinterpLoc,yinterpLoc,var_pre)

    def Write_data(self,var,varname,data_description_file=False,dim=''):
        var.tofile(self.work_dir+'/'+self.wrfdata_name+'/'+varname+'.pybin')
        if data_description_file is not False:
            data_description_file.write('\n'+(varname+'.pybin').ljust(30)+('dim: '+dim).ljust(20))
            print(f'[INFO] Constructing {varname} finished.')

    def Read_data(self,varname,dim,dtype='float64'):
        tmp = np.fromfile(f'{self.work_dir}/{self.wrfdata_name}/{varname}.pybin')
        
        if dim == 'Z       R':
            var = tmp.reshape([self.Nz,self.data_Nr])
        elif dim == '  Theta R':
            var = tmp.reshape([self.Nt,self.data_Nr])
        return var


    def DTdt2(self,data_description_file):
        '''
        找local溫度趨勢(時間二階中差分法) 水平交錯 垂直交錯
        前時間點:prevtime
        後時間點:nexttime
        兩時間點時間差(秒):dt
        '''
        #取得時間
        prevtime = Get_wrfout_time(self.yr,self.mo,self.day,self.hr,self.mi,self.sc,-self.dt)
        nexttime = Get_wrfout_time(self.yr,self.mo,self.day,self.hr,self.mi,self.sc,self.dt)
        #讀取wrf資料
        filenc_prev = Dataset(self.wrf_dir+'/'+self.wrfout_prefix_1+'_'+self.wrf_domain+'_'+prevtime)    #欲讀取之nc檔
        filenc_next = Dataset(self.wrf_dir+'/'+self.wrfout_prefix_1+'_'+self.wrf_domain+'_'+nexttime)    #欲讀取之nc檔
        z_prev = np.array(wrf.getvar(filenc_prev,'z'))
        z_next = np.array(wrf.getvar(filenc_next,'z'))
        
        #各自時間點找中心
        centerLocation_prev = self.Find_center(filenc_prev,z_prev)
        centerLocation_next = self.Find_center(filenc_next,z_next)
        
        #各自時間點建立坐標系
        #data_r, data_rs = Make_radial_axis(1000,self.data_rmax,self.dr)
        #theta, thetas = Make_tangential_axis(self.thetastart,self.thetaend,self.dtheta)
        
        xTR_prev, yTR_prev = Make_cyclinder_coord(centerLocation_prev,self.r,self.theta)
        xTR_next, yTR_next = Make_cyclinder_coord(centerLocation_next,self.r,self.theta)
        
        tk_prev, ThTR_prev = self.Get_wrf_data_cyclinder(filenc_prev,'theta',z_prev,self.z,xTR_prev,yTR_prev)
        tk_next, ThTR_next = self.Get_wrf_data_cyclinder(filenc_next,'theta',z_next,self.z,xTR_next,yTR_next)

        TempTendTR = (ThTR_next - ThTR_prev)/self.dt/2*3600     #(K/s) -> (K/hr)
        method = '2nd central difference'
        
        return TempTendTR, method

    

    def RW_DATASET(self):
        try:
            #讀取已經前處理好的資料(內插到圓柱座標)，準備製作初始場
            #如果資料不存在(沒有齊)、資料損毀、網格數錯誤，將重新進行前處理
            #還要加上這個WRFDATA的資料夾
            
            self.centerLocation = np.fromfile(self.work_dir+'/'+self.wrfdata_name+'/centerLocation.pybin')
            self.uTR = self.Read_data('uTR','Z       R')
            self.vTR = self.Read_data('vTR','Z       R')
            self.fricvTR = self.Read_data('fricvTR','Z       R')
            self.TTR = self.Read_data('TTR','Z       R')
            self.ThTR = self.Read_data('ThTR','Z       R')
            self.pTR = self.Read_data('pTR','Z       R')
            self.hdyTR = self.Read_data('hdyTR','Z       R')
            self.wTR = self.Read_data('wTR','Z       R')
            self.fTR = self.Read_data('fTR','  Theta R')
            self.PBLQTR = self.Read_data('PBLQTR','Z       R')
            self.RADQTR = self.Read_data('RADQTR','Z       R')
            
        
        except:
            ###當前處理資料無法使用時，重新前處理產生資料
            #讀取要分析的wrf輸出資料、wrf metadata      
            filenc = Dataset(f'{self.wrf_dir}/{self.wrfout_prefix_1}_{self.wrf_domain}_{self.curr_time}')    #欲讀取之nc檔
            try:
                filenc2 = Dataset(f'{self.wrf_dir}/{self.wrfout_prefix_2}_{self.wrf_domain}_{self.curr_time}')    #欲讀取之nc檔
            except:
                filenc2 = 0
                print(f'Warning: wrfout file {self.wrfout_prefix_2}_{self.wrf_domain}_{self.curr_time} is not found, RTHRATEN is set to 0.')
            self.leny = filenc.dimensions['south_north'].size
            self.lenx = filenc.dimensions['west_east'].size
            
            os.makedirs(f'{self.work_dir}/{self.wrfdata_name}',exist_ok=True)
            data_description_file = open(f'{self.work_dir}/{self.wrfdata_name}/data_description.txt','w')
            data_description_file.write(f'Time: {self.curr_time}\nwrf_dir: {self.wrf_dir}\ndomain: {self.wrf_domain[2:]}\n')
            data_description_file.write(f'\nVertical range, interval: {self.zstart} ~ {self.zend}, {self.dz} (hPa)')
            if self.cyclic == True:
                data_description_file.write(f'\nTangential range, interval: {np.rad2deg(self.thetastart)} ~ {np.rad2deg(self.thetaend)}, {np.rad2deg(self.dtheta)} (deg) (cyclic)')
            else:
                data_description_file.write(f'\nTangential range, interval: {np.rad2deg(self.thetastart)} ~ {np.rad2deg(self.thetaend)}, {np.rad2deg(self.dtheta)} (deg)')
            data_description_file.write(f'\nRadial range, interval: 1000 ~ {self.data_rmax}, {self.dr} (m)\n')
            
            #讀取高度，interplevel用 (m)
            z = np.array(wrf.getvar(filenc,'z'))
            
            #定颱風中心
            self.centerLocation = self.Find_center(filenc,z)
            self.centerLocation.tofile(self.work_dir+'/'+self.wrfdata_name+'/centerLocation.pybin')
            print('[INFO] Constructing centerLocation finished.')

    
            #建立需要取樣的同心圓環狀位置(第1維:切向角度theta，第二維:中心距離r)
            #非水平交錯
            self.xTR, self.yTR = Make_cyclinder_coord(self.centerLocation,self.r,self.theta)
            
            ### 讀取變數並內插至圓柱座標 (u v w Tk h_diabatic dTdt f dBz) ###
            ##風速、摩擦力部分
            theta = self.theta.reshape(1,-1,1)
            ua, WEwind = self.Get_wrf_data_cyclinder(filenc,'ua',z,self.z,interp2cylinder=False)
            va, NSwind = self.Get_wrf_data_cyclinder(filenc,'va',z,self.z,interp2cylinder=False)
            
            #徑向風 u 、切向風 v
            WEwindTR = interp(self.dx,self.dy,self.xTR,self.yTR,WEwind)
            NSwindTR = interp(self.dx,self.dy,self.xTR,self.yTR,NSwind)
            self.uTR = NSwindTR*np.sin(theta) + WEwindTR*np.cos(theta)
            self.vTR = NSwindTR*np.cos(theta) - WEwindTR*np.sin(theta)
            
            self.uTR = np.nanmean(self.uTR,axis=1)
            self.vTR = np.nanmean(self.vTR,axis=1)
            self.Write_data(self.uTR,'uTR',data_description_file)
            self.Write_data(self.vTR,'vTR',data_description_file)
            
            
            #切向PBL v 
            fu, WEfric = self.Get_wrf_data_cyclinder(filenc,'RUBLTEN',z,self.z,interp2cylinder=False)
            fv, NSfric = self.Get_wrf_data_cyclinder(filenc,'RVBLTEN',z,self.z,interp2cylinder=False)
            WEfricTR = interp(self.dx,self.dy,self.xTR,self.yTR,WEfric)
            NSfricTR = interp(self.dx,self.dy,self.xTR,self.yTR,NSfric)
            self.fricvTR = NSfricTR*np.cos(theta) - WEfricTR*np.sin(theta)
            self.fricvTR = np.nanmean(self.fricvTR,axis=1)
            self.Write_data(self.fricvTR,'fricvTR',data_description_file, 'Z       R')
            
            #徑向PBL u 
            #self.fricuTR = NSfricTR*np.sin(theta) + WEfricTR*np.cos(theta)
            #self.fricuTR = MetaArray(self.fricuTR, stag='3000')
            #self.Write_data(self.fricuTR,'fricuTR',data_description_file)

    
            #溫度 tk 
            tk, self.TTR = self.Get_wrf_data_cyclinder(filenc,'tk',z,self.z,self.xTR,self.yTR)
            self.TTR = np.nanmean(self.TTR,axis=1)
            self.Write_data(self.TTR,'TTR',data_description_file, 'Z       R')
            
            
            #位溫 th 
            th, self.ThTR = self.Get_wrf_data_cyclinder(filenc,'theta',z,self.z,self.xTR,self.yTR)
            self.ThTR = np.nanmean(self.ThTR,axis=1)
            self.Write_data(self.ThTR,'ThTR',data_description_file, 'Z       R')
            
            
            #氣壓 P
            p, self.pTR = self.Get_wrf_data_cyclinder(filenc,'P_HYD',z,self.z,self.xTR,self.yTR)
            self.pTR = np.nanmean(self.pTR,axis=1)
            self.Write_data(self.pTR,'pTR',data_description_file, 'Z       R')
    
            
            #非絕熱加熱(雲微物理潛熱釋放) hdy 
            hdy, self.hdyTR = self.Get_wrf_data_cyclinder(filenc,'H_DIABATIC',z,self.z,self.xTR,self.yTR)
            self.hdyTR = np.nanmean(self.hdyTR,axis=1)
            self.Write_data(self.hdyTR,'hdyTR',data_description_file, 'Z       R')
            
            #PBL scheme 造成的theta 趨勢
            PBLQ, self.PBLQTR = self.Get_wrf_data_cyclinder(filenc,'RTHBLTEN',z,self.z,self.xTR,self.yTR)
            self.PBLQTR = np.nanmean(self.PBLQTR,axis=1)
            self.Write_data(self.PBLQTR,'PBLQTR',data_description_file, 'Z       R')
            
            #輻射非絕熱加熱
            if filenc2 != 0:
                RADQ, self.RADQTR = self.Get_wrf_data_cyclinder(filenc2,'RTHRATEN',z,self.z,self.xTR,self.yTR)
                self.RADQTR = np.nanmean(self.RADQTR,axis=1)
            else:
                self.RADQTR = self.PBLQTR*0
                self.RADQTR = self.RADQTR
            self.Write_data(self.RADQTR,'RADQTR',data_description_file, 'Z       R')

            
            #垂直運動速度 w
            wa, self.wTR = self.Get_wrf_data_cyclinder(filenc,'wa',z,self.z,self.xTR,self.yTR)
            self.wTR = np.nanmean(self.wTR,axis=1)
            self.Write_data(self.wTR,'wTR',data_description_file, 'Z       R')
    
    
            #科氏參數 f
            fco, self.fTR = self.Get_wrf_data_cyclinder(filenc,'F',z,self.z,self.xTR,self.yTR)
            self.Write_data(self.fTR,'fTR',data_description_file, '  Theta R')
            


