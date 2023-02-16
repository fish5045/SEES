'''
此函式庫包含讀取設定之功能
'''
import numpy as np
import os
import netCDF4 as nc

    

class MetaArray(np.ndarray):
    """Array with metadata."""

    def __new__(cls, array, dtype=None, order=None, stag='0000'):
        obj = np.asarray(array, dtype=dtype, order=order).view(cls)                                 
        obj.stag = stag
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.metadata = getattr(obj, 'metadata', None)

        
    def data_dim_info(self):
        dim = int(self.stag[0])
        if dim == 2:
            return 'Z         R '
        elif dim == 3:
            return 'Z  Theta  R '
        elif dim == 5:
            return 'Z           '
        elif dim == 6:
            return '   Theta  R '
        elif dim == 7:
            return '          R '
        elif dim == 8:
            return '   Theta  R '
        elif dim == 9:
            return 'Z  Theta    '
        
    def data_stag_info(self):
        stagz = int(self.stag[1])
        stagt = int(self.stag[2])
        stagr = int(self.stag[3])
        zstag_info = ['  ','Z ','Z2']
        tstag_info = ['      ','Theta ','Theta2']
        rstag_info = ['  ','R','R2']
        return zstag_info[stagz]+' '+tstag_info[stagt]+' '+rstag_info[stagr]
    '''
    def __repr__(self):
        return f'{self}\ndim = {self.data_dim_info()}\nstag = {self.data_stag_info()}'
    '''



def Write_ncfile(filename,M,wrf_dir,*varlist):
    #print(run_name)
    #print(var)
    
    
    def Select_coord(MetaArray):
        coord = []
        if MetaArray.stag[0] == '5':
            if MetaArray.stag[1] == '0':
                coord.append('height')
            elif MetaArray.stag[1] == '1':
                coord.append('height_stag')
            elif MetaArray.stag[1] == '2':
                coord.append('height_2')
        if MetaArray.stag[0] == '7':
            if MetaArray.stag[3] == '0':
                coord.append('radius')
            elif MetaArray.stag[3] == '1':
                coord.append('radius_stag')
            elif MetaArray.stag[3] == '2':
                coord.append('radius_2')
        if MetaArray.stag[0] == '2':
            if MetaArray.stag[1] == '0':
                coord.append('height')
            elif MetaArray.stag[1] == '1':
                coord.append('height_stag')
            elif MetaArray.stag[1] == '2':
                coord.append('height_2')
            if MetaArray.stag[3] == '0':
                coord.append('radius')
            elif MetaArray.stag[3] == '1':
                coord.append('radius_stag')
            elif MetaArray.stag[3] == '2':
                coord.append('radius_2')
        return coord
    
    def Add_var(var_features):
        var = output.createVariable(var_features['varname'],'f8',Select_coord(var_features['var']))
        var.stag = ''.join(var_features['var'].data_stag_info().split())
        for key in var_features:
            if key != 'var' and key != 'varname':
                exec(f'var.{key} = var_features[key]')
        var[:] = var_features['var'].data
    
    with nc.Dataset(filename,'w',format='NETCDF4') as output:
        output.createDimension('height',M.Nz)
        output.createDimension('height_stag',M.Nz-1)
        output.createDimension('height_2',M.Nz-2)
        output.createDimension('radius',M.Nr)
        output.createDimension('radius_stag',M.Nr-1)
        output.createDimension('radius_2',M.Nr-2)
        Add_var({'var':M.height,'varname':'height','description':'Pressure coordinate','Unit':'hPa'})
        Add_var({'var':M.heights,'varname':'height_stag','description':'Pressure coordinate at stagger grid','Unit':'hPa'})
        Add_var({'var':MetaArray('5200',M.height[1:-1]),'varname':'height_2','description':'Pressure coordinate without bottom and top boundary','Unit':'hPa'})
        Add_var({'var':M.r,'varname':'radius','description':'Radius coordinate','Unit':'m'})
        Add_var({'var':M.rs,'varname':'radius_stag','description':'Radius coordinate at stagger grid','Unit':'m'})
        Add_var({'var':MetaArray('7002',M.r[1:-1]),'varname':'radius_2','description':'Radius coordinate without inner and outer boundary','Unit':'m'})
        for var in varlist:
            Add_var(var)
        output.work_directory = M.work_dir
        output.wrf_output_directory = wrf_dir
        output.diagnosis_time = f'{M.yr:04d}-{M.mo:02d}-{M.day:02d}_{M.hr:02d}:{M.mi:02d}:{M.sc:02d}'
        output.heightstart = M.heightstart  
        output.heightend = M.heightend    
        output.dz = M.dz          
        output.Nz = M.Nz  
        output.dtheta = M.dtheta      
        output.thetastart = M.thetastart 
        output.thetaend = M.thetaend    
        output.Nt = M.Nt           
        output.rmax = M.rmax         
        output.dr = M.dr          
        output.Nr = M.Nr
        output.smooth = M.smooth                   #九點平滑的次數
        output.stablize_AC = M.stablize_AC                 #是否強制慣性穩定度、靜力穩定度不穩定(<0)時設為中性(0)以增加數值穩定
        output.stablize_BB = M.stablize_BB                 #是否強制慣性穩定度、靜力穩定度不穩定(<0)時設為中性(0)以增加數值穩定
        output.run_name = M.subfix
        
        if M.exp_B == 1:
            output.experiment_type = 'B'
            output.B_factor         = M.B_factor
            output.B_avg            = M.B_avg
        
        if M.exp_hdy == 1:
            output.experiment_type = 'Q'
            output.hdy_factor       = M.hdy_factor
            output.hdy_cut_idx      = M.hdy_cut_idx
            output.hdy_fac_in       = M.hdy_fac_in
            output.hdy_fac_out      = M.hdy_fac_out
            output.hdy_n            = M.hdy_n
            output.hdy_d            = M.hdy_d

        if M.exp_A == 1:
            output.experiment_type = 'A'
            output.A_factor         = M.A_factor
        
        if M.exp_C == 1:
            output.experiment_type = 'C'
            output.C_factor         = M.C_factor
            output.C_cut_idx        = M.C_cut_idx
            output.C_fac_in         = M.C_fac_in
            output.C_fac_out        = M.C_fac_out
            output.C_n              = M.C_n
            output.C_d              = M.C_d




