! compile:
! f2py -m SEeqn -c SE_eqn_coreV3.3.f95 --opt=-O3
subroutine Sawyer_Eliassen(Nz, Nr, dz, dr, omega, tol, iter_max, lateral_bdy, &
                         & RHS, psiin, AA, BB, CC, dAA_dr, dBB_dr, dBB_dz, dCC_dz, psiout)
implicit none

! 疊代相關設定
integer :: k,i,s=0,old=0,new=1, bdy     ! iter indexs
integer, intent(in) :: Nz, Nr, iter_max        ! dimensions
character*6, intent(in) :: lateral_bdy            ! boundary
real*8, intent(in) :: dr, dz, omega, tol  !1.942  1d0
real*8, intent(in) :: AA(Nz, Nr), BB(Nz, Nr), CC(Nz, Nr), RHS(Nz, Nr), &
                   & dAA_dr(Nz, Nr), dBB_dr(Nz, Nr), dBB_dz(Nz, Nr), dCC_dz(Nz, Nr)
real*8, intent(in) :: psiin(Nz, Nr)
real*8, intent(out) :: psiout(Nz, Nr)
real*8 :: tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,error
real*8,allocatable :: s1(:,:),s2(:,:),s3(:,:),s4(:,:),s5(:,:),s6(:,:), psi(:,:,:)

allocate(s1(Nz, Nr),s2(Nz, Nr),s3(Nz, Nr),s4(Nz, Nr),&
        &s5(Nz, Nr),s6(Nz, Nr), psi(2, Nz, Nr))

psi= 0
error = 10000
psi(new,:,:) = psiin
if (lateral_bdy == "N") then 
    bdy = 1
elseif (lateral_bdy == "D") then
    bdy = 0
endif


do k = 1,Nz
    do i = 1,Nr
        s1(k,i) =  dAA_dr(k,i)/2/dr + AA(k,i)/dr**2 + dBB_dz(k,i)/2/dr
        s2(k,i) = -dAA_dr(k,i)/2/dr + AA(k,i)/dr**2 - dBB_dz(k,i)/2/dr
        s3(k,i) =  dCC_dz(k,i)/2/dz + CC(k,i)/dz**2 + dBB_dr(k,i)/2/dz
        s4(k,i) = -dCC_dz(k,i)/2/dz + CC(k,i)/dz**2 - dBB_dr(k,i)/2/dz
        s5(k,i) =  BB(k,i)/2/dr/dz
        s6(k,i) =  2*(AA(k,i)/dr**2 + CC(k,i)/dz**2)
    end do
end do

! 計算psi
do s=1,iter_max
        old = mod(old,2)+1
        new = mod(new,2)+1
    
    !網格計算
        do k = 2, Nz-1
            do i = 2, Nr
                if (i == Nr) then
                    psi(new,k,i) = psi(new,k,i-1)*bdy
                    cycle
                end if
                
                tmp1 = s1(k,i)*psi(old,k,i+1)
                tmp2 = s2(k,i)*psi(new,k,i-1)
                tmp3 = s3(k,i)*psi(old,k+1,i)
                tmp4 = s4(k,i)*psi(new,k-1,i)
                tmp5 = s5(k,i)*(psi(old,k+1,i+1) - psi(old,k+1,i-1) - psi(new,k-1,i+1) + psi(new,k-1,i-1))
                tmp6 = RHS(k,i)
                tmp7 = s6(k,i)
                psi(new,k,i) = (1-omega)*psi(old,k,i)+ omega*(tmp1 + tmp2 + tmp3 + tmp4 + tmp5 - tmp6)/tmp7
            end do
        end do    

    ! 計算誤差
    if (mod(s,10) == 0) then
        error = MAXVAL(dabs(psi(1,:,:)-psi(2,:,:)))
        write(*,*) s,error/omega, psi(new,10,10)!, omega, error
    end if  


    if (error/omega < tol) then
        psiout = psi(new,:,:)
        deallocate(s1,s2,s3,s4,s5,s6,psi)
        return 
    end if
    
    
end do

psiout = psi(new,:,:)
deallocate(s1,s2,s3,s4,s5,s6,psi)

end subroutine

subroutine interp3d(lenx,leny,lenz,dx,dy,Nt,Nr,xRThetaLocation,yRThetaLocation,xyData,RThetaData)
implicit none
real*8 :: Z00, Z01, Z10, Z11, tmpZ0, tmpZ1
integer :: x0, y0, x1, y1
integer :: k,j,i
integer,intent(in) :: lenx, leny, lenz, Nt, Nr   ! 分別表示直角座標x、y、z，圓柱座標theta, r
real*8,intent(in) :: dx,dy
real*8,intent(in) :: xRThetaLocation(Nt,Nr), yRThetaLocation(Nt,Nr), xyData(lenz,leny,lenx)
real*8,intent(out) :: RThetaData(lenz,Nt,Nr)

do i = 1, Nr
    do j = 1, Nt
        do k = 1, lenz
            x0 = int(xRThetaLocation(j,i)/dx)+1
            y0 = int(yRThetaLocation(j,i)/dy)+1
            x1 = x0 + 1
            y1 = y0 + 1
            Z00 = xyData(k,y0,x0)
            Z10 = xyData(k,y0,x1)
            Z01 = xyData(k,y1,x0)
            Z11 = xyData(k,y1,x1)
            tmpZ0 = (Z10-Z00)*(xRThetaLocation(j,i)/dx-(x0-1))+Z00
            tmpZ1 = (Z11-Z01)*(xRThetaLocation(j,i)/dx-(x0-1))+Z01
            RThetaData(k,j,i) = (tmpZ1-tmpZ0)*(yRThetaLocation(j,i)/dy-(y0-1))+tmpZ0

        end do
    end do
end do

return
end subroutine


!updating log
! v3.3
! - 新增subroutine interp3d，與SE無關，為方便管理，將所有的fortran集中到一個module
! v3.2
! - 新增側邊界的選擇，可選擇0邊界或是Neumann邊界
! v3.1
! - 改為可以直接被python呼叫的function
! v3.0
! - 使用等高座標的S-E eqn
! v2.2
! - 改變記憶體存取方式
! v2.1
! - 增加輸入初始場的功能，可輸入較接近收斂結果的初始場，減少疊代次數
! v2.0
! - 修正微分項
! - 增加斜壓性Bt、Bv的分別
! - 在等壓座標運算
! v1.2
! - 增加 S.O.R. 加速收斂 
! v1.0
! - 理想化軸對稱颱風模式的第一個版本