from numpy.polynomial import Polynomial
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import Akima1DInterpolator
from scipy.interpolate import PchipInterpolator
import scipy.interpolate as interpolate

import numpy as np
def do_extrapolation(input_lane, sampling_pts, mode, degree=3,knots = None,k=3):
    """
    mode = ['linear', 'polynomial','cubic_spline','b-spline','akima_spline','pchip']
    
    """
    # input_y = input_lane[:, 1][np.unique(input_lane[:, 1], return_index=True)]
    # input_lane = input_lane[np.unique(np.linspace(0,len(input_lane)-1,num=min(100,len(input_lane)-1),dtype=np.uint8)),:]
    sorted_idx = input_lane[:, 1].argsort(0)
    input_y = input_lane[:, 1][sorted_idx].reshape(-1)
    input_x = input_lane[:, 0][sorted_idx].reshape(-1)
    input_z = input_lane[:, 2][sorted_idx].reshape(-1)
    unique_idx = np.unique(input_y,return_index=True)[1]
    input_y = input_y[unique_idx]
    input_z = input_z[unique_idx]
    input_x = input_x[unique_idx]

    # input_x = input_lane[:, 1].reshape(-1).astype(np.float16)
    # input_x = input_lane[:, 0].reshape(-1).astype(np.float16)
    # input_z = input_lane[:, 2].reshape(-1).astype(np.float16)
    # f_z = interp1d(input_y, input_z, fill_value="extrapolate")
    if mode == 'linear':
        f_x = interp1d(input_y, input_x, fill_value="extrapolate")
        f_z = interp1d(input_y, input_z, fill_value="extrapolate")
    elif mode == 'polynomial':
        '''
        전체 포인트들이 n차식이라 가정한다
        '''
        f_x = Polynomial.fit(input_y, input_x, degree)
        f_z = Polynomial.fit(input_y, input_z, degree)
    elif mode == 'cubic_spline':
        '''
        각 포인트들 사이 관계를 3차식을 가정
        '''
        f_x = CubicSpline(input_y, input_x, bc_type='not-a-knot')
        f_z = CubicSpline(input_y, input_z, bc_type='not-a-knot')
    elif mode == 'b-spline':
        '''
        basic-spline
        '''
        f_x = InterpolatedUnivariateSpline(input_y, input_x, k=1)
        f_z= InterpolatedUnivariateSpline(input_y, input_z, k=1)
    elif mode == 'akima_spline':
        '''
        각 점에서 기울기를 이용하여 구간의 다항식을 결정한다.
        복잡한 변화에 장점을 가진다.
        '''
        f_x = Akima1DInterpolator(input_y, input_x,extrapolate=True)
        f_z = Akima1DInterpolator(input_y, input_z,extrapolate=True)
    elif mode == 'pchip':
        '''
        Piecewise Cubic Hermite Interpolating Polynomial
        각 구간을 3차 Hermite 다항식을 이용한다. 
        추가로 각 점에서의 기울기도 이용한다.
        단조로운 변화에 장점을 가진다.
        '''
        f_x = PchipInterpolator(input_y, input_x)
        f_z = PchipInterpolator(input_y, input_z)
    elif mode == 'custom_spline':
        if knots is None:
            knots = np.array([])
        tck_x = interpolate.splrep(input_y,input_x,t = knots,k=k)
        tck_z = interpolate.splrep(input_y,input_z,t = knots,k=k)

    # if mode == 'b-spline':
    #     x_new = interpolate.splev(sampling_pts,f_x)
    #     z_new = interpolate.splev(sampling_pts,f_z)
    # else:
    if mode == 'custom_spline':
        x_new = interpolate.splev(sampling_pts,tck_x)
        z_new = interpolate.splev(sampling_pts,tck_z)
        return x_new,sampling_pts,z_new
    x_new = f_x(sampling_pts)
    z_new = f_z(sampling_pts)
    # if mode != 'linear':
    #     x_2d = f_x.derivative(n=2)(sampling_pts)
    #     return x_new, sampling_pts, z_new, np.mean(np.abs(x_2d))
    # sampling_pts = np.array(sampling_pts)
    # return x_new, sampling_pts, z_new, np.mean(np.abs(np.gradient(np.gradient(x_new,sampling_pts.reshape(-1),axis=0),sampling_pts.reshape(-1),axis=0)))
    return x_new, sampling_pts, z_new


