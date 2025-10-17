import numpy as np
import scipy as sp
import torch

from math import pi
from time import time

from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------------------------------
def find_fft_friendly(ngiven):

    numint = np.array(range(18))
    num2pow = 16 * (2 ** numint)
    num2pow3 = num2pow * 3 // 2
    index = 1 + np.array(range(18))
    goodnumbers = np.insert(num2pow, index, num2pow3)

    for j in goodnumbers:
        # print(j,ngiven-1)
        if (j >= ngiven - 1):
            break
    ngood = j + 1
    return ngood


# ---------------------------------------------------------------------------------------------------
def my_bilinear_intrp(data, hsizex, hsizey, xcoor, ycoor, lprecompute_forward, device):

    global a1x_f, a2x_f, nx_f, goodx_f, a1y_f, a2y_f, ny_f, goody_f

    if (lprecompute_forward):
        ndimy = data.size(dim=0)
        ndimx = data.size(dim=1)

        a1x_f, a2x_f, nx_f, goodx_f = my_search(xcoor, hsizex, ndimx, device)
        a1y_f, a2y_f, ny_f, goody_f = my_search(ycoor, hsizey, ndimy, device)

    res = (a1y_f* (a1x_f*data[ny_f, nx_f]       + a2x_f*data[ny_f, nx_f + 1])
         + a2y_f* (a1x_f * data[ny_f + 1, nx_f] + a2x_f*data[ny_f + 1, nx_f + 1]))

    res[(goodx_f == False) | (goody_f == False)] = 0.0
    return res


# ---------------------------------------------------------------------------------------------------
def my_quadrature(asy_step, length, n, b):

    bn = b ** n
    large_number = round(2. * length / asy_step)
    tryarr = np.array(range(large_number))
    eta = asy_step * np.array(range(large_number))
    etan = eta ** n

    xxx = eta * etan / (bn + etan)
    nodes = np.argmax(xxx > length)

    xxx = xxx[0:nodes]
    etan = etan[0:nodes]

    rjacob = etan * (bn * (n + 1) + etan) / ((bn + etan) * (bn + etan))
    return xxx, rjacob, nodes


# ---------------------------------------------------------------------------------------------------
def cart_to_polar_coor(xx, yy):  # ------------------------- begin cart_to_polar

    rad = torch.sqrt(xx * xx + yy * yy)
    angle = torch.arccos(xx / (rad + 1.e-300))
    angle[yy < 0.] = 2. * pi - angle[yy < 0.]
    return rad, angle  # ------------------------- end cart_to_polar


# ---------------------------------------------------------------------------------------------------
def my_search(x, hsize, ndim, device):

    shx = 2. * hsize / (ndim - 1)
    rnx = (x + hsize) / shx
    rnx_fl = torch.floor(rnx)
    nx = rnx_fl.type(torch.int32)

    dx = x + hsize - nx * shx
    a2 = dx / shx

    good = torch.ones(x.shape, dtype=torch.bool, device=device)
    good[(nx == 0) & (a2 < -.05)] = False
    good[(nx < 0)] = False
    good[(nx == ndim - 1) & (a2 > 0.05)] = False
    good[nx > ndim - 1] = False

    nx[nx < 0] = 0
    nx[nx > ndim - 2] = ndim - 2

    dx = x + hsize - nx * shx  # this is a repetion of the line above, it is needed for the case nx = ndim-1
    a2 = dx / shx
    a1 = 1.0 - a2
    return a1, a2, nx, good


# ---------------------------------------------------------------------------------------------------  pre-computing Bessel functions
def prepare_forward_bessels(ndet, num_freq, freq_scale_t, device):

    global bessels_forward, freq_array_forward, ikarray_forward, bessel_factor

    midk = round(ndet / 2)
    ind_grid = torch.linspace(0, midk, midk + 1, device=device)
    indices_torch, frequencies_torch = torch.meshgrid(ind_grid, freq_scale_t, indexing='xy')

    indices = indices_torch.cpu().numpy()
    frequencies = frequencies_torch.cpu().numpy()

    bess_temp = sp.special.jv(indices, frequencies)

    bess_flip = bess_temp[:, ::-1]
    bessels_forward = np.zeros((num_freq, ndet))
    bessels_forward[:, midk:] = np.copy(bess_temp[:, 0:midk])
    bessels_forward[:, 0:midk] = np.copy(bess_flip[:, 0:midk])
    bessels_forward = torch.from_numpy(bessels_forward).to(device)

    ones = torch.ones(ndet).to(device)
    freq_array_forward = torch.outer(freq_scale_t, ones)

                                                        # ------------------------------ now prepare the array of (-i)^k
    ikarray_forward = torch.zeros((num_freq, ndet), dtype=torch.complex64, device=device)

    ones = torch.ones(num_freq, device=device)

    ik = (-1j) ** ind_grid                               # ------------ this is (-i) in power 0,1,2,3,....
    ikarraytemp = torch.outer(ones, ik)
    ikflip = torch.flip(ikarraytemp, [1])
    ikarray_forward[:, midk:] = ikarraytemp[:, 0:midk]
    ikarray_forward[:, 0:midk] = ikflip[:, 0:midk]
    return


# ----------------------------------------------------------------------------------------------------
def my_search_from_zero(x, hsize, ndim, device):

    shx = hsize / (ndim - 1)

    rnx = x / shx
    rnx_fl = torch.floor(rnx)
    nx = rnx_fl.type(torch.int32)

    dx = x - nx * shx
    a2 = dx / shx

    good = torch.ones(x.shape, dtype=torch.bool, device=device)
    good[(nx == 0) & (a2 < -.05)] = False
    good[(nx < 0)] = False
    good[(nx == ndim - 1) & (a2 > 0.05)] = False
    good[nx > ndim - 1] = False

    nx[nx < 0] = 0
    nx[nx > ndim - 2] = ndim - 2

    dx = x - nx * shx  # this is a repetion of the line above, it is needed for the case nx = ndim-1
    a2 = dx / shx
    a1 = 1.0 - a2
    return a1, a2, nx, good


# -----------------------------------------------------------------------------------------------------------------
def my_bilinear_intrp_from_zero(data, hsizex, hsizey, xcoor, ycoor, lprecompute_adj, device):

    global a1x, a2x, nx, goodx, a1y, a2y, ny, goody

    if (lprecompute_adj):
        ndimy = data.size(dim=0)
        ndimx = data.size(dim=1)
        a1x, a2x, nx, goodx = my_search_from_zero(xcoor, hsizex, ndimx, device)
        a1y, a2y, ny, goody = my_search_from_zero(ycoor, hsizey, ndimy, device)

    res = (a1y*(a1x * data[ny, nx]     + a2x*data[ny, nx + 1])
         + a2y*(a1x * data[ny + 1, nx] + a2x*data[ny + 1, nx + 1]) )

    res[(goodx == False) | (goody == False)] = 0.0
    return res


# ------------------------------------------------------------------------------------------------------------------
def prepare_bessels(ndet, freq_radii, optype):   # ---- prepare_bessels: pre-computing Bessel functions and their first derivatives

    global bessels, freq_array, ikarray, bess_ders, factor_adj_t, factor_inv_t

    midk = round(ndet / 2)
    numfreq = np.size(freq_radii)
    ind_grid = np.array(range(midk))

    indices, frequencies = np.meshgrid(ind_grid, freq_radii)

    bess_temp = sp.special.jv(indices, frequencies)
    bess_flip = bess_temp[:, ::-1]
    bessels = np.zeros((numfreq, ndet))
    bessels[:, midk:] = np.copy(bess_temp)
    bessels[:, 1:midk] = np.copy(bess_flip[:, 0:midk - 1])

    ones = np.ones(ndet)
    freq_array = np.outer(freq_radii, ones)
                                                        # ------------------------------ now prepare the array of (-i)^k
    ikarray = np.zeros((numfreq, ndet), dtype='complex')
    ones = np.ones(numfreq)

    minusik = (-1j) ** ind_grid  # ------------ this is (-i) in power 0,1,2,3,....
    ikarraytemp = np.outer(ones, minusik)
    ikflip = ikarraytemp[:, ::-1]
    ikarray[:, midk:] = np.copy(ikarraytemp)
    ikarray[:, 1:midk] = np.copy(ikflip[:, 0:midk - 1])

                                                         # ---- finding the derivatives of Bessel functions, needed for the inverse operator
    bess_der_temp = np.zeros(np.shape(bess_temp))
    bess_der_temp[:, 0] = - bess_temp[:, 1]  # since J0'=-J1
    indices_der = range(midk)
    indices_der = indices_der[1:]

    freq_radii[0] = 1.e-30
    for k in indices_der:
        bess_der_temp[:, k] = bess_temp[:, k - 1] - k * bess_temp[:, k] / freq_radii

    bess_der_temp[0, :] = 0.
    bess_der_temp[0, 1] = 0.5
    bess_der_flip = bess_der_temp[:, ::-1]
    bess_ders = np.zeros((numfreq, ndet))
    bess_ders[:, midk:] = np.copy(bess_der_temp)
    bess_ders[:, 1:midk] = np.copy(bess_der_flip[:, 0:midk - 1])
    return


# ================================================================================================================================================================
def torch_forward(device, init_cond, ndet, ntimes, tmax, raddet, speed_of_sound, lprecompute_forward):

    ldebug = False
    lcorr = True  # it works well, why make it optional?
    linfo = False

    global bessels_forward, freq_array_forward, ikarray_forward, bessel_factor
    global a1x_f, a2x_f, nx_f, goodx_f, a1y_f, a2y_f, ny_f, goody_f
    global bessels, freq_array, ikarray, bess_ders, factor_adj_t, factor_inv_t

    start_time = time()

    nrow = init_cond.size(dim=0)
    ncol = init_cond.size(dim=1)

    #-------------------- some silly checks -------------------------------------------
    if (ncol != nrow):
        print('Check init. cond. size, the array should be square ')
        quit()
    else:
        ndim = ncol

    if ndim % 2 != 1:
        print('ndim should be an odd number !')
        quit()
    # ------------------------------- figuring out good computational grids ---------

    t_rescaled_wanted = tmax / (raddet / speed_of_sound)
    if (linfo):
        print('t_rescaled wanted = ', t_rescaled_wanted)
    tstep = t_rescaled_wanted / (ntimes - 1)
    if (t_rescaled_wanted < 3.):
        t_rescaled_temp = 6.
    else:
        t_rescaled_temp = 2. * t_rescaled_wanted

    newtimes_temp = t_rescaled_temp / tstep
    newtimes = find_fft_friendly(newtimes_temp)

    t_rescaled = (newtimes - 1) * tstep

    if (linfo):
        print('t_rescaled chosen = ', t_rescaled, '  newtimes = ', newtimes)

    halfsize_large_needed = 1.5 * (t_rescaled / 2 + 1)       #   this is a rather arbitrary choice, but it works
    halfsize = 1.                                            # the radius of the square containing the detectors and the object

    xstep = 2. * halfsize / (ndim - 1)
    nsize_needed = int((ndim - 1) * halfsize_large_needed) + 1
    ndiml = find_fft_friendly(nsize_needed)
    largehalf = 0.5 * xstep * (ndiml - 1)

    if (linfo):
        print('ndiml = ', ndiml, '  largehalf = ', largehalf)

    nlhalf = round((ndiml - 1) / 2)
                                                             # ---------- figuring frequency domains
    spatial_freq_step = 2.0 * pi / (2. * largehalf)          # ------- step of the Cartesian discretization of the Fourier space
    polar_freq_step = 2.0 * pi / t_rescaled                  # ------- radial step of the polar grid in the Fourier domain

    num_freq = int((newtimes - 1) / 2)
    freq_scale_t = polar_freq_step * torch.tensor(range(num_freq)).to(device)  # this is time frequency scale !
    freq_scale = freq_scale_t.cpu().numpy()

    freq_max = (num_freq - 1) * polar_freq_step

    if (linfo):
        print('max polar frequency = ', freq_max)

    det_angle_ext = torch.linspace(0., 2 * pi, ndet + 1, device=device)  # ---------- detector geometry
    det_angle_torch = det_angle_ext[0:ndet]  # polar angle coordinate for the detectors
    det_angle = det_angle_torch.cpu().numpy()

    # stepf = pi/largehalf        #---- this is the frequency step in the Fourier transform of the zero padded initial condition
    frmax = nlhalf * spatial_freq_step  # max Cartesian frequency i.e. Nyquist

    if (linfo):
        print('max Cartesian frequency = ', frmax)
    # -------------------------------------------- starting computations ------------
    startFFTtime = time()

    ioffset = round((ndiml - ndim) / 2)
    ileft = ioffset
    iright = ioffset + ndim

    init_large = torch.zeros((ndiml - 1, ndiml - 1), device=device)
    init_large[ileft:iright, ileft:iright] = init_cond

    # -------------------------------------- Fourier transform of the initial condition ------------

    shifted_f = torch.fft.fftshift(init_large, [0, 1])
    f_trans0 = torch.fft.fft2(shifted_f)
    f_trans1 = torch.fft.fftshift(f_trans0, [0, 1])

    f_trans = torch.zeros((ndiml, ndiml), dtype=torch.complex64, device=device)
    f_trans[0:ndiml - 1, 0:ndiml - 1] = f_trans1

    cartfft_time = time()

    if (linfo):
        print('Cartesian FFT ', '{:05.2f}'.format(cartfft_time - startFFTtime), ' sec')

    if (ldebug):
        np.save('ftrans_good_r', np.real(f_trans.cpu().numpy()))
        np.save('ftrans_good_i', np.imag(f_trans.cpu().numpy()))

    # ------------------------ interpolate to the polar grid in the Fourier space --------------
    maxint = num_freq
    for j in range(num_freq):
        if (freq_scale[j] > frmax):
            maxint = j
            break

    freq_scale_small_t = freq_scale_t[:maxint]
    freq_scale_small = freq_scale_small_t.cpu().numpy()
    freq_angle_t = det_angle_torch

    pol_r, pol_tet = torch.meshgrid(freq_scale_small_t, freq_angle_t, indexing='ij')
    pol_x = pol_r * torch.cos(pol_tet)
    pol_y = pol_r * torch.sin(pol_tet)

    hsize = frmax
    pol_val = my_bilinear_intrp(f_trans, frmax, frmax, pol_x, pol_y, lprecompute_forward, device)

    interp_time = time()
    if (linfo):
        print('Linear interpolation  ', '{:05.2f}'.format(interp_time - cartfft_time), ' sec')

    if (ldebug):
        pol_val1 = pol_val.data.cpu().numpy()
        np.save('pol_trans_r', np.real(pol_val1))
        np.save('pol_trans_i', np.imag(pol_val1))

    # ------------------------- angular FFT ----------------------------------------------------------
    hank_tran = torch.fft.fftshift(torch.fft.fft(pol_val, axis=1), [1])
    fft_time = time()

    if (linfo):
        print('Polar FFT ', '{:05.2f}'.format(fft_time - interp_time), ' sec')

    nzero_harm = round(ndet / 2)
    if (lcorr):
        zero_harm = np.real(np.copy(hank_tran[:, nzero_harm].cpu().numpy()))  # saving the zero circular harmonics
        hank_tran[:, nzero_harm] = 0.  # removing the zero circular

        harm_one = hank_tran[:, nzero_harm + 1].cpu().numpy()  # saving the first circular harmonics
        hank_tran[:, nzero_harm + 1] = 0.
        hank_tran[:, nzero_harm - 1] = 0.

    if (ldebug):
        np.save('hank_tran_r', np.real(hank_tran))
        np.save('hank_tran_i', np.imag(hank_tran))

    # ------------------------ prepare Bessels ------------------------------------------------------
    if (lprecompute_forward):
        prepare_forward_bessels(ndet, maxint, freq_scale_small_t, device)

        bessel_factor = bessels_forward * freq_array_forward * ikarray_forward

    bessel_time = time()

    if (lprecompute_forward):
        if (linfo):
            print('Prepare Bessels ', '{:05.2f}'.format(bessel_time - fft_time), ' sec')

    # ------------------------------ multiply -------------------------------------------------------
    hank_tran = hank_tran * bessel_factor  # s_forward*freq_array_forward*ikarray_forward)

    if (ldebug):
        cyl_trans = hank_tran.data.cpu().numpy()
        np.save('cyl_trans_h_r', np.real(cyl_trans))
        np.save('cyl_trans_h_i', np.imag(cyl_trans))

    # ---------- FFT in angular variable ---------------------------------------------------
    cyl_tran = torch.fft.fftshift(hank_tran, [1])
    cyl_in_freq_prep = torch.fft.fftshift(torch.fft.ifft(cyl_tran, axis=1), [1])
    cyl_in_freq = torch.zeros((2 * num_freq, ndet), device=device)

    n_upperlimit = min(maxint, num_freq)

    cyl_in_freq[:n_upperlimit, :] = torch.real(cyl_in_freq_prep)

    angfft_time = time()
    if (linfo):
        print('Angular FFT ', '{:05.2f}'.format(angfft_time - bessel_time), ' sec')

    # ---------- FFT transform from frequencies to time domain -----------------------------

    coefficient = 4. / (ndim - 1) / (ndim - 1) / t_rescaled

    cylinder = coefficient * torch.real(torch.fft.fft(cyl_in_freq, axis=0))

    if (ldebug):
        np.save('full_cyl_h_r_fft', cylinder)

    fft2_time = time()
    if (linfo):
        print('Final FFT ', '{:05.2f}'.format(fft2_time - angfft_time), ' sec')

    # ------ slow computation for harmonics -1,0,1 ------------------------------------
    if (lcorr):
        nexp = 4.
        quad_factor = 0.5

        b = 0.25 * polar_freq_step * (n_upperlimit - 1)
        asy_step = quad_factor * polar_freq_step

        length = polar_freq_step * (n_upperlimit - 1)
        new_freqs, rjacob, newn = my_quadrature(asy_step, length, nexp, b)

        # ---------- preparing bessel functions on a non-uniform frequency grid
        bess0 = sp.special.j0(new_freqs)
        bess1 = sp.special.j1(new_freqs)

        bessels0 = 2. * bess0 * new_freqs  # -------- differentiation without i
        bessels1 = bess1 * new_freqs  # -------- differentiation without i

        # ---- interpolating zero harmonic to the new frequency grid

        interpolator = interp1d(freq_scale[:n_upperlimit], np.real(zero_harm), kind='cubic', bounds_error='False',
                                fill_value=0.)
        hantransform0 = interpolator(new_freqs)

        interpolator1 = interp1d(freq_scale[:n_upperlimit], harm_one, kind='cubic', bounds_error='False', fill_value=0.)
        hantransform1 = interpolator1(new_freqs)

        # ------ preparing data to Fourier transform in time

        zero_Fourier = hantransform0 * rjacob * bessels0
        one_Fourier = hantransform1 * rjacob * bessels1

        one_Four_r = np.real(one_Fourier)
        one_Four_i = np.imag(one_Fourier)
        # ------- transform to torch
        zero_Fourier_torch = torch.from_numpy(zero_Fourier.astype("csingle")).to(device)
        one_Four_r_torch = torch.from_numpy(one_Four_r.astype("csingle")).to(device)
        one_Four_i_torch = torch.from_numpy(one_Four_i.astype("csingle")).to(device)

        new_freqs_torch = torch.from_numpy(new_freqs.astype("float32")).to(device)
        time_grid_torch = torch.linspace(0., t_rescaled_wanted, ntimes, device=device)

        new_fs_torch, times_torch = torch.meshgrid(new_freqs_torch, time_grid_torch, indexing='xy')

        arg_array = times_torch * new_fs_torch
        cos_array = torch.cos(arg_array)
        sin_array = torch.sin(arg_array)

        exp_array_torch = cos_array - 1j * sin_array  # making of an array of Fourier exponents

        timeseries_t = torch.reshape(torch.real(torch.matmul(exp_array_torch, zero_Fourier_torch)), (ntimes, 1))
        timeseries1r_t = torch.reshape(torch.real(torch.matmul(exp_array_torch, one_Four_r_torch)), (ntimes, 1))
        timeseries1i_t = torch.reshape(torch.real(torch.matmul(exp_array_torch, one_Four_i_torch)), (ntimes, 1))

        many_det_series_torch = timeseries_t.repeat(1, ndet)
        cos_det_series_torch = torch.matmul(timeseries1i_t, torch.reshape(torch.cos(det_angle_torch),
                                                                          (1, ndet)))  # --- making cosine in angle
        sin_det_series_torch = torch.matmul(timeseries1r_t, torch.reshape(torch.sin(det_angle_torch),
                                                                          (1, ndet)))  # --- making sine in angle

        harm_coef = 2. * quad_factor / ndet / (ndim - 1) / (
                    ndim - 1) / t_rescaled  # additional /ndet compared with write-up is because this harmonics is not processed through
        #  the inverse FFT in angle

        correction_torch = harm_coef * (many_det_series_torch - 4. * cos_det_series_torch - 4. * sin_det_series_torch)

        zero_harm_time = time()

        if (linfo):
            print('Recomputing the zero and first harmonics ', '{:05.2f}'.format(zero_harm_time - fft2_time), ' sec')

        if (ldebug):
            np.save('zero_harm', many_det_series_torch.data.cpu().numpy())
            np.save('cos_harm', cos_det_series_torch.data.cpu().numpy())
            np.save('sin_harm', sin_det_series_torch.data.cpu().numpy())

    # --------------------------------------------------------------------------------------------------------

    proj = cylinder[:ntimes, :]
    if (lcorr):
        proj = proj + correction_torch

    const_corr = torch.mean(proj[0, :])
    proj = proj - const_corr
    proj[0, :] = 0.

    endtime = time()

    if (linfo):
        print('Total time ', '{:05.2f}'.format(endtime - start_time), ' sec')

    return proj


# ================================================================================================================================================================
def torch_adj_inv(device, ndet, ntimes, ndim, data, tmax, raddet, speed_of_sound, optype, lprecompute_adj, rad_corr):

    global bessels, freq_array, ikarray, bess_ders, factor_adj_t, factor_inv_t

    # ------------------------------ silly check -----------------------------

    linfo = False

    nrow = data.size(dim=0)
    ncol = data.size(dim=1)

    if (nrow != ntimes):
        print('Wrong times !')
        quit()

    if (ncol != ndet):
        print('Wrong number of detectors !')
        quit()

    if ndim % 2 != 1:
        print('ndim should be an odd number !')
        quit()

    if (optype == 'inverse'):
        if (linfo):
            print('Inverting   ...')
    else:
        if (optype == 'adjoint'):
            if (linfo):
                print('Computing adjoint   ...')
        else:
            print('optype should be either "inverse" or "adjoint"')
            quit()

    # --------------------------- define the geometry and the timing of the problem -------------------------------------------------
    start_time = time()

    t_rescaled = tmax / (raddet / speed_of_sound)
    tstep = t_rescaled / (ntimes - 1)

    t_wanted = 4. * t_rescaled  # ------------- this may be increased for higher accuracy but a slower run time
    if (t_wanted < 2.1):
        t_wanted = 2.1

    ntimes_wanted = t_wanted / tstep + 1

    newtimes = find_fft_friendly(ntimes_wanted)

    tnew = (newtimes - 1) * tstep

    if (linfo):
        print('newtimes = ', newtimes, '   tnew = ', tnew)
    if (tnew < 2.0):
        print(' ')
        print('================================================')
        print('Using rescaled t less then 2 is not recommended!')
        print('================================================')
        print(' ')

    cdata_t = torch.zeros((newtimes, ndet), dtype=torch.complex64, device=device)

    if (newtimes > ntimes):
        cdata_t[0:ntimes, :] = data
    else:
        if (newtimes < ntimes):
            cdata_t = data[0:newtimes, :]
            print('This should not happen at all !!!!!')
            quit()
        else:
            cdata_t = data

    halfsize_large_needed = 1.1 + t_rescaled  # ????????????
    # print('halfsize_large_needed = ',halfsize_large_needed)

    halfsize = 1.  # the radius of the square containing the detectors and the object

    xstep = 2. * halfsize / (ndim - 1)

    nsize_needed = int((ndim - 1) * halfsize_large_needed) + 1
    # print('nsize_needed = ',nsize_needed)

    ndiml = find_fft_friendly(nsize_needed)

    largehalf = 0.5 * xstep * (ndiml - 1)
    # print('ndiml = ',ndiml)

    med = round((ndim + 1))
    nlhalf = round((ndiml - 1) / 2)

    ndet_half = round(ndet / 2)

    spatial_freq_step = 2.0 * pi / (2. * largehalf)  # ------- step of the Cartesian discretization of the Fourier space

    polar_freq_step = pi / tnew  # ------- radial step of the polar grid in the Fourier domain

    freq_scale = np.linspace(-nlhalf, nlhalf, ndiml) * spatial_freq_step

    freq_scale = freq_scale[0:ndiml - 1]
    freq_scale_t = torch.from_numpy(freq_scale).to(device)

    freq_max = -freq_scale[0]
    # print('Max Cartesian frequency: ',freq_max)

    det_angle_ext = np.linspace(0., 2 * pi, ndet + 1)                    # ---------- detector geometry
    det_angle = np.copy(det_angle_ext[0:ndet])                           # polar angle coordinate for the detectors

    times = np.linspace(0., tnew, ntimes)  # -------------- time grid

    # -------------------------------------- making the Cartesian grid in the Fourier domain ----
    ksi1, ksi2 = torch.meshgrid(freq_scale_t, freq_scale_t, indexing='xy')
    freq_rho, freq_teta = cart_to_polar_coor(ksi1, ksi2)

    # -------------------------------------- making polar grids in the Fourier domain -----------
    numfreq = newtimes
    numfreq_small = int(1.5 * freq_max / polar_freq_step)
    if (numfreq_small > newtimes):
        numfreq_small = newtimes
    if (linfo):
        print('Numfreq_small = ', numfreq_small)

    freq_radii = np.array(range(numfreq)) * polar_freq_step                # ----- discretizing polar distance in the Fourier space
    freq_radii_small = freq_radii[0:numfreq_small]

    maxpolarfr = freq_radii[numfreq - 1]
    maxpolarfr_small = freq_radii_small[numfreq_small - 1]

    # ------------------------ preparing Bessel functions and (in the case of inverse operator) their derivatives ---------------------------------------
    if (lprecompute_adj):
        besselstarttime = time()

        prepare_bessels(ndet, freq_radii_small, optype)

        factor_adj = sp.fft.fftshift(bessels * ikarray, [1])
        factor_inv = - 2. * sp.fft.fftshift(bess_ders * ikarray, [1])

        factor_adj_t = torch.from_numpy(factor_adj).to(device)
        factor_inv_t = torch.from_numpy(factor_inv).to(device)

        besseltime = time() - besselstarttime

        if (linfo):
            print('Preparing Bessels functions', '{:05.2f}'.format(besseltime), ' sec')

    # ------------------------------ computing Fourier series of the data in the angular direction only ---------------------------
    startfft = time()
    cdataseries = torch.fft.fft(cdata_t, axis=1)
    endfft = time()
    if (linfo):
        print('Expanding in Fourier series', '{:05.2f}'.format(endfft - startfft), ' sec')

    # ------------------------------------- forming solution in Fourier space in polar coordinates
    startprop = time()

    cdata_series_long = torch.zeros((2 * (newtimes - 1), ndet), dtype=torch.complex64, device=device)

    if (optype == 'adjoint'):
        cdata_series_long[:newtimes, :] =     cdataseries[:, :]
        creverse = torch.resolve_conj(torch.flip(cdataseries, [0]))
        cdata_series_long[newtimes:, :] = creverse[1:newtimes - 1, :]  # this implements a cos transform through a double size FFT

        res = torch.fft.ifft(cdata_series_long, dim=0, norm=None)      # summation of cosines with different frequencies using the inverse FFT transform
        cpolar_tr_t = res[:numfreq_small, :]
        cpolar_tr_t = cpolar_tr_t * factor_adj_t

    if (optype == 'inverse'):
        cdata_series_long[1:newtimes - 1, :] = -1j * cdataseries[1:-1, :]
        creverse = torch.resolve_conj(torch.flip(cdataseries, [0]))
        cdata_series_long[newtimes:, :] = 1j * creverse[1:newtimes - 1,:]     # this implements a sine transform through a double size FFT

        res = torch.fft.ifft(cdata_series_long, dim=0, norm=None)        # summation of sines with different frequencies using the inverse FFT transform
        cpolar_tr_t = res[:numfreq_small, :]
        cpolar_tr_t = cpolar_tr_t * factor_inv_t

    endprop = time()

    if (linfo):
        print('Applying the propagator', '{:05.2f}'.format(endprop - startprop), ' sec')

    # --------------------------------------------------------------------------  summing up the Fourier series in angle

    startfft = time()

    lmult = 2

    cpolar_tr_new = torch.zeros((newtimes, lmult*ndet), dtype=torch.complex64, device=device)
    cpolar_tr_new[:numfreq_small,0:ndet_half]                         = cpolar_tr_t[:, 0:ndet_half]
    cpolar_tr_new[:numfreq_small,lmult*ndet - ndet_half:lmult * ndet] = cpolar_tr_t[:, ndet - ndet_half:ndet]

    ctran_new_t = lmult*torch.fft.ifft(cpolar_tr_new, axis=1)

    endfft = time()
    if (linfo):
        print('Summing up the Fourier series', '{:05.2f}'.format(endfft - startfft), ' sec')

    # ------------------------------------ interpolation  from the polar grid to Cartesian in the Fourier domain ------
    ctran_ext = torch.zeros((numfreq_small, lmult * ndet + 1), dtype=torch.complex64, device=device)

    ctran_ext[:, 0:lmult * ndet] = ctran_new_t[:numfreq_small, :]
    ctran_ext[:, lmult * ndet]   = ctran_ext[:, 0]

    stop_prep_spline_time = time()

    freq_rho_half  = freq_rho [0:nlhalf + 1, :]
    freq_teta_half = freq_teta[0:nlhalf + 1, :]
    radfr = (numfreq_small - 1) * polar_freq_step

    half_cart_transform = my_bilinear_intrp_from_zero(ctran_ext, 2. * pi, radfr, freq_teta_half, freq_rho_half, lprecompute_adj, device)

    if (linfo):
        print(half_cart_transform.dtype)

    if (linfo):
        print('half_cart_transform = ', half_cart_transform.size(dim=0), half_cart_transform.size(dim=1), )
        print('ctran_ext           = ', ctran_ext.size(dim=0), ctran_ext.size(dim=1))

    fliptr = torch.flip(half_cart_transform, [0, 1])
    cart_transform_t = torch.zeros((ndiml - 1, ndiml - 1), dtype=torch.complex64, device=device)
    cart_transform_t[0:nlhalf + 1, :] = half_cart_transform
    cart_transform_t[nlhalf + 1:, 1:] = torch.conj(fliptr[1:-1, :-1])

    spline_time = time()

    if (linfo):
        print('Evaluation of splines ', '{:05.2f}'.format(spline_time - stop_prep_spline_time), ' sec')

    # -------------------------------------- Fourier transform back to the plane t=T ------------

    startfft = time()

    shifted_u_trans = torch.fft.fftshift(cart_transform_t, [0, 1])
    four_coef = (ndiml - 1) * pi / largehalf / (xstep / tstep) * (newtimes - 1)

    u_spatial_t = torch.fft.fftshift(torch.fft.ifft2(shifted_u_trans), [0, 1]) * four_coef
    endfft = time()

    if (linfo):
        print('Completing 2D FFT', '{:05.2f}'.format(endfft - startfft), ' sec')

    # u_spatial = u_spatial_t.data.cpu().numpy()

    # -------------------- Finding the restriction of u(t,x) to the disk inside the detector circle ----------------------

    ioffset = round((ndiml - ndim) / 2)
    ileft = ioffset
    iright = ioffset + ndim
    usmall = torch.real(torch.clone(u_spatial_t[ileft:iright, ileft:iright]))

    x1 = torch.linspace(-1., 1., ndim)
    x2 = x1
    xx1, xx2 = torch.meshgrid(x1, x2, indexing='ij')

    radius = torch.sqrt(xx1 * xx1 + xx2 * xx2)

    if (optype == 'inverse' and rad_corr <= 1. and rad_corr > 0.):  #--------additional correction if "inverse"
        aaa = torch.zeros((ndim, ndim), device=device)
        aaa[radius > rad_corr - 2. * xstep] = 1.
        aaa[radius > rad_corr] = 0.

        number_of_pixels = torch.sum(aaa)
        average = torch.sum(aaa * usmall) / number_of_pixels

        usmall = usmall - average
        if (linfo):
            print('average = ', average)
    #-----------------------------------------------------------------------------------------

    usmall[radius >= raddet] = 0.

    final_time = time() - start_time
    if (linfo):
        print('Total time ', '{:05.2f}'.format(final_time), ' sec')

    # return usmall, torch.real(u_spatial_t)
    return usmall
