# Module torch_TAT

This module contains two routines that can be used for
a fast computation (*O(n^2 log n)* flops) of operators needed in
2D thermo- or photoacoustic tomography (TAT/PAT).
The computation is done
under the assumption of a constant speed of sound,
and circular acquisition geometry.
Namely, the forward, inverse and adjoint operators
are computed.

The module is implemented through the torch package.
It provides parallelization of the computation
either on a multiple core processor or, if a GPU card
is present, on a GPU.

While a fast inverse algorithm for the circular geometry has been reported before,
fast methods for computing the adjoint and forward operators
are new at the time of publication.
The algorithms for all three operators are designed
in such a way that the use of Hankel functions
is avoided. This is most important in the case
of the forward and adjoint operators, where
(in the most straightforward formulation) approximately
computed quantities would need to be multiplied by
Hankel functions of high orders. Such functions
grow very fast at low frequencies, creating
numerical instabilities. No Hankel functions
are used in the present module.

### Important limitation

The algorithms presented here utilize interpolations from a Cartesian grid
to a polar grid, and in the inverse direction. In order to achieve
an accurate result, the number of time steps **ntimes**, the dimension
**ndim** of the Cartesian grid (of size *(* **ndim** *x* **ndim** *)*, and the
number **ndet** of transducers (= detectors) should all be of the same order
of magnitude (prefereably several hundreds). Otherwise, accuracy of computations cannot
be guaranteed.

There two main routines in this module are **torch_forward** and **torch_adj_inv**.

## Routine "torch_forward"

Propagation of a pressure wave *p(t,x)* in compressible fluid with a
given speed of sound *C*, with initial condition *p(0,x) = f(x), p'(0,x) = 0*
is described by the solution of the Cauchy problem for the wave equation.
Function "torch_forward" computes this solution.

The trace of the solution is registered at **ndet** points uniformly spaced
on a circle of radius *R*=**raddet**.
If the initial condition *f(x)* is given on a Cartesian grid of size *(n x n)*,
the number of points (detectors) **ndet** is *O(n)* and the number of time steps
**ntimes** is also *O(n)*, then the computation requires  *O(n^2 log n)* flops.

The solution is computed for a time interval *[0, t_max]*. An important dimensionless parameter
of this problem is *t_rescaled = t_max * C /R*. Namely, the original problem is reduced
to a new problem, posed in the unit circle with a unit speed of sound and the new time
interval *[0, t_rescaled]*. Theoretically, the inverse problem of TAT/PAT can be
solved with *t_rescaled* equal to 1. However, this requires a specialized algorithm.
Most of the published algorithms require *t_rescaled* to be at least 2, with
more accurate reconstructions obtained with larger *t_rescaled*, for example 3 or 4.
The present algorithm assumes that *t_rescaled* is more than 2; if it is less,
additional zero-padding will be performed internally (but the returned array will
not contain additional zeros). Choosing parameters in such a way that "t_rescaled"
is larger than 6 will slow down the computation without bringing much benefit.



### Limitations
The computations is done under the assumption that **ndet** is *O(n)* and
**ntimes** is also *O(n)*. If one or both of these numbers are significantly smaller, the accuracy cannot
be guaranteed. In this case, the user may want to compute more values
than he/she needs, and then discard the unneeded ones.

Also, since the computation is Fast Fourier transform (FFT)-based, the parameters
of the computational greeds should be FFT-friendly, as described below.


### Calling the routine

The routine is called as follows:

#### proj =  torch_forward(device, init_cond, ndet, ntimes, tmax, raddet, speed_of_sound, lprecompute_forward)


Parameters:

**device : string**

Should be either "cpu" or "cuda", depending on the presence or absence of the GPU card.


**init_cond      : torch.tensor**

This is the input data, representing the initial condition *f(x)*
sampled by an *(n x n)* Cartesian grid
corresponding to the square *[-R,R]x[-R,R]*, where *R=* **raddet**.
The detector array is assumed to lie on the circle of radius **raddet**
centered at the origin.
IMPORTANTLY, the dimension *n* of the square torch tensor should
be ODD. Moreover, it is recommended that the integer number *n-1* is selected
to be FFT-friendly, i.e that it can be factored in small primes, for example *(n-1) =2^k^* or *(n-1)=3 x 2^k^*.

**ndet     : int**

number of detectors and of the columns in the output data array.
Detectors are assumed to be uniformly spaced over the circle inscribed
in the square defined by the array **init_cond**.
Detectors are numbered counterclockwise, with the first detector located at (1,0)
(i.e. 3 pm position).
For each detector, the routine will generate a time series, with the first data
corresponding to *t=0*, and the last to *t=* **tmax**.
The time discretization is uniform.

**ntimes   : int**

number of time samples, the first at *t=0*, the last at *t=* **tmax**

**tmax :  float**

time interval for which the values of the pressure on the detectors will be computed; the time step is chosen so that **tmax** is equal to *(* **ntimes** *-1) x* time step.

**raddet : float**

radius of the circle with the detectors

**speed_of_sound : float**

self-explanatory

**lprecompute_forward : boolean** (*True* or *False*)

if the routine is used for multiple computations of the forward problem with the same geometry, the computation can be significantly
sped-up by re-using the table of Bessel functions (and some internal variables) computed on the first run. Correspondingly, the first run of the routine should be
with **lprecompute_forward**=*True*, and the consecutive runs can have **lprecompute_forward**=*False*. However, if the parameters (time and/or geometry)
of the problem are changed, the Bessel functions should be recomputed by running the routine with **lprecompute_forward**=*True* again.


### Output:

The routine returns a torch tensor of size *(* **ntimes** *x* **ndet** *)* representing the computed pressure values
at the detectors' locations, at the uniform time steps. Each column of the tensor corresponds
to a particular detector, with the first row corresponding to *t=0*, and the last row representing time *t=* **tmax**.

----------------------------------------------------------------------------------------------

## Routine "torch_adj_inv"

This routine works in two different modes. If the **optype** parameter is set to "adjoint",
the routine will evaluate the result of application of the adjoint operator to data given
on the time/space cylinder. If the **optype** parameter is set to "inverse", the routine
evaluates the inverse operator given by the well-known "universal backprojection formula"
in 2D. The latter formula requires the data measured on the infinite interval in time
to be theoretically exact. However, if the rescaled time *t_rescaled* is larger than 2
(preferably 3 to 6), the error introduced by truncating the time interval is negligible
for most practical purposes. If *t_rescaled* is smaller than 2 the inversion will contain a
significant error.

Both the inverse and the adjoint algorithms are asymptotically fast: they take, roughly speaking,
*O(n^2 log n)* flops for *( O(n) x O(n))* data and *(n x n)* image.

Both algorithms are designed to be free of Hankel functions that otherwise could create numerical
instabilities. The algorithms are based on evaluating the single- or double- layer potentials
representing, respectively, the adjoint and the inverse, in the free 2D space.
This is done fast using the FFT techniques. Since FFT's represent periodic functions,
the computational domain is extended so that during the needed time interval *[0, t_rescaled]*
the reflections from the computational boundary do not reach the region of interest lying
inside the unit circle (in rescaled coordinates) or the circle of radius **radddet**
(in physical coordinates). The routine returns two square arrays: the larger one represents
the whole computed field (it is useful for checking that the size of the computational domain is chosen
correctly), and the smaller array of size **ndim x ndim** representing the part
of the computed solution lying in the circle of radius **raddet**. In the case
of the "inverse" mode this is the desired approximate reconstruction of the initial pressure.

### Limitations
The computations is done under the assumption that **ndet** is *O(n)* and
**ntimes** is also *O(n)*. If one or both of these numbers are significantly smaller, the accuracy cannot
be guaranteed. In this case, the user may want to compute more values
than he/she needs, and then discard the unneeded ones.

Also, since the computation is Fast Fourier transform (FFT)-based, the parameters
of the computational greeds should be FFT-friendly.



### Calling the routine

The routine is called as follows:

#### usmall = torch_adj_inv(device, ndet, ntimes, ndim, data, tmax, raddet, speed_of_sound, optype, lprecompute_adj, rad_corr)

**device : string**

Should be either "cpu" or "cuda", depending on the presence or absence of the GPU card.

**ndet     : int**

number of detectors and of the columns in the input data. Detectors are assumed to be uniformly spaced over
the circle of radius **raddet** centered at the origin. They are numbered counterclockwise, with the first
detector located at *(* **raddet** *,0)* (i.e. 3 pm position). If data only known on
a subset of a circle, zero data detectors should be added so that
detectors cover the circle uniformly.

**ntimes   : int**

number of time slices, the first at *t=0*, the last at *t=* **tmax**

**ndim : int, odd number!**

the image will be reconstructed on a square grid of size **ndim x ndim**,
corresponding to square *[-R,R]x[-R,R]*, where *R=* **raddet**.
Parameter **ndim** SHOULD BE ODD. This way there is a pixel at the position (0,0),
which is convenient. All the pixels outside of the circle of radius *R*
are set to 0. It is recommended that the integer number **ndim** *-1* is selected
to be FFT-friendly, i.e that it can be factored in small primes, for example *(* **ndim** *-1) =2^k^* or *(n-1)=3 x 2^k^*.


**data      : torch.tensor**

This is input data, representing the trace of the solution of the wave
equation on a circle of radius **raddet**. The number of columns
corresponds to the number of the detectors **ndet**.
Each column represents a time series, with the first data corresponding
to *t=0*, and the last to *t=* **tmax**. The time discretization is uniform.
Correspondingly, the number of rows in this tensor equals to **ntimes**.

**tmax :  float**

time interval on which data are given. **tmax** is equal to *(* **ntimes** *-1) x* time step.


**raddet : float**

radius of the circle with the detectors

**speed_of_sound : float**

self-explanatory

**optype : string** (*adjoint* or *inverse*)

the type of operator to compute, the adjoint or the inverse

**lprecompute_adj : boolean** (*True* or *False*)

if the routine is used for multiple computations of the adjoint or inverse operator with the same geometry, the computation can be significantly
sped-up by re-using the table of Bessel functions computed on the first run. Correspondingly, the first run of the routine should be
with **lprecompute_adj**=*True*, and the consecutive runs can have **lprecompute_adj**=*False*. However, if the parameters (time and/or geometry)
of the problem are changed, the Bessel functions should be recomputed by running the routine with **lprecompute_adj**=*True* again.
The table of the Bessel functions computed within this routine is independent and different from the one in the **torch_forward** routine,
so each of the routines requires its own initialization. On the other hand, within the present routine the same table is used
in the "adjoint" and "inverse" mode, so only one initialization is required if the geometry (including time) is not changed.
(Some other internal parameters are also precomputed, not only Bessel functions)


**rad_corr : float**

a property of the algorithm is that the constant component of the
image may contain a noticeable error (depending on discretization
parameter). In order to correct that, the module will add or subtract a
constant in such a way that the average of the reconstructed image in a
narrow ring of relative radius **rad_corr** is equal to 0.
For example, if it is known that the correct image should vanish close to
the detectors' circle, **rad_corr** should be set to 1.
If **rad_corr** is greater than 1 or less than 0, no correction is done (no
constant will be added).

### Output:

**usmall : torch.tensor**
tensor of size *(* **ndim** *x* **ndim** *)* representing either
the image reconstructed within the circle of radius **raddet**,
or the result of application of the adjoint, computed within
the same circle. All the values in the square array lying
outside of the inscribed circle are set to zero.


## Examples

In the subfolders, there are several examples illustrating the use of the functions contained
in the module **torch_TAT**.

### Forward operator

Subfolder **\demo_forward** contains script **demo_torch_forward.py**.
This is a demo program illustrating the use of the **torch_forward** function.
The initial pressure is modeled by the square *(* 257 *x* 257 *)* array that is read from the
file **micro_two_257.npy**.
First, the forward problem is computed three times (the results are identical).
On the first run the flag **lprecompute_forward** is set to True,
all the other runs are with **lprecompute_forward = False**.
After the first three runs, the computation is repeated a hundred times for
timing purposes.

A more accurate forward computation (with a **kWave**-like algorithm, not included here)
produces the file **accurate_full_data.npy**. The result of application of function **torch_forward**
is compared to the accurate result to evaluate the accuracy of the algorithm.
The comparison is done by the script **compare_res** (included). (The script should
be placed on the path found by Python....)


### Inverse operator

Subfolder **\demo_inverse** contains script **demo_inverse.py**.
This is a demo program illustrating the use for the **torch_adj_inv** function
for the solution of the inverse problem of TAT/PAT in circular geometry.

The ground truth (AKA initial pressure) is modeled by the square *(* 257 *x* 257 *)* array that
can be read from the file **micro_two_257.npy**.

The forward computation was done by a **kWave**-like algorithm (not included here)
with the result saved in the file **accurate_full_data.npy**.

The present demo script reads the forward data from the above file, and computes
the inverse, first two times and then a hundred times for the timing purposes.
The very first computation is done with the flag **lprecompute_adj = True**,
all consecutive runs are done with **lprecompute_adj = False**.

To obtain the inverse, parameter **optype** should be set to **"inverse"**.

The result of the image reconstruction is compared to the ground truth to evaluate
the accuracy of the reconstruction.
The comparison is done by the script **compare_res** (included). (This script should
lie on the path found by Python....)

### Iterations with forward and adjoint operators

Subfolder **\demo_iterations** contains script **demo_forward_adjoint.py**.
This is a demo program illustrating the use for the **torch_adj_inv** function
for computing the application of the adjoint operator to data given on the time-space cylinder,
and the use of the **torch_forward** to compute the forward problem of TAT/PAT.

This script uses forward/adjoint iterations to solve the inverse problem
of TAT/PAT with data known on a 180 degree arch (subset of the unit circle).

The ground truth (=initial pressure) is modeled by the square *(* 257 *x* 257 *)* array that
can be read from the file **micro57.npy**.
The data come from a forward computation that was done by a **kWave**-like algorithm (not included here)
with a  half of the data set to zero. The result is saved in the file **clean_180_data.npy**.

The present demo script reads the forward data from the above file, and solves the inverse
problem by iterating the forward/adjoint composition of operators.
The very first computation is done with the flags

   **lprecompute_adj = True**,

   **lprecompute_forward = True**.


All the consecutive calls to the forward and adjoint functions are done with these flags set to False.

To obtain the adjoint when calling **torch_adj_inv**, parameter **optype** should be set to **"adjoint"**.

The result of the image reconstruction can be compared to the ground truth **micro257.npy** to evaluate
the accuracy of the reconstruction.
The comparison can be done by the script **compare_res** (included). (This script should
lie on the path found by Python....)

Total computation time is reported when the script is completed.



### Additional Python scripts

Script **compare_res.py** requires two additional command
line parameters representing two file names.
Those files should contain numpy arrays of the same dimension.
The first file is considered an approximation, and the second
is considered the ground truth. The script outputs
the relative error computed in both the *L2*-norm and the *L-infinity*
norm.


