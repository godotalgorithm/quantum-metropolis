// numerical verification of empirical error bound in Eq. (B6) of "Measurement-Based Quantum Metropolis Algorithm"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

// ASSUMPTIONS:
// - the optimization over delta is restricted to only two values (detectable false negatives are possible)
// - residual extrema are assumed to be bracketed on a coarse grid (missed maxima can cause a silent false positive)
// - coarse grid is uniformly distributed over the domain (residual extrema of Fourier approximants are typically uniformly distributed)
// - number of points on the coarse grid is proportional to the approximant order (& thus expected number of local extrema)
// - parameters are checked with uniform coverage of bounds on a log scale w/ adjustable grid spacing
// - the maximum t_max value is limited to avoid numerical problems resolving residual errors below machine precision

// truncated Fourier coefficients for the Gaussian approximation in Eq. (B5)
void fourier_fit(int r, int n, double t_max, double complex *coeff)
{
    int ncoeff = 1<<n;
    int nfilter = 1<<r;
    for(int i=0 ; i<ncoeff ; i++)
    {
        double t = t_max*(double)(2*i + 1 - ncoeff)/(double)ncoeff;

        coeff[i] = 0.0;
        for(int j=0 ; j<nfilter ; j++)
        {
            double w = (0.5*M_PI/t_max)*(double)(2*j + 1 - nfilter);
            coeff[i] += cexp(-I*w*t - 0.25*w*w*t_max/M_PI);
        }
        coeff[i] /= (double)ncoeff;
    }
}

// Gaussian function target in Eq. (B6) & its analytical w derivatives
double target(double w, double t_max)
{ return exp(-0.25*w*w*t_max/M_PI); }
double dtarget(double w, double t_max)
{ return (-0.5*w*t_max/M_PI)*exp(-0.25*w*w*t_max/M_PI); }
double d2target(double w, double t_max)
{ return (0.5*w*w*t_max/M_PI - 1.0)*(0.5*t_max/M_PI)*exp(-0.25*w*w*t_max/M_PI); }

// Gaussian function approximant in Eq. (B6) & its analytical w derivatives
double approx(double w, double t_max, int n, double complex *coeff)
{
    int ncoeff = 1<<n;
    double complex func = 0.0;
    for(int i=0 ; i<ncoeff ; i++)
    {
        double t = t_max*(double)(2*i + 1 - ncoeff)/(double)ncoeff;
        func += coeff[i]*cexp(I*w*t);
    }
    return creal(func);
}
double dapprox(double w, double t_max, int n, double complex *coeff)
{
    int ncoeff = 1<<n;
    double complex func = 0.0;
    for(int i=0 ; i<ncoeff ; i++)
    {
        double t = t_max*(double)(2*i + 1 - ncoeff)/(double)ncoeff;
        func += coeff[i]*I*t*cexp(I*w*t);
    }
    return creal(func);
}
double d2approx(double w, double t_max, int n, double complex *coeff)
{
    int ncoeff = 1<<n;
    double complex func = 0.0;
    for(int i=0 ; i<ncoeff ; i++)
    {
        double t = t_max*(double)(2*i + 1 - ncoeff)/(double)ncoeff;
        func -= coeff[i]*t*t*cexp(I*w*t);
    }
    return creal(func);
}

// residual error optimand from minimax problem in Eq. (B6)
double residual(double w, double t_max, double delta, int n, double complex *coeff)
{ return target(w, t_max) - (1.0+delta)*approx(w, t_max, n, coeff); }
double dresidual(double w, double t_max, double delta, int n, double complex *coeff)
{ return dtarget(w, t_max) - (1.0+delta)*dapprox(w, t_max, n, coeff); }
double d2residual(double w, double t_max, double delta, int n, double complex *coeff)
{ return d2target(w, t_max) - (1.0+delta)*d2approx(w, t_max, n, coeff); }

// Newton refinement of a bracketed extremum
#define NEWTON_TOL 1e-11
double refine(double left, double right, double t_max, double delta, int n, double complex *coeff)
{
    double dr_left = dresidual(left, t_max, delta, n, coeff);
    double dr_right = dresidual(left, t_max, delta, n, coeff);
    double w = (left+right)/2.0, dw;
    do
    {
        dw = -dresidual(w, t_max, delta, n, coeff)/d2residual(w, t_max, delta, n, coeff);
        if(w + dw < left || w + dw > right)
        {
            double dr_mid = dresidual(w, t_max, delta, n, coeff);
            if(dr_left*dr_mid >= 0.0)
            { left = w; dr_left = dr_mid; }
            else
            { right = w; dr_right = dr_mid; }
            w = (left+right)/2.0;
            dw = (right-left)/2.0; 
        }
        else
        { w += dw; }
    } while (dw > NEWTON_TOL);
    return w;
}

// scan local extrema bracketed by a coarse grid for the global extrema
#define COARSE_GRID_MULTIPLE 10
double scan(double W_max, double t_max, double delta, int n, double complex *coeff)
{
    // start w/ origin as the global extrema
    double res_max = fabs(residual(0.0, t_max, delta, n, coeff));

    // check if W_max is a larger extrema
    double res_new = fabs(residual(W_max, t_max, delta, n, coeff));
    if(res_max < res_new)
    { res_max = res_new; }

    // find extrema bracketed on coarse grid
    int num_coarse = COARSE_GRID_MULTIPLE<<n;
    for(int i=1 ; i<num_coarse-1 ; i++)
    {
        int is_new_ext = 0;
        double w_new;
        double w_left = W_max*(double)i/(double)num_coarse;
        double w_right = W_max*(double)(i+1)/(double)num_coarse;
        double dres_left = dresidual(w_left, t_max, delta, n, coeff);
        double dres_right = dresidual(w_right, t_max, delta, n, coeff);

        // check for extrema that is bracketed or at one of the end points
        if(dres_left*dres_right < 0.0)
        {
            is_new_ext = 1;
            w_new = refine(w_left, w_right, t_max, delta, n, coeff);
        }
        else if(dres_left == 0.0)
        {
            is_new_ext = 1;
            w_new = w_left;
        }
        else if(dres_right == 0.0)
        {
            is_new_ext = 1;
            w_new = w_right;
        }

        // replace global extrema if necessary
        if(is_new_ext)
        {
            res_new = fabs(residual(w_new, t_max, delta, n, coeff));
            if(res_max < res_new)
            { res_max = res_new; }
        }
    }
    return res_max;
}

// optimize delta over only two important values: delta = 0 & delta set to balance residual errors on the left & right
double optimize_delta(double W_max, double t_max, double *delta, int n, double complex *coeff)
{
    double delta1 = 0.0;
    double res1 = scan(W_max, t_max, delta1, n, coeff);

    double r1 = target(0.0, t_max) - approx(0.0, t_max, n, coeff);
    double r2 = target(W_max, t_max) - approx(W_max, t_max, n, coeff);
    double f1 = approx(0.0, t_max, n, coeff);
    double f2 = approx(W_max, t_max, n, coeff);
    double s1 = 1.0, s2 = 1.0;
    if(f1 < 0.0) { s1 = -1.0; }
    if(f2 < 0.0) { s2 = -1.0; }
    double delta2 = (s1*r1 + s2*r2)/(s1*f1 + s2*f2);
    double res2 = scan(W_max, t_max, delta2, n, coeff);

    // return the smaller of the two residual error maxima
    if(res1 > res2)
    { res1 = res2; delta1 = delta2; }
    *delta = delta1;
    return res1;
}

// calculate the value of epsilon by minimizing over delta
double error_bound(int r, int n, double t_max, double E_max)
{
    // assign the approximation domain
    double W_max = E_max + ((1<<n)-1)*0.5*M_PI/t_max;
    if(W_max > (1<<n)*M_PI/t_max)
    { W_max = (1<<n)*M_PI/t_max; }

    // calculate the coefficients of the Fourier approximant
    double complex *coeff = (double complex*)malloc(sizeof(double complex)*(1<<n));
    fourier_fit(r, n, t_max, coeff);

    // optimize delta & calculate the associated global residual extrema
    double delta;
    double res_max = optimize_delta(W_max, t_max, &delta, n, coeff);

    free(coeff);
    return res_max;
}

#define BOUND_MIN 1e-14
int main(int argc, char **argv)
{
    // input the number of ancilla qubits (n) & spacing of bounds tests
    if(argc < 3) { printf("SYNTAX: <executable> <number_of_ancilla> <relative_bound_spacing>\n"); exit(1); }
    int n;
    double ratio;
    sscanf(argv[1], "%d", &n);
    sscanf(argv[2], "%lf", &ratio);

    // assign lower & upper bounds & number of steps to t_max
    double t_maxmax = -log(BOUND_MIN)/M_PI;
    double delta_t_max = -log(ratio)/M_PI;
    int t_num = ceil(t_maxmax/delta_t_max);

    // loop over r, t_max, & E_max
    int E_num = 50;
    double heuristic2_old;
    #pragma omp parallel for
    for(int t=1 ; t<=t_num ; t++)
    {
        double t_max = t_maxmax*(double)t/(double)t_num;
        double w_max = (1<<n)*0.5*M_PI/t_max;
        double heuristic1 = exp(-M_PI*t_max);

        for(int r=1 ; r<=n ; r++)
        {
            double heuristic2 = exp(-0.25*(1<<(2*r-2))*M_PI/t_max);
            if(heuristic2 < BOUND_MIN)
            { break; }

            double E_max = w_max;
            while(E_max >= 0.0)
            {
                double heuristic3 = exp(-0.25*pow((1<<(n-1)) - E_max*t_max/M_PI,2)*M_PI/t_max);
                if(heuristic3 < BOUND_MIN)
                { break; }

                double err = error_bound(r, n, t_max, E_max);
                double hmax = heuristic1;
                if(heuristic2 > hmax) { hmax = heuristic2; }
                if(heuristic3 > hmax) { hmax = heuristic3; }

                printf("%d %e %d %e   %e %e %e   %e %e\n", n, t_max, r, E_max, heuristic1, heuristic2, heuristic3, hmax, err);
                if(hmax < err)
                { printf("ERROR: error bound has been violated (%e < %e) for %d %e %d %e\n", hmax, err, n, t_max, r, E_max); exit(1); }

                // assign next value of E_max
                E_max = (M_PI/t_max)*((1<<(n-1)) - sqrt(-4.0*t_max*log(ratio)/M_PI + pow((1<<(n-1)) - E_max*t_max/M_PI,2)));
            }
        }
    }
    return 0;
}