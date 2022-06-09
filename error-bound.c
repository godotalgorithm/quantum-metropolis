// numerical verification of empirical error bound in Eq. (B6) of "Measurement-Based Quantum Metropolis Algorithm"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

// ASSUMPTIONS:
// - residual extrema are assumed to be bracketed on a coarse grid (missed maxima can underestimate maximum error)
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
double residual(double w, double t_max, int n, double complex *coeff)
{ return target(w, t_max) - approx(w, t_max, n, coeff); }
double dresidual(double w, double t_max, int n, double complex *coeff)
{ return dtarget(w, t_max) - dapprox(w, t_max, n, coeff); }
double d2residual(double w, double t_max, int n, double complex *coeff)
{ return d2target(w, t_max) - d2approx(w, t_max, n, coeff); }

// Newton refinement of a bracketed extremum
#define NEWTON_TOL 1e-11
double refine(double left, double right, double t_max, int n, double complex *coeff)
{
    double dr_left = dresidual(left, t_max, n, coeff);
    double dr_right = dresidual(left, t_max, n, coeff);
    double w = (left+right)/2.0, dw;
    do
    {
        dw = -dresidual(w, t_max, n, coeff)/d2residual(w, t_max, n, coeff);
        if(w + dw < left || w + dw > right)
        {
            double dr_mid = dresidual(w, t_max, n, coeff);
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
double scan(double W_max, double t_max, int n, double complex *coeff)
{
    // start w/ origin as the global extrema
    double res_max = fabs(residual(0.0, t_max, n, coeff));

    // check if W_max is a larger extrema
    double res_new = fabs(residual(W_max, t_max, n, coeff));
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
        double dres_left = dresidual(w_left, t_max, n, coeff);
        double dres_right = dresidual(w_right, t_max, n, coeff);

        // check for extrema that is bracketed or at one of the end points
        if(dres_left*dres_right < 0.0)
        {
            is_new_ext = 1;
            w_new = refine(w_left, w_right, t_max, n, coeff);
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
            res_new = fabs(residual(w_new, t_max, n, coeff));
            if(res_max < res_new)
            { res_max = res_new; }
        }
    }
    return res_max;
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

    // calculate the global residual extrema
    double res_max = scan(W_max, t_max, n, coeff);

    free(coeff);
    return res_max;
}

#define BOUND_MIN 1e-14
int main(int argc, char **argv)
{
    // input the number of ancilla qubits (n) & spacing of bounds tests
    if(argc < 3) { printf("SYNTAX: <executable> <number_of_ancilla> <relative_bound_spacing>\n"); exit(1); }
    int r;
    double ratio;
    sscanf(argv[1], "%d", &r);
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
        double w_max = (1<<r)*0.5*M_PI/t_max;
        double heuristic1 = 2.0*exp(-M_PI*t_max);

        for(int s=1 ; s<=r ; s++)
        {
            double heuristic2 = 2.0*exp(-0.25*(1<<(2*s-2))*M_PI/t_max);
            if(heuristic2 < BOUND_MIN)
            { break; }

            double E_max = w_max;
            while(E_max >= 0.0)
            {
                double heuristic3 = 2.0*exp(-0.25*pow((1<<(r-1)) - E_max*t_max/M_PI,2)*M_PI/t_max);
                if(heuristic3 < BOUND_MIN)
                { break; }

                double err = error_bound(s, r, t_max, E_max);
                double hmax = heuristic1;
                if(heuristic2 > hmax) { hmax = heuristic2; }
                if(heuristic3 > hmax) { hmax = heuristic3; }

                printf("%d %e %d %e   %e %e %e   %e %e\n", r, t_max, s, E_max, heuristic1, heuristic2, heuristic3, hmax, err);
                if(hmax < err)
                { printf("ERROR: error bound has been violated (%e < %e) for %d %e %d %e\n", hmax, err, r, t_max, s, E_max); exit(1); }

                // assign next value of E_max
                E_max = (M_PI/t_max)*((1<<(r-1)) - sqrt(-4.0*t_max*log(ratio)/M_PI + pow((1<<(r-1)) - E_max*t_max/M_PI,2)));
            }
        }
    }
    return 0;
}
