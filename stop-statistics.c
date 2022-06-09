// quantum Metropolis stopping-time statistics for a single-energy Hamiltonian
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

// mode 0: finite width case
// mode 1: conditional sampling of zero-width limit
// mode 2: conditional sampling with fast-forwarding

// parameters specifying the integration over E_old (2*ENERGY_MAX is used as an "infinite" energy reference)
#define ENERGY_MAX 12.0
#define ENERGY_NUM 10001

// energy separation for Taylor series expansion in rejection PDF & CDF
#define ENERGY_TOL 1e-3

// energy tolerance & maximum iterations for the bisection + Newton root finder
#define NEWTON_TOL 1e-12
#define BISECT_TOL 1e-14
#define ROOT_MAX 50

// NOTE: As this is a form of direct rare event sampling, we are dealing with some extremely small probabilities
//       that require numerical care to handle. The main numerical issue is the evaluation of the rejection CDF
//       when E_old and E_min are very close together, which suffers from a large amplification of rounding errors.
//       We mitigate this problem by evaluating the function as a truncated Taylor series and choose a crossover
//       point where approximation and numerical errors are approximately balanced. However, the Taylor series is
//       not of a high enough order to preserve all digits of accuracy, and so we need to use bisection as a fail-safe
//       root finder when performing root-finding on this problematic CDF on the unstable side of the crossover point.
//       These biasing errors are very small relative to our target sampling errors (our conditional sampling approach
//       achieves high relative accuracy), thus they merely complicate the numerics and solver strategies.

// NOTE: The data generation process revealed some useful asymptotics about the distribution. The initiation
//       probability was apparent from constructing the initial PDF & CDF, which produced a probability that can
//       be calculated analytically as a simple Gaussian integral. The fast-forwarding clarified that the tail
//       process reduces to finding energies below the initial energy, once the possibility of accepting an
//       energy above has been exhausted. While the probability distribution of this tail process can be written
//       down as an integral, it cannot be evaluated analytically. Instead, I used Laplace's method (i.e. the method
//       of steepest descent, the saddle-point approximation) to construct a large-n asymptotic form of
//       n^{-2}/sqrt(a + b*log(n)), which fit the tail of the data to within the statistical error bars,
//       allowing for the coefficient of the limiting form, n^{-2}/sqrt(log(n)), to be extracted accurately.

// pseudorandom 64-bit unsigned integer [xorshift1024s() from http://en.wikipedia.org/wiki/Xorshift]
uint64_t random64(const uint32_t seed) // 0 for normal use, nonzero seed value to reseed
{
  static uint64_t s[16];
  static uint8_t p;

  // seed & "warm up" the PRNG
  if(seed != 0)
  {
    p = 0;
    uint32_t i;
    for(i=0 ; i<16 ; i++) s[i] = seed + i;
    for(i=0 ; i<16384 ; i++) random64(0);
  }

  uint64_t s0 = s[p];
  p = (p + 1) & 15;
  uint64_t s1 = s[p];

  s1 ^= s1 << 31; // a
  s1 ^= s1 >> 11; // b
  s0 ^= s0 >> 30; // c
  s[p] = s0 ^ s1;

  return s[p] * 1181783497276652981ULL;
}

// pseudorandom uniform distribution over (0,1]
double random_uniform()
{
  // reduce from 64 random bits to 53 random bits that span the representable unpadded integers using a double
  return (double)((random64(0) >> 11) + 1)/9007199254740992.0;
}

// pseudorandom normal distribution w/ unit variance (half of a Box-Muller transform, not efficient)
double random_normal()
{
    return sqrt(fabs(2.0*log(random_uniform())))*sin(2.0*M_PI*random_uniform());
}

// normal distribution PDF & CDF
double pdf(double x)
{ return exp(-0.5*x*x)/sqrt(2.0*M_PI);}
double cdf(double x) // high relative accuracy for x < 0
{ return 0.5*erfc(-sqrt(0.5)*x); }
double cdf2(double x) // constant shift, high relative accuracy for x > 0
{ return -0.5*erfc(sqrt(0.5)*x); }

// PDF of an initial E_new
double initial_pdf(double E_new, double E_old)
{
    if(E_new <= E_old)
    { return 0.0; }
    return pdf(E_new)*(E_new - E_old);
}

// CDF of an initial E_new
double initial_cdf(double E_new, double E_old)
{
    if(E_new <= E_old)
    { return 0.0; }
    if(E_old > 0.0)
    { return pdf(E_old) - pdf(E_new) + E_old*(cdf2(E_old) - cdf2(E_new)); }
    return pdf(E_old) - pdf(E_new) + E_old*(cdf(E_old) - cdf(E_new));
}

// PDF of a rejected E_new (y < z)
double reject_pdf(double x, double y, double z)
{
    if(x <= y)
    { return 0.0; }
    if(x <= z) // y < x <= z
    {
        // Taylor series expansion in x from x=y, to match CDF
        if(fabs(z-y) < ENERGY_TOL)
        { return (1.0 + (x-y)*(-y + 0.5*(y*y - 1.0)*(x-y)))*pdf(y)*(x-y)/(z-y); }
        return pdf(x)*(x - y)/(z - y);
    }
    return pdf(x);
}

// CDF of a rejected E_new (y < z)
double reject_cdf(double x, double y, double z)
{
    if(x <= y)
    { return 0.0; }
    if(x <= z) // y < x <= z
    {
        // Taylor series expansion in x from x=y
        if(fabs(z-y) < ENERGY_TOL)
        { return (0.5 + (x-y)*(-y/3.0 + 0.125*(y*y - 1.0)*(x-y)))*pdf(y)*(x-y)*(x-y)/(z-y); }
        if(y > 0.0)
        { return (pdf(y) - pdf(x) + y*(cdf2(y) - cdf2(x)))/(z - y); }
        return (pdf(y) - pdf(x) + y*(cdf(y) - cdf(x)))/(z - y);
    }
    // Taylor series expansion in x from x=y
    if(fabs(z-y) < ENERGY_TOL)
    {
        if(y > 0.0)
        { return cdf2(x) - cdf2(z) + (0.5 + (z-y)*(-y/3.0 + 0.125*(y*y - 1.0)*(z-y)))*pdf(y)*(z-y); }
        return cdf(x) - cdf(z) + (0.5 + (z-y)*(-y/3.0 + 0.125*(y*y - 1.0)*(z-y)))*pdf(y)*(z-y);
    }
    if(y > 0.0)
    { return cdf2(x) - cdf2(z) + (pdf(y) - pdf(z) + y*(cdf2(y) - cdf2(z)))/(z - y); }
    return cdf(x) - cdf(z) + (pdf(y) - pdf(z) + y*(cdf(y) - cdf(z)))/(z - y);
}

// sample of an initial E_new
double initial_sample(double E_old)
{
    // calculate source of randomness
    double u = random_uniform();

    // calculate total rejection probability
    double p_total = initial_cdf(2.0*ENERGY_MAX, E_old);

    // assign a search interval
    double E_left = E_old;
    double E_right = 2.0*ENERGY_MAX;

    // perform bisection until Newton iterations are within the bound
    double E = 0.5*(E_left + E_right);
    double P = initial_cdf(E, E_old)/p_total;
    double dPdE = initial_pdf(E, E_old)/p_total;
    int niter = 0;
    while( (E+(u-P)/dPdE) <= E_left || (E+(u-P)/dPdE) >= E_right)
    {
        if(u - P > 0.0)
        { E_left = E; }
        else
        { E_right = E; }
        E = 0.5*(E_left + E_right);
        P = initial_cdf(E, E_old)/p_total;
        dPdE = initial_pdf(E, E_old)/p_total;

        // check if maximum iteration number has been reached
        niter++;
        if(niter > ROOT_MAX)
        { break; }
//        { printf("WARNING: Initial bisection stalled (%15.15e %15.15e %15.15e %e %e)\n", E_left, E, E_right, u-P, dPdE); break; }
    }

    // perform Newton iterations until convergence or breach of bounds
    while( (E+(u-P)/dPdE) > E_left && (E+(u-P)/dPdE) < E_right && fabs(u-P) > NEWTON_TOL*fabs(dPdE))
    {
        E += (u-P)/dPdE;
        P = initial_cdf(E, E_old)/p_total;
        dPdE = initial_pdf(E, E_old)/p_total;

        // check if maximum iteration number has been reached
        niter++;
        if(niter > ROOT_MAX)
        { break; }
//        { printf("WARNING: Newton method stalled (%15.15e %15.15e %15.15e %e %e)\n", E_left, E, E_right, fabs(u-P), NEWTON_TOL*fabs(dPdE)); break; }
    }

    // perform fail-safe bisection on failure
    if( (E+(u-P)/dPdE) <= E_left || (E+(u-P)/dPdE) >= E_right || (E+(u-P)/dPdE) > E_left && (E+(u-P)/dPdE) < E_right && fabs(u-P) > NEWTON_TOL*fabs(dPdE))
    {
        while( E_right - E_left > BISECT_TOL)
        {
            if(u - P > 0.0)
            { E_left = E; }
            else
            { E_right = E; }
            E = 0.5*(E_left + E_right);
            P = initial_cdf(E, E_old)/p_total;
            dPdE = initial_pdf(E, E_old)/p_total;
            niter++;
        }
    }

    return E;
}

// sample of a rejected E_new up to E_max
double reject_sample(double E_max, double E_old, double E_min)
{
    // calculate source of randomness
    double u = random_uniform();

    // calculate total rejection probability
    double p_total = reject_cdf(E_max, E_old, E_min);

    // assign a search interval
    double E_left = E_old;
    double E_right = E_max;

    // perform bisection until Newton iterations are within the bound
    double E = 0.5*(E_left + E_right);
    double P = reject_cdf(E, E_old, E_min)/p_total;
    double dPdE = reject_pdf(E, E_old, E_min)/p_total;
    int niter = 0;
    while( (E+(u-P)/dPdE) <= E_left || (E+(u-P)/dPdE) >= E_right)
    {
        if(u - P > 0.0)
        { E_left = E; }
        else
        { E_right = E; }
        E = 0.5*(E_left + E_right);
        P = reject_cdf(E, E_old, E_min)/p_total;
        dPdE = reject_pdf(E, E_old, E_min)/p_total;

        // check if maximum iteration number has been reached
        niter++;
        if(niter > ROOT_MAX)
        { break; }
//        { printf("WARNING: Initial bisection stalled (%15.15e %15.15e %15.15e %e %e)\n", E_left, E, E_right, u-P, dPdE); break; }
    }

    // perform Newton iterations until convergence or breach of bounds
    while( (E+(u-P)/dPdE) > E_left && (E+(u-P)/dPdE) < E_right && fabs(u-P) > NEWTON_TOL*fabs(dPdE))
    {
        E += (u-P)/dPdE;
        P = reject_cdf(E, E_old, E_min)/p_total;
        dPdE = reject_pdf(E, E_old, E_min)/p_total;

        // check if maximum iteration number has been reached
        niter++;
        if(niter > ROOT_MAX)
        { break; }
//        { printf("WARNING: Newton method stalled (%15.15e %15.15e %15.15e %e %e)\n", E_left, E, E_right, fabs(u-P), NEWTON_TOL*fabs(dPdE)); break; }
    }

    // perform fail-safe bisection on failure
    if( (E+(u-P)/dPdE) <= E_left || (E+(u-P)/dPdE) >= E_right || (E+(u-P)/dPdE) > E_left && (E+(u-P)/dPdE) < E_right && fabs(u-P) > NEWTON_TOL*fabs(dPdE))
    {
        while( E_right - E_left > BISECT_TOL)
        {
            if(u - P > 0.0)
            { E_left = E; }
            else
            { E_right = E; }
            E = 0.5*(E_left + E_right);
            P = reject_cdf(E, E_old, E_min)/p_total;
            dPdE = reject_pdf(E, E_old, E_min)/p_total;
            niter++;
        }
    }

    return E;
}

// sample of how many attempts before an event occurs
long long int skip_sample(double p_event)
{
    if(p_event == 0.0)
    { return LLONG_MAX; }

    // calculate source of randomness
    double u = random_uniform();

    double num = floor(log1p(-u)/log1p(-p_event));
    if(num > (double)(LLONG_MAX-1))
    { return LLONG_MAX; }
    return (long long int)num;
}

// quantum rejection-free Metropolis algorithm for a single-energy Hamiltonian
int metropolis_quantum(double width, long long int max_iter)
{
    double u;

    long long int n = 0;
    double q = 0.0;
    double q_max = 0.0;
    double E_old = random_normal();
    do
    {
        n++;
        double E_new = random_normal();
        if(q > q_max) { q_max = q; }
        q = exp(sqrt(width)*(E_old - E_new) - width);
        u = random_uniform();
    } while(u > (q - q_max)/(1.0 - q_max) && n != max_iter);
    return n;
}

// integrate over old energy, calculate conditional probabilities for rejected new energies
void metropolis_scan(long long int max_iter, double *p_stop)
{
    for(long long int i=0 ; i<max_iter ; i++)
    { p_stop[i] = 0.0; }

    // precompute the discretized Gaussian distribution
    double E[ENERGY_NUM], p[ENERGY_NUM], wt[ENERGY_NUM], p_sum = 0.0;
    for(int i=0 ; i<ENERGY_NUM ; i++)
    {
        E[i] = -ENERGY_MAX + 2.0*ENERGY_MAX*(double)i/(double)(ENERGY_NUM-1);
        p[i] = exp(-0.5*E[i]*E[i])/sqrt(2.0*M_PI);
        p_sum += p[i];
    }
    for(int i=0 ; i<ENERGY_NUM ; i++)
    { p[i] /= p_sum; }

    // loop over possible values of E_old
    for(int i=0 ; i<ENERGY_NUM ; i++)
    {
        double E_min = 2.0*ENERGY_MAX;
        double E_new = 2.0*ENERGY_MAX;
        double E_old = E[i];
        double p_now = p[i];

        for(long long int n=0 ; n<max_iter ; n++)
        {
            if(E_new < E_min) { E_min = E_new; }

            // calculate rejection probability
            double p_reject;
            if(n == 0) { p_reject = initial_cdf(2.0*ENERGY_MAX, E_old); }
            else { p_reject = reject_cdf(2.0*ENERGY_MAX, E_old, E_min); }
            p_stop[n] += p_now*(1.0 - p_reject);
            p_now *= p_reject;

            // sample an E_new value post-selected on rejection
            if(n == 0) { E_new = initial_sample(E_old); }
            else { E_new = reject_sample(2.0*ENERGY_MAX, E_old, E_min); }
        }
    }
    return;
}

// preliminary fast-forwarding implementation based on metropolis_scan
void metropolis_fast0(int num, long long int *iter, double *p_stop)
{
    for(int i=0 ; i<num ; i++)
    { p_stop[i] = 0.0; }

    // precompute the discretized Gaussian distribution
    double E[ENERGY_NUM], p[ENERGY_NUM], wt[ENERGY_NUM], p_sum = 0.0;
    for(int i=0 ; i<ENERGY_NUM ; i++)
    {
        E[i] = -ENERGY_MAX + 2.0*ENERGY_MAX*(double)i/(double)(ENERGY_NUM-1);
        p[i] = exp(-0.5*E[i]*E[i])/sqrt(2.0*M_PI);
        p_sum += p[i];
    }
    for(int i=0 ; i<ENERGY_NUM ; i++)
    { p[i] /= p_sum; }

    // loop over possible values of E_old
    for(int i=0 ; i<ENERGY_NUM ; i++)
    {
        double E_min = 2.0*ENERGY_MAX;
        double E_old = E[i];
        double p_now = p[i];

        int j = 0, is_event = 1;
        double p_reject = initial_cdf(2.0*ENERGY_MAX, E_old);
        double p_event;
        if(iter[0] == 1)
        { p_stop[0] += p_now*(1.0 - p_reject); j++; }
        p_now *= p_reject;
        E_min = initial_sample(E_old);
        for(long long int n=1 ; n<iter[num-1] ; n++)
        {
            // calculate rejection probability
            if(is_event)
            {
                p_reject = reject_cdf(2.0*ENERGY_MAX, E_old, E_min);
                p_event = reject_cdf(E_min, E_old, E_min)/p_reject;
                is_event = 0;
            }
            if(iter[j] == n+1)
            { p_stop[j] += p_now*(1.0 - p_reject); j++; }
            p_now *= p_reject;

            // sample an E_new value post-selected on rejection
            double u = random_uniform();
            if(u < p_event)
            {
                E_min = reject_sample(E_min, E_old, E_min);
                is_event = 1;
            }
        }
    }
    return;
}

// calculate conditional probabilities for rejected new energies & fast-forward over simple rejection events
void metropolis_fast(int num, long long int *iter, double *p_stop)
{
    for(int i=0 ; i<num ; i++)
    { p_stop[i] = 0.0; }

    // precompute the discretized Gaussian distribution
    double E[ENERGY_NUM], p[ENERGY_NUM], wt[ENERGY_NUM], p_sum = 0.0;
    for(int i=0 ; i<ENERGY_NUM ; i++)
    {
        E[i] = -ENERGY_MAX + 2.0*ENERGY_MAX*(double)i/(double)(ENERGY_NUM-1);
        p[i] = exp(-0.5*E[i]*E[i])/sqrt(2.0*M_PI);
        p_sum += p[i];
    }
    for(int i=0 ; i<ENERGY_NUM ; i++)
    { p[i] /= p_sum; }

    // loop over possible values of E_old
    for(int i=0 ; i<ENERGY_NUM ; i++)
    {
        double E_min = 2.0*ENERGY_MAX;
        double E_old = E[i];
        double p_now = p[i];

        int j = 0;
        long long int n = 1;
        double p_reject = initial_cdf(2.0*ENERGY_MAX, E_old);
        if(iter[0] == n)
        { p_stop[0] += p_now*(1.0 - p_reject); j++; }
        p_now *= p_reject;
        E_min = initial_sample(E_old);
        do
        {
            // calculate rejection probability
            p_reject = reject_cdf(2.0*ENERGY_MAX, E_old, E_min);
            double p_event = reject_cdf(E_min, E_old, E_min)/p_reject;
            long long int n_event = skip_sample(p_event);
            if(n_event >= iter[num-1]-n)
            { n_event = iter[num-1]-n-1; }

            // fast-forward until the next event
            while(n_event != 0)
            {
                // process to the next recorded iteration
                if(n_event >= iter[j]-n)
                {
                    p_now *= pow(p_reject, iter[j]-n-1);
                    n_event -= iter[j]-n;
                    n = iter[j];
                    p_stop[j] += p_now*(1.0 - p_reject);
                    j++;
                    p_now *= p_reject;
                }
                // process all remaining iterations
                else
                {
                    p_now *= pow(p_reject, n_event);
                    n += n_event;
                    n_event = 0;
                }
            }

            // update probabilities for the eventful iteration (or the final iteration)
            n++;
            if(iter[j] == n)
            { p_stop[j] += p_now*(1.0 - p_reject); j++; }
            p_now *= p_reject;

            // sample an E_new value post-selected on rejection
            double u = random_uniform();
            E_min = reject_sample(E_min, E_old, E_min);
        } while(n < iter[num-1]);
    }
    return;
}

int main(int argc, char **argv)
{
    if(argc < 5) { printf("SYNTAX: <executable> <seed> <mode> <num_sample> <max_iter>\n"); exit(1); }
    int seed, mode;
    long long int max_iter, num_sample;
    double width, resolution;
    sscanf(argv[1], "%d", &seed);
    sscanf(argv[2], "%d", &mode);
    sscanf(argv[3], "%lld", &num_sample);
    sscanf(argv[4], "%lld", &max_iter);
    if(mode == 0)
    {
        if(argc < 6) { printf("SYNTAX: <executable> <seed> <mode == 0> <num_sample> <max_iter> <width>\n"); exit(1); }
        sscanf(argv[5], "%lf", &width);
    }
    if(mode == 2)
    {
        if(argc < 6) { printf("SYNTAX: <executable> <seed> <mode == 2> <num_sample> <max_iter> <resolution>\n"); exit(1); }
        sscanf(argv[5], "%lf", &resolution);
    }

    random64(seed);

    switch(mode)
    {
        case 0:
        {
            long long int *count = (long long int*)malloc(sizeof(long long int)*max_iter);
            for(long long int i=0 ; i<max_iter ; i++)
            { count[i] = 0; }

            for(long long int i=0 ; i<num_sample ; i++)
            { count[metropolis_quantum(width, max_iter) - 1]++; }

            printf("%d %d %lld %lld %e\n", seed, mode, num_sample, max_iter, width);

            for(long long int i=0 ; i<max_iter ; i++)
            {
                double p = (double)count[i]/(double)num_sample;
                printf("%lld %e %e\n", i+1, p, sqrt(p*(1.0-p)/(double)num_sample));
            }

            free(count);
            break;
        }

        case 1:
        {
            double *p_stop = (double*)malloc(sizeof(double)*max_iter);
            double *p2_stop = (double*)malloc(sizeof(double)*max_iter);
            double *p_now = (double*)malloc(sizeof(double)*max_iter);
            for(long long int i=0 ; i<max_iter ; i++)
            { p_stop[i] = p2_stop[i] = 0.0; }

            for(long long int i=0 ; i<num_sample ; i++)
            {
                metropolis_scan(max_iter, p_now);
                for(long long int j=0 ; j<max_iter ; j++)
                {
                    p_stop[j] += p_now[j];
                    p2_stop[j] += p_now[j]*p_now[j];
                }
            }

            printf("%d %d %lld %lld\n", seed, mode, num_sample, max_iter);

            for(long long int i=0 ; i<max_iter ; i++)
            {
                p_stop[i] /= (double)num_sample;
                p2_stop[i] /= (double)num_sample;
                printf("%lld %e %e\n", i+1, p_stop[i], sqrt(fabs(p2_stop[i] - pow(p_stop[i],2))/(double)num_sample));
            }

            free(p_stop);
            free(p2_stop);
            free(p_now);
            break;
        }

        case 2:
        {
            int num = 1;
            long long int iter0 = 1;
            while(iter0 < max_iter)
            { iter0 += ceil(iter0*resolution); num++; }
            long long int *iter = (long long int*)malloc(sizeof(long long int)*num);
            double *p_stop = (double*)malloc(sizeof(double)*num);
            double *p2_stop = (double*)malloc(sizeof(double)*num);
            double *p_now = (double*)malloc(sizeof(double)*num);
            iter[0] = 1;
            num = 1;
            while(iter[num-1] < max_iter)
            { iter[num] = iter[num-1] + ceil(iter[num-1]*resolution); num++; }
            iter[num-1] = max_iter;
            for(int i=0 ; i<num ; i++)
            { p_stop[i] = p2_stop[i] = 0.0; }

            for(long long int i=0 ; i<num_sample ; i++)
            {
                metropolis_fast(num, iter, p_now);
                for(int j=0 ; j<num ; j++)
                {
                    p_stop[j] += p_now[j];
                    p2_stop[j] += p_now[j]*p_now[j];
                }
            }

            printf("%d %d %lld %lld %e\n", seed, mode, num_sample, max_iter, resolution);

            for(int i=0 ; i<num ; i++)
            {
                p_stop[i] /= (double)num_sample;
                p2_stop[i] /= (double)num_sample;
                printf("%lld %e %e\n", iter[i], p_stop[i], sqrt(fabs(p2_stop[i] - pow(p_stop[i],2))/(double)num_sample));
            }

            free(iter);
            free(p_stop);
            free(p2_stop);
            free(p_now);
            break;
        }

        default:
        {
            printf("ERROR: unknown mode (%d)\n", mode);
            exit(1);
        }
    }
    return 0;
}
