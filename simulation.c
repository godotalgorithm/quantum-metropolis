// simulation of the quantum Metropolis algorithm on the transverse-field Ising model
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

// NOTE: Beta is absorbed into the Hamiltonian and doesn't explicitly appear anywhere,
//       which necessitates a rescaling of the output energies.

// cutoff of cumulative probability for the scan over direct sampling cutoff energies
#define CP_CUTOFF 1e-16

// BLAS matrix-vector multiplication
void dgemv_(char*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

// LAPACK symmetric eigensolver
void dsyev_(char*, char*, int*, double*, int*, double*, double*, int*, int*);

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

// pseudorandom sampling from a list of events & their cumulative probabilities
int random_list(int n, double *cp)
{
    double p0 = random_uniform();
    if(p0 < cp[0]) { return 0; }
    int min = 0, max = n-1;
    while(max - min > 1)
    {
        int new = (min+max)/2;
        if(p0 < cp[new]) { max = new; }
        else { min = new; }
    }
    return max;
}

// computational basis measurement
int measure_basis(int num_site, double *state)
{
    int n = 1<<num_site;
    double p = 0.0, p0 = random_uniform();
    for(int i=0 ; i<n ; i++)
    {
        p += state[i]*state[i];
        if(p >= p0)
        { return i; }
    }
    return n-1;
}

// measure Pauli Z
double measure_site(int site, int num_site, double *state)
{
    int n = 1<<num_site, filter = 1<<site;
    double p = 0.0;
    for(int i=0 ; i<n ; i++)
    {
        if(filter&i)
        { p += state[i]*state[i]; }
    }
    double p0 = random_uniform();

    int outcome;
    double wt;
    if(p0 <= p)
    { outcome = 1; wt = 1.0/sqrt(p); }
    else
    { outcome = 0; wt = 1.0/sqrt(1.0-p); }

    // collapse the state according to the outcome
    for(int i=0 ; i<n ; i++)
    {
        if(filter&i)
        {
            if(outcome)
            { state[i] *= wt; }
            else
            { state[i] = 0.0; }
        }
        else
        {
            if(outcome)
            { state[i] = 0.0; }
            else
            { state[i] *= wt; }
        }
    }

    return 1.0 - 2.0*outcome;
}

// apply Pauli X
void flip_site(int site, int num_site, double *state)
{
    int n = 1<<num_site, filter = 1<<site;
    for(int i=0 ; i<n ; i++)
    {
        if(filter&i)
        {
            double swap = state[i];
            state[i] = state[i^filter];
            state[i^filter] = swap;
        }
    }
}

// Hamiltonian construction
void init_hamiltonian(int num_site, double field, double coupling, double *hamiltonian)
{
    int n = 1<<num_site, n2 = n*n;

    // clear matrix
    for(int i=0 ; i<n2 ; i++)
    { hamiltonian[i] = 0.0; }

    for(int i=0 ; i<num_site ; i++)
    {
        // field term
        int flip = 1<<i;
        for(int j=0 ; j<n ; j++)
        { hamiltonian[j+(j^flip)*n] += field; }

        // coupling term
        int filter1 = 1<<i, filter2 = 1<<((i+1)%num_site);
        for(int j=0 ; j<n ; j++)
        {
            int filter = 0;
            if(filter1&j) { filter ^= 1; }
            if(filter2&j) { filter ^= 1; }
            if(filter)
            { hamiltonian[j+j*n] += -coupling; }
            else
            { hamiltonian[j+j*n] += coupling; }
        }
    }
}

// calculate Hamiltonian eigenbasis
void init_eigenbasis(int num_site, double *hamiltonian, double *energy)
{
    char jobz = 'V', uplo = 'L';
    int n = 1<<num_site, lwork = -1, info;
    double work0, *work;
    dsyev_(&jobz, &uplo, &n, hamiltonian, &n, energy, &work0, &lwork, &info);
    if(info) { printf("ERROR: dsyev failure during allocation (%d)\n", info); }
    lwork = (int)work0;
    work = (double*)malloc(sizeof(double)*lwork);
    dsyev_(&jobz, &uplo, &n, hamiltonian, &n, energy, work, &lwork, &info);
    if(info) { printf("ERROR: dsyev failure during solve (%d)\n", info); }
    free(work);
}

// transform to & from Hamiltonian eigenbasis
// (trans = 'N' : from site basis to eigenbasis, trans = 'T' : from eigenbasis to site basis)
void transform(char trans, int num_site, double *eigenbasis, double *state_site, double *state_eigen)
{
    int n = 1<<num_site, inc = 1;
    double zero = 0.0, one = 1.0;
    if(trans == 'T')
    { dgemv_(&trans, &n, &n, &one, eigenbasis, &n, state_site, &inc, &zero, state_eigen, &inc); }
    else if(trans == 'N')
    { dgemv_(&trans, &n, &n, &one, eigenbasis, &n, state_eigen, &inc, &zero, state_site, &inc); }
    else
    { printf("ERROR: trans in dgemv must be either 'N' or 'T' (%c)\n", trans); }
}

struct gqpe
{
    // Hamiltonian parameters
    int num_site;
    double field;
    double coupling;

    // GQPE parameters
    double epsilon;

    // derived quantities
    int num_ancilla;
    int num_ancilla2;
    double tmax;
    double gamma;

    // Hamiltonian data
    double *energy;
    double *eigenbasis;

    // GQPE data
    double *label;
    double *povm;

    // workspaces
    double *state_eigen;
    double *povm2;
    double *p;
};

void free_gqpe(struct gqpe *gqpe0)
{ free(gqpe0->energy); free(gqpe0->eigenbasis); free(gqpe0->label); free(gqpe0->povm); free(gqpe0->state_eigen); free(gqpe0->povm2); free(gqpe0->p); }

// initialize & precompute POVM for approximate GQPE operation from preset parameters
void init_gqpe(struct gqpe *gqpe0)
{
    int n = 1<<gqpe0->num_site;

    // construct & diagonalize Hamiltonian
    gqpe0->energy = (double*)malloc(sizeof(double)*n);
    gqpe0->eigenbasis = (double*)malloc(sizeof(double)*n*n);
    init_hamiltonian(gqpe0->num_site, gqpe0->field, gqpe0->coupling, gqpe0->eigenbasis);
    init_eigenbasis(gqpe0->num_site, gqpe0->eigenbasis, gqpe0->energy);

    // set epsilon-dependent parameters
    double Emax = fabs(gqpe0->energy[0]);
    for(int i=1 ; i<n ; i++)
    { if(Emax < fabs(gqpe0->energy[i])) { Emax = fabs(gqpe0->energy[i]); } }
    gqpe0->tmax = log(2.0/gqpe0->epsilon)/M_PI;
    gqpe0->num_ancilla = ceil(log(log(2.0/gqpe0->epsilon)*(2.0*Emax/(M_PI*M_PI) + 4.0/M_PI))/log(2.0));
    gqpe0->num_ancilla2 = ceil(log(log(2.0/gqpe0->epsilon)*4.0/M_PI)/log(2.0));
    int m = 1<<gqpe0->num_ancilla, m2 = 1<<gqpe0->num_ancilla2;

    // allocate workspaces
    gqpe0->state_eigen = (double*)malloc(sizeof(double)*n);
    gqpe0->povm2 = (double*)malloc(sizeof(double)*m*n);
    gqpe0->p = (double*)malloc(sizeof(double)*m);

    // construct energy grid
    gqpe0->label = (double*)malloc(sizeof(double)*m);
    double wmax = (double)m*0.5*M_PI/gqpe0->tmax;
    for(int i=0 ; i<m ; i++)
    { gqpe0->label[i] = wmax*(double)(2*i+1-m)/(double)m; }

    // construct unnormalized povm outcomes
    gqpe0->povm = (double*)malloc(sizeof(double)*m*n);
    gqpe0->gamma = M_PI/gqpe0->tmax;
    for(int i=0 ; i<m ; i++)
    {
        for(int j=0 ; j<n ; j++)
        {
            double energy_diff = gqpe0->label[i] - gqpe0->energy[j];
            double complex povm0 = 0.0;
            for(int k=0 ; k<m ; k++)
            {
                double t = gqpe0->tmax*(double)(2*k+1-m)/(double)m;
                for(int l=0 ; l<m2 ; l++)
                {
                    double omega = gqpe0->label[l+(m-m2)/2];
                    povm0 += cexp(I*(energy_diff - omega)*t)*exp(-0.25*omega*omega/gqpe0->gamma)/(double)m;
                }
            }
            gqpe0->povm[j+i*n] = creal(povm0);
        }
    }

    // normalize to match the trace of identity matrix
    double norm = 0.0;
    for(int i=0 ; i<m ; i++)
    for(int j=0 ; j<n ; j++)
    { norm += gqpe0->povm[j+i*n]*gqpe0->povm[j+i*n]; }
    norm /= (double)n;
    norm = 1.0/sqrt(norm);
    for(int i=0 ; i<m ; i++)
    for(int j=0 ; j<n ; j++)
    { gqpe0->povm[j+i*n] *= norm; }

    // store squares of POVM elements
    for(int i=0 ; i<m*n ; i++)
    { gqpe0->povm2[i] = gqpe0->povm[i]*gqpe0->povm[i]; }
}

// perform diagonal POVM
double measure_gqpe(double *state, struct gqpe *gqpe0)
{
    int n = 1<<gqpe0->num_site, m = 1<<gqpe0->num_ancilla;

    // transform state to eigenbasis
    transform('T', gqpe0->num_site, gqpe0->eigenbasis, state, gqpe0->state_eigen);

    // decide on the measurement outcome (use BLAS & state to temporarily store eigenstate probabilities)
    double p0 = random_uniform();
    for(int i=0 ; i<n ; i++)
    { state[i] = gqpe0->state_eigen[i]*gqpe0->state_eigen[i]; }
    char trans = 'T';
    int inc = 1;
    double zero = 0.0, one = 1.0;
    dgemv_(&trans, &n, &m, &one, gqpe0->povm2, &n, state, &inc, &zero, gqpe0->p, &inc);
    int outcome = m-1;
    double p_sum = 0.0;
    for(int i=0 ; i<m ; i++)
    {
        p_sum += gqpe0->p[i];
        if(p_sum >= p0)
        {
            outcome = i;
            break;
        }
    }

    // collapse the state according to the outcome
    double wt = 0.0;
    for(int i=0 ; i<n ; i++)
    {
        gqpe0->state_eigen[i] *= gqpe0->povm[i+outcome*n];
        wt += gqpe0->state_eigen[i]*gqpe0->state_eigen[i];
    }
    wt = 1.0/sqrt(wt);
    for(int i=0 ; i<n ; i++)
    { gqpe0->state_eigen[i] *= wt; }

    // transform state back to site basis
    transform('N', gqpe0->num_site, gqpe0->eigenbasis, state, gqpe0->state_eigen);

    return gqpe0->label[outcome];
}

// postselect GQPE state for a given outcome
void post_gqpe(double label0, double *state, struct gqpe *gqpe0)
{
    int n = 1<<gqpe0->num_site, m = 1<<gqpe0->num_ancilla;

    // find outcome associated with label
    int outcome = -1;
    for(int i=0 ; i<m ; i++)
    { if(label0 == gqpe0->label[i]) { outcome = i; } }

    // transform state to eigenbasis
    transform('T', gqpe0->num_site, gqpe0->eigenbasis, state, gqpe0->state_eigen);

    // collapse the state according to the outcome
    double wt = 0.0;
    for(int i=0 ; i<n ; i++)
    {
        gqpe0->state_eigen[i] *= gqpe0->povm[i+outcome*n];
        wt += gqpe0->state_eigen[i]*gqpe0->state_eigen[i];
    }
    wt = 1.0/sqrt(wt);
    for(int i=0 ; i<n ; i++)
    { gqpe0->state_eigen[i] *= wt; }

    // transform state back to site basis
    transform('N', gqpe0->num_site, gqpe0->eigenbasis, state, gqpe0->state_eigen);
}

// success probability of GQPE-based postselection of thermal states
double postselection_probability(double omega_min, int num_outcome, double *energy_outcome, double *p_outcome)
{
    double p = 0.0;
    for(int i=0 ; i<num_outcome ; i++)
    {
        double p_acceptance = 1.0;
        if(energy_outcome[i] > omega_min)
        { p_acceptance = exp(omega_min - energy_outcome[i]); }
        p += p_outcome[i]*p_acceptance;
    }
    return p;
}

// observable expectation values of GQPE-based postselection of thermal states
void postselection_obs(double omega_min, int num_outcome, double *energy_outcome, double *p_outcome, double *obs_outcome,
                       double *mean, double *err)
{
    double ev = 0.0, ev2 = 0.0, wt = 1.0/postselection_probability(omega_min, num_outcome, energy_outcome, p_outcome);
    for(int i=0 ; i<num_outcome ; i++)
    {
        double p_acceptance = 1.0;
        if(energy_outcome[i] > omega_min)
        { p_acceptance = exp(omega_min - energy_outcome[i]); }
        ev += obs_outcome[i]*wt*p_outcome[i]*p_acceptance;
        ev2 += obs_outcome[i]*obs_outcome[i]*wt*p_outcome[i]*p_acceptance;
    }
    *mean = ev;
    *err = sqrt(fabs(ev2 - ev*ev));
}

// optimize bracketed omega_min to reduce bias below standard error
#define BIAS_SUPPRESSION 0.1
#define POSTSELECTION_TOLERANCE 0.001
double postselection_optimize(double mean0, int upper_index, int num_sample, int num_outcome, double *energy_outcome, double *p_outcome, double *obs_outcome)
{
    double lower_bound = energy_outcome[upper_index-1], upper_bound = energy_outcome[upper_index];

    while(upper_bound - lower_bound > 0.001)
    {
        double new_omega = 0.5*(lower_bound + upper_bound), mean, err;
        postselection_obs(new_omega, num_outcome, energy_outcome, p_outcome, obs_outcome, &mean, &err);
        if(fabs(mean - mean0) > BIAS_SUPPRESSION*err/sqrt(num_sample))
        { upper_bound = new_omega; }
        else
        { lower_bound = new_omega; }
    }
    return lower_bound;
}

// tune omega_min to reduce bias below standard error
double postselection_cutoff(int min_outcome, int num_sample, int num_outcome, double *energy_outcome, double *p_outcome, double *site_outcome)
{
    double omega_min, energy_mean0, energy_err0, site_mean0, site_err0;
    postselection_obs(energy_outcome[min_outcome], num_outcome, energy_outcome, p_outcome, energy_outcome, &energy_mean0, &energy_err0);
    postselection_obs(energy_outcome[min_outcome], num_outcome, energy_outcome, p_outcome, site_outcome, &site_mean0, &site_err0);
    for(int i=min_outcome+1 ; i<num_outcome ; i++)
    {
        omega_min = energy_outcome[i];
        double energy_mean, energy_err, site_mean, site_err;
        postselection_obs(omega_min, num_outcome, energy_outcome, p_outcome, energy_outcome, &energy_mean, &energy_err);
        postselection_obs(omega_min, num_outcome, energy_outcome, p_outcome, site_outcome, &site_mean, &site_err);

        // if a bias gets too large, return an optimized omega_min value from the bracketed interval
        int stop = 0;
        double cut1 = energy_outcome[num_outcome-1], cut2 = energy_outcome[num_outcome-1];
        if(fabs(energy_mean - energy_mean0) > BIAS_SUPPRESSION*energy_err/sqrt(num_sample))
        { stop = 1; cut1 = postselection_optimize(energy_mean0, i, num_sample, num_outcome, energy_outcome, p_outcome, energy_outcome); }
        if(fabs(site_mean - site_mean0) > BIAS_SUPPRESSION*site_err/sqrt(num_sample))
        { stop = 1; cut2 = postselection_optimize(site_mean0, i, num_sample, num_outcome, energy_outcome, p_outcome, site_outcome); }
        if(stop)
        { if(cut1 < cut2) { return cut1; } else { return cut2; } }
    }
    return energy_outcome[num_outcome-1];
}

// quantum Metropolis algorithm
void metropolis(int flip_site, int n_max, double *state, struct gqpe *gqpe0, int *n, double *E, int *a)
{
    double u, omega;
    int num_omega = 1<<(gqpe0->num_ancilla);
    double *cp_accept = (double*)malloc(sizeof(double)*num_omega);
    double *cp_reject = (double*)malloc(sizeof(double)*num_omega);

    // line 2
    *E = measure_gqpe(state, gqpe0);
    // line 3
    *a = measure_basis(gqpe0->num_site, state);
    // line 4
    int b = (*a)^(1<<flip_site); // spin-flip operation
    // line 5
    int c = *a;
    *n = 0;
    double q_max = 0.0;
    double q = 0.0;
    // line 6
    do
    {
        // line 7
        int swap = b;
        b = c;
        c = swap;
        (*n)++;
        if(q_max < q) { q_max = q; }
        // line 8
        if(*n <= 2) // reuse simulation results after 2nd iteration
        {
            for(int i=0 ; i<(1<<(gqpe0->num_site)) ; i++)
            { state[i] = 0.0; }
            state[c] = 1.0;
        }
        // line 9
        if(*n <= 2) // reuse simulation results after 2nd iteration
        { omega = measure_gqpe(state, gqpe0); }
        if(*n == 1) // save acceptance branch
        {
            cp_accept[0] = gqpe0->p[0];
            for(int i=1 ; i<num_omega ; i++)
            { cp_accept[i] = cp_accept[i-1] + gqpe0->p[i]; }
        }
        else if(*n == 2) // save rejection branch
        {
            cp_reject[0] = gqpe0->p[0];
            for(int i=1 ; i<num_omega ; i++)
            { cp_reject[i] = cp_reject[i-1] + gqpe0->p[i]; }
        }
        else if((*n)%2 == 1) // reuse acceptance branch statistics
        { omega = gqpe0->label[random_list(num_omega, cp_accept)]; }
        else if((*n)%2 == 0) // reuse rejection branch statistics
        { omega = gqpe0->label[random_list(num_omega, cp_reject)]; }
        // line 10
        q = exp(*E - omega - gqpe0->gamma);
        // line 11
        u = random_uniform();
    // line 12
    } while(u > (q - q_max)/(1.0 - q_max) && *n != n_max);
    // line 13 (E & a are returned through pointers in the argument list, state is postselected on measurement outcomes)
    for(int i=0 ; i<(1<<(gqpe0->num_site)) ; i++)
    { state[i] = 0.0; }
    state[c] = 1.0;
    post_gqpe(omega, state, gqpe0);
    free(cp_accept);
    free(cp_reject);
    return;
}

int main(int argc, char **argv)
{
    if(argc < 7) { printf("SYNTAX: <executable> <seed> <num_site> <num_sample> <epsilon> <field> <coupling>\n"); exit(1); }
    struct gqpe gqpe0;
    int seed, num_sample;
    sscanf(argv[1], "%d", &seed);
    sscanf(argv[2], "%d", &(gqpe0.num_site));
    sscanf(argv[3], "%d", &num_sample);
    sscanf(argv[4], "%lf", &(gqpe0.epsilon));
    sscanf(argv[5], "%lf", &(gqpe0.field));
    sscanf(argv[6], "%lf", &(gqpe0.coupling));
    random64(seed);

    // initial the GQPE measurement
    init_gqpe(&gqpe0);
    printf("tmax = %e\n", gqpe0.tmax);
    printf("num_ancilla = %d\n", gqpe0.num_ancilla);
    printf("num_ancilla2 = %d\n", gqpe0.num_ancilla2);
    int n_max = floor(0.5*log(2.0)/log(1.0+gqpe0.epsilon)) - 1;

    // calculate the site moments for each eigenstate
    int N = 1<<gqpe0.num_site;
    double *site_moment = (double*)malloc(sizeof(double)*N);
    for(int i=0 ; i<N ; i++)
    {
        site_moment[i] = 0.0;
        for(int j=0 ; j<N ; j++)
        {
            int filter = 0;
            if(j&1) { filter ^= 1; }
            if(j&2) { filter ^= 1; }
            if(filter)
            { site_moment[i] -= pow(gqpe0.eigenbasis[j+i*N],2); }
            else
            { site_moment[i] += pow(gqpe0.eigenbasis[j+i*N],2); }
        }
    }

    // calculate reference expectation values
    double Z = 0.0, energy_ref = 0.0, site_ref = 0.0;
    for(int i=0 ; i<N ; i++)
    {
        Z += exp(-gqpe0.energy[i]);
        energy_ref += gqpe0.energy[i]*exp(-gqpe0.energy[i]);
        site_ref += site_moment[i]*exp(-gqpe0.energy[i]);
    }
    energy_ref /= Z;
    site_ref /= Z;

    // calculate GQPE thermal average & standard error
    int num_label = 1<<gqpe0.num_ancilla;
    double omega_ave = 0.0, omega_err = 0.0;
    for(int i=0 ; i<num_label ; i++)
    for(int j=0 ; j<N ; j++)
    {
        omega_ave += gqpe0.label[i]*gqpe0.povm2[j+i*N]*exp(-gqpe0.energy[j])/Z;
        omega_err += gqpe0.label[i]*gqpe0.label[i]*gqpe0.povm2[j+i*N]*exp(-gqpe0.energy[j])/Z;
    }
    omega_err = sqrt(omega_err - omega_ave*omega_ave);
    printf("reference (E_min, E, site, omega, d_omega) = %e %e %e %e %e\n", gqpe0.energy[0], energy_ref, site_ref, omega_ave, omega_err);

    // calculate max-mix probabilities & moments
    double *p_mix = (double*)malloc(sizeof(double)*num_label);
    double *site_mix = (double*)malloc(sizeof(double)*num_label);
    for(int i=0 ; i<num_label ; i++)
    {
        p_mix[i] = 0.0;
        site_mix[i] = 0.0;
        for(int j=0 ; j<N ; j++)
        {
            p_mix[i] += gqpe0.povm2[j+i*N]/(double)N;
            site_mix[i] += site_moment[j]*gqpe0.povm2[j+i*N]/(double)N;
        }
    }

    // ignore outcomes with extremely low thermal probability for numerical stability
    int min_outcome = 0;
    double p_acc = 0.0;
    for(int i=0 ; i<num_label ; i++)
    {
        for(int j=0 ; j<N ; j++)
        { p_acc += gqpe0.povm2[j+i*N]*exp(-gqpe0.energy[j])/Z; }
        if(p_acc > CP_CUTOFF) { break; }
        else { min_outcome++; }
    }

    // efficiency analysis for direct sampling
    double omega_min = postselection_cutoff(min_outcome, num_sample, num_label, gqpe0.label, p_mix, site_mix);
    double p_post = postselection_probability(omega_min, num_label, gqpe0.label, p_mix);
    printf("postselection (omega_min, probability) = %e %e\n", omega_min, p_post);

    // initialize to a uniformly random state
    double *state = (double*)malloc(sizeof(double)*N);
    for(int i=0 ; i<N ; i++)
    { state[i] = 0.0; }
    state[(int)floor(random_uniform()*N)] = 1.0;

    // gather statistics from the Metropolis Markov chain
    double energy_ave = 0.0, site_ave = 0.0, site;
    long long int n_tot = 0;
    double start_time = omp_get_wtime();
    for(int i=0 ; i<num_sample ; i++)
    {
        int flip = i%gqpe0.num_site, n, a;
        double E;
        metropolis(flip, n_max, state, &gqpe0, &n, &E, &a);
        energy_ave += E;
        // extract an instantaneous site average from the computational basis state
        site = 0.0;
        for(int j=0 ; j<gqpe0.num_site ; j++)
        {
            int filter = 0;
            if(a&(1<<j)) { filter ^= 1; }
            if(a&(1<<((j+1)%gqpe0.num_site))) { filter ^= 1; }
            if(filter) { site -= 1.0; }
            else { site += 1.0; }
        }
        site /= (double)gqpe0.num_site;
        site_ave += site;
        n_tot += n;
        printf("%d %d %e %e\n", i, n, E, site);
        fflush(stdout);
    }
    printf("energy_ave = %e (%e)\n", energy_ave/(double)num_sample, energy_ref);
    printf("site_ave = %e (%e)\n", site_ave/(double)num_sample, site_ref);
    printf("iter_tot = %lld\n", n_tot);
    double end_time = omp_get_wtime();
    printf("elapsed_time = %e\n",end_time-start_time);

    free(p_mix);
    free(site_mix);
    free(site_moment);
    free(state);
    free_gqpe(&gqpe0);
    return 0;
}
