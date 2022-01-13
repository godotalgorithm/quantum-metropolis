// simulation of the quantum rejection-free Metropolis algorithm on the transverse-field Ising model
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

// NOTE: This implementation uses several convenient simplifications that cause a few superficial
//       deviations from the paper. Beta is absorbed into the Hamiltonian and doesn't explicitly appear anywhere,
//       which necessitates a rescaling of the output energies. Pauli X & Z are exchanged, which means that
//       the simulation differs from the paper by a Hadamard transformation of each qubit. These simulations
//       are thus equivalent up to a simple unitary transformation and produce equivalent statistics.

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

    return 2.0*outcome - 1.0;
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
        int filter = 1<<i;
        for(int j=0 ; j<n ; j++)
        {
            if(filter&j)
            { hamiltonian[j+j*n] += field; }
            else
            { hamiltonian[j+j*n] += -field; }
        }

        // coupling term
        int flip = (1<<i) ^ (1<<((i+1)%num_site));
        for(int j=0 ; j<n ; j++)
        { hamiltonian[j+(j^flip)*n] += coupling; }
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
    gqpe0->tmax = log(1.0/gqpe0->epsilon)/M_PI;
    gqpe0->num_ancilla = ceil(log(log(1.0/gqpe0->epsilon)*(2.0*Emax/(M_PI*M_PI) + 4.0/M_PI))/log(2.0));
    gqpe0->num_ancilla2 = ceil(log(log(1.0/gqpe0->epsilon)*4.0/M_PI)/log(2.0));
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
    gqpe0->gamma = 2.0*M_PI/gqpe0->tmax;
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
                    povm0 += cexp(I*(energy_diff - omega)*t)*exp(-0.5*omega*omega/gqpe0->gamma)/(double)m;
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

// quantum rejection-free Metropolis algorithm
void metropolis(int site, int max_iter, double *state, struct gqpe *gqpe0, int *iter, double *E, double *a)
{
    *iter = 0;
    double u;

    // line 2
    double q = 0.0;
    // line 3
    double q_max = 0.0;
    // line 4
    *E = measure_gqpe(state, gqpe0);
    // line 5
    *a = measure_site(site, gqpe0->num_site, state);
    // line 6
    do
    {
        // line 7
        int b = measure_site(site, gqpe0->num_site, state);
        flip_site(site, gqpe0->num_site, state);
        // line 8
        double E_new = measure_gqpe(state, gqpe0);
        // line 9
        if(q_max < q) { q_max = q; }
        // line 10
        q = exp(*E - E_new - 0.5*gqpe0->gamma);
        // line 11
        u = random_uniform();
    // line 12
    } while(++(*iter) < max_iter && u > (q - q_max)/(1.0 - q_max));
    // line 13
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
    int max_iter = ceil(log(2.0)/log((1.0+gqpe0.epsilon)/(1.0-gqpe0.epsilon)));

    // calculate reference expectation values
    int n = 1<<gqpe0.num_site;
    double Z = 0.0, energy_ref = 0.0, site_ref = 0.0;
    for(int i=0 ; i<n ; i++)
    {
        Z += exp(-gqpe0.energy[i]);
        energy_ref += gqpe0.energy[i]*exp(-gqpe0.energy[i]);
        double site_moment = 0.0;
        for(int j=0 ; j<n ; j++)
        {
            if(j&1)
            { site_moment += pow(gqpe0.eigenbasis[j+i*n],2); }
            else
            { site_moment -= pow(gqpe0.eigenbasis[j+i*n],2); }
        }
        site_ref += site_moment*exp(-gqpe0.energy[i]);
    }
    printf("energy_ref = %e\n", energy_ref/Z);
    printf("site_ref = %e\n", site_ref/Z);

    // initialize to a uniformly random state
    double *state = (double*)malloc(sizeof(double)*n);
    for(int i=0 ; i<n ; i++)
    { state[i] = 0.0; }
    state[(int)floor(random_uniform()*n)] = 1.0;

    // gather statistics from the Metropolis Markov chain
    double energy_ave = 0.0, site_ave = 0.0;
    int iter_tot = 0;
    double start_time = omp_get_wtime();
    for(int i=0 ; i<num_sample ; i++)
    {
        int site = i%gqpe0.num_site, iter;
        double E, a;
        metropolis(site, max_iter, state, &gqpe0, &iter, &E, &a);
        energy_ave += E;
        site_ave += a;
        iter_tot += iter;
        printf("%d %d %e %e\n", i, iter, E, a);
        fflush(stdout);
    }
    printf("energy_ave = %e\n", energy_ave/(double)num_sample);
    printf("site_ave = %e\n", site_ave/(double)num_sample);
    printf("iter_tot = %d\n", iter_tot);
    double end_time = omp_get_wtime();
    printf("elapsed_time = %e\n",end_time-start_time);

    free(state);
    free_gqpe(&gqpe0);
    return 0;
}