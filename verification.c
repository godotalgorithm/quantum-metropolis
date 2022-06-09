// verification of the quantum Metropolis algorithm
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

// state transition operation
int transition(int input, int num, double *rate)
{
    int output = 0;
    double accumulator = rate[input*num];
    double threshold = random_uniform();
    while(accumulator < threshold && output < num-1)
    { accumulator += rate[(++output) + input*num]; }
    return output;
}

// quantum Metropolis algorithm (w/ eigenstate transitions)
// NOTE: width = 0 is the classical case
int metropolis(int a, int num, int max_iter, double width, double *energy, double *rate)
{
    double u;

    // line 2
    double E = energy[a] + random_normal()*sqrt(width);
    // line 3 (trivial here)
    // line 4
    int b = transition(a, num, rate);
    // line 5
    int c = a;
    int n = 0;
    double q_max = 0.0;
    double q = 0.0;
    // line 6
    do
    {
        // line 7
        int swap = b;
        b = c;
        c = swap;
        n++;
        if(q_max < q) { q_max = q; }
        // line 8
        a = c;
        // line 9
        double omega = energy[a] + random_normal()*sqrt(width);
        // line 10
        q = exp(E - omega - width);
        // line 11
        u = random_uniform();
    // line 12
    } while(u > (q - q_max)/(1.0 - q_max) && n != max_iter);
    // line 13 (internal state returned, as in the classical algorithm)
    return a;
}

// arguments: seed, max_iter, num_data, & width
int main(int argc, char **argv)
{
    if(argc < 6) { printf("SYNTAX: <executable> <seed> <dimension> <max_iter> <num_sample> <width>\n"); exit(1); }
    int seed, dim, max_iter, num_sample;
    double width;
    sscanf(argv[1], "%d", &seed);
    sscanf(argv[2], "%d", &dim);
    sscanf(argv[3], "%d", &max_iter);
    sscanf(argv[4], "%d", &num_sample);
    sscanf(argv[5], "%lf", &width);
    random64(seed);

    double *p = (double*)malloc(sizeof(double)*dim);
    double *energy = (double*)malloc(sizeof(double)*dim);
    double *rate = (double*)malloc(sizeof(double)*dim*dim);

    // randomly assign state energies in [-1,1]
    double norm = 0.0;
    for(int i=0 ; i<dim ; i++)
    {
        energy[i] = 2.0*random_uniform() - 1.0;
        p[i] = exp(-energy[i]);
        norm += p[i];
    }
    for(int i=0 ; i<dim ; i++)
    { p[i] /= norm; }

    // randomly assign transition rates & normalize
    for(int i=0 ; i<dim ; i++)
    {
        rate[i+i*dim] = 0.0;
        for(int j=0 ; j<i ; j++)
        { rate[j+i*dim] = rate[i+j*dim] = random_uniform(); }
    }
    double max_norm = 0.0;
    for(int i=0 ; i<dim ; i++)
    {
        norm = 0.0;
        for(int j=0 ; j<dim ; j++)
        { norm += rate[j+i*dim]; }
        if(norm > max_norm) { max_norm = norm; }
    }
    for(int i=0 ; i<dim*dim ; i++)
    { rate[i] /= max_norm; }
    for(int i=0 ; i<dim ; i++)
    {
        norm = 0.0;
        for(int j=0 ; j<dim ; j++)
        { norm += rate[j+i*dim]; }
        rate[i+i*dim] = 1.0 - norm;
    }

    // set up block averaging buffer
    int num_block = 0;
    int num_sample2 = num_sample;
    while(num_sample2) { num_block++; num_sample2 >>= 1; }
    int *block_counter = (int*)malloc(sizeof(int)*num_block);
    double *p_block = (double*)malloc(sizeof(double)*dim*num_block);
    double *p_block_ave = (double*)malloc(sizeof(double)*dim*num_block);
    double *p_block_var = (double*)malloc(sizeof(double)*dim*num_block);
    for(int i=0 ; i<num_block ; i++)
    {
        block_counter[i] = 0;
        for(int j=0 ; j<dim ; j++)
        { p_block[j+i*dim] = p_block_ave[j+i*dim] = p_block_var[j+i*dim] = 0.0; }
    }

    // gather statistics from the Metropolis Markov chain
    int state = 0;
    for(int i=1 ; i<dim ; i++)
    { if(p[i] > p[state]) { state = i; } }
    for(int i=0 ; i<num_sample ; i++)
    {
        state = metropolis(state, dim, max_iter, width, energy, rate);
        for(int j=0 ; j<num_block ; j++)
        {
            p_block[state+j*dim] += 1.0;
            block_counter[j]++;
            if(block_counter[j] == 1<<j)
            {
                block_counter[j] = 0;
                for(int k=0 ; k<dim ; k++)
                {
                    double p_block0 = p_block[k+j*dim]/(double)(1<<j);
                    p_block_ave[k+j*dim] += p_block0;
                    p_block_var[k+j*dim] += p_block0*p_block0;
                    p_block[k+j*dim] = 0.0;
                }
            }
        }
    }

    // print observed block statistics
    printf("%d",-1);
    for(int i=0 ; i<dim ; i++)
    { printf("  %e %e",p[i],0.0); }
    printf("\n");
    for(int i=0 ; i<num_block ; i++)
    {
        printf("%d", i);
        for(int j=0 ; j<dim ; j++)
        {
            double mean = p_block_ave[j+i*dim]/(double)(num_sample>>i);
            double var = p_block_var[j+i*dim]/(double)(num_sample>>i) - mean*mean;
            printf("  %e %e", mean, sqrt(var/(double)(num_sample>>i)));
        }
        printf("\n");
    }

    free(p);
    free(energy);
    free(rate);
    free(block_counter);
    free(p_block);
    free(p_block_ave);
    free(p_block_var);

    return 0;
}
