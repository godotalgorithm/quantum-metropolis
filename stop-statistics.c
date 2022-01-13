// quantum Metropolis stopping-time statistics for a single-energy Hamiltonian
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

// quantum rejection-free Metropolis algorithm for a single-energy Hamiltonian
int metropolis_quantum(double width, int max_iter)
{
    int n = 0;
    double u;

    // line 2
    double q = 0.0;
    // line 3
    double q_max = 0.0;
    // line 4
    double E = random_normal()*sqrt(0.5*width);
    // line 6
    do
    {
        n++;
        // line 8
        double E_new = random_normal()*sqrt(0.5*width);
        // line 9
        if(q > q_max) { q_max = q; }
        // line 10
        q = exp(E - E_new - 0.5*width);
        // line 11
        u = random_uniform();
    // line 12
    } while(u > (q - q_max)/(1.0 - q_max) && n < max_iter);
    // line 13
    return n;
}

int main(int argc, char **argv)
{
    if(argc < 4) { printf("SYNTAX: <executable> <seed> <num_sample> <max_iter>\n"); exit(1); }
    int seed, num_sample, max_iter;
    sscanf(argv[1], "%d", &seed);
    sscanf(argv[2], "%d", &num_sample);
    sscanf(argv[3], "%d", &max_iter);

    random64(seed);

    // set the width that balances against other errors
    double width = 4.0*M_PI*M_PI/log((double)max_iter);

    double moment[4];
    for(int i=0 ; i<4 ; i++)
    { moment[i] = 0.0; }
    for(int i=0 ; i<num_sample ; i++)
    {
        int num_iter = metropolis_quantum(width, max_iter);
        for(int j=0 ; j<4 ; j++)
        { moment[j] += pow(num_iter,j+1); }
    }
    for(int i=0 ; i<4 ; i++)
    { moment[i] /= (double)num_sample; }

    // recenter moments around the mean
    moment[3] = moment[3] - 4.0*moment[0]*moment[2] + 6.0*pow(moment[0],2)*moment[1] - 3.0*pow(moment[0],4);
    moment[2] = moment[2] - 3.0*moment[0]*moment[1] + 2.0*pow(moment[0],3);
    moment[1] = moment[1] - pow(moment[0],2);

    printf("%d %d %e %e %e %e\n", num_sample, max_iter, moment[0], sqrt(moment[1]/(double)num_sample), moment[1], sqrt((moment[3] - pow(moment[1],2))/(double)num_sample));

    return 0;
}