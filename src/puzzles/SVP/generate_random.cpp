#include <sstream>
#include <iostream>
#include <NTL/LLL.h>

NTL_CLIENT;

#include "tools.h"
#include <chrono>

int main(int argc,char** argv)
{
    long n = 80;
    long bit = 20;
    ZZ seed; seed = 0;

    PARSE_MAIN_ARGS {
	MATCH_MAIN_ARGID("--dim",n);
	MATCH_MAIN_ARGID("--seed",seed);
//	MATCH_MAIN_ARGID("--bit",bit);
	SYNTAX();
    }
	int count = 1<<12;
	auto start = std::chrono::high_resolution_clock::now();
	vec_ZZ v;
	for(int i=0;i<count;i++)
	     generate_random_HNF(v,n,bit,seed);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << duration.count() << "ms" << std::endl;

    //mat_ZZ B; B.SetDims(n,n); clear(B);
    //B(1,1) = v(1);
    //for (int i=2; i<=n; i++)
    //{
	//B(i,1)=v(i);
	//B(i,i)=1;
    //}
    //cout << B << endl;
}
