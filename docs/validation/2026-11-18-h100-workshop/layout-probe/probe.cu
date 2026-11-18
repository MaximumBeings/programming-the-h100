#include <cute/tensor.hpp>
#include <cstdio>
using namespace cute;
int main() {
 auto nested=make_layout(make_shape(make_shape(Int<3>{},Int<4>{}),Int<2>{}),make_stride(make_stride(Int<1>{},Int<3>{}),Int<40>{}));
 auto simplified=coalesce(nested);
 print("coalesced=");print(simplified);print("\n");
 for(int i=0;i<size(nested);++i)if(nested(i)!=simplified(i))return 1;
 auto product=logical_product(make_layout(Int<4>{},Int<1>{}),make_layout(Int<3>{},Int<1>{}));
 print("product=");print(product);print("\n");
 for(int i=0;i<4;++i)for(int j=0;j<3;++j)if(product(make_coord(i,j))!=i+4*j)return 1;
 auto compact=make_layout(Int<4>{},Int<1>{});
 auto compact_rest=complement(compact,Int<24>{});
 print("compact_rest="); print(compact_rest); print("\n");
 for(int i=0;i<size(compact_rest);++i) printf("%d%s",int(compact_rest(i)),i+1==size(compact_rest)?"\n":" ");
 auto even=make_layout(Int<4>{},Int<2>{});
 auto even_rest=complement(even,Int<16>{});
 print("even_rest="); print(even_rest); print("\n");
 for(int i=0;i<size(even_rest);++i) printf("%d%s",int(even_rest(i)),i+1==size(even_rest)?"\n":" ");
 int owners[16]={};
 for(int r=0;r<size(even_rest);++r)
   for(int t=0;t<size(even);++t){int x=int(even(t)+even_rest(r)); if(x<0||x>=16)return 1; ++owners[x];}
 for(int n:owners)if(n!=1)return 1;
 print("COMPLEMENT PASS: 16 unique offsets from strided tile plus complement\n");
}
