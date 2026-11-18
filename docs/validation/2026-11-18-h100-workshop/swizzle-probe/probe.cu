#include <cute/tensor.hpp>
#include <cstdio>
int main(){
 auto linear=cute::make_layout(cute::make_shape(cute::Int<32>{},cute::Int<32>{}),cute::make_stride(cute::Int<32>{},cute::Int<1>{}));
 auto good=cute::composition(cute::Swizzle<5,0,5>{},linear);
 auto bad=cute::composition(cute::Swizzle<3,2,-3>{},linear);
 int errors=0;
 for(int i=0;i<32;++i)for(int j=0;j<32;++j){errors+=good(cute::make_coord(i,j))!=(32*i+(j^i));errors+=bad(cute::make_coord(i,j))!=(32*(i^(j>>2))+j);}
 printf("good(3,5)=%d bad(3,20)=%d errors=%d\n",int(good(cute::make_coord(3,5))),int(bad(cute::make_coord(3,20))),errors);
 return errors?1:0;
}
