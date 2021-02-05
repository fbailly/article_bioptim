/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) model_2021_02_05_12495939_cost_y_e_hess_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

static const casadi_int casadi_s0[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s1[3] = {0, 0, 0};
static const casadi_int casadi_s2[11] = {7, 1, 0, 7, 0, 1, 2, 3, 4, 5, 6};
static const casadi_int casadi_s3[11] = {4, 4, 0, 2, 4, 4, 4, 0, 1, 0, 1};

/* model_2021_02_05_12495939_cost_y_e_hess:(i0[4],i1[],i2[7],i3[])->(o0[4x4,4nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a4, a5, a6, a7, a8, a9;
  a0=5.8942910739687680e-02;
  a1=1.0668228978597614e-01;
  a2=-8.0520021982363552e-03;
  a3=arg[2]? arg[2][6] : 0;
  a4=(a2*a3);
  a5=(a1*a4);
  a6=-6.2511643845589493e-01;
  a7=arg[0]? arg[0][1] : 0;
  a8=cos(a7);
  a9=(a6*a8);
  a10=7.7320652313482108e-01;
  a11=sin(a7);
  a12=(a10*a11);
  a9=(a9-a12);
  a12=-1.7676989936934065e-01;
  a13=(a12*a3);
  a14=(a9*a13);
  a5=(a5+a14);
  a14=(a10*a8);
  a15=(a6*a11);
  a14=(a14+a15);
  a15=-1.3071942448447652e-01;
  a16=(a15*a3);
  a17=(a14*a16);
  a5=(a5+a17);
  a17=1.6671699644008754e-02;
  a18=(a17*a3);
  a5=(a5+a18);
  a5=(a0*a5);
  a18=9.9923839826218319e-01;
  a19=3.8499763654015404e-02;
  a20=arg[2]? arg[2][5] : 0;
  a21=(a2*a20);
  a22=(a19*a21);
  a23=7.7965829948778420e-01;
  a24=(a23*a8);
  a25=6.2502056305245079e-01;
  a26=(a25*a11);
  a24=(a24-a26);
  a26=(a12*a20);
  a27=(a24*a26);
  a22=(a22+a27);
  a8=(a25*a8);
  a11=(a23*a11);
  a8=(a8+a11);
  a11=(a15*a20);
  a27=(a8*a11);
  a22=(a22+a27);
  a27=-2.8994080392798782e-01;
  a28=(a27*a20);
  a22=(a22+a28);
  a28=(a18*a22);
  a5=(a5+a28);
  a28=-3.8952964437603196e-02;
  a21=(a1*a21);
  a29=(a9*a26);
  a21=(a21+a29);
  a29=(a14*a11);
  a21=(a21+a29);
  a20=(a17*a20);
  a21=(a21+a20);
  a20=(a28*a21);
  a5=(a5+a20);
  a20=3.9020807762349584e-02;
  a29=arg[2]? arg[2][4] : 0;
  a2=(a2*a29);
  a30=(a19*a2);
  a12=(a12*a29);
  a31=(a24*a12);
  a30=(a30+a31);
  a15=(a15*a29);
  a31=(a8*a15);
  a30=(a30+a31);
  a31=(a27*a29);
  a30=(a30+a31);
  a31=(a20*a30);
  a5=(a5+a31);
  a31=9.9750107761097473e-01;
  a1=(a1*a2);
  a9=(a9*a12);
  a1=(a1+a9);
  a14=(a14*a15);
  a1=(a1+a14);
  a17=(a17*a29);
  a1=(a1+a17);
  a17=(a31*a1);
  a5=(a5+a17);
  a17=arg[0]? arg[0][0] : 0;
  a29=cos(a17);
  a5=(a5*a29);
  a19=(a19*a4);
  a24=(a24*a13);
  a19=(a19+a24);
  a8=(a8*a16);
  a19=(a19+a8);
  a27=(a27*a3);
  a19=(a19+a27);
  a19=(a0*a19);
  a22=(a28*a22);
  a19=(a19+a22);
  a21=(a18*a21);
  a19=(a19-a21);
  a30=(a31*a30);
  a19=(a19+a30);
  a1=(a20*a1);
  a19=(a19-a1);
  a1=sin(a17);
  a19=(a19*a1);
  a5=(a5-a19);
  if (res[0]!=0) res[0][0]=a5;
  a5=cos(a7);
  a19=cos(a17);
  a1=(a18*a19);
  a30=sin(a17);
  a21=(a28*a30);
  a1=(a1-a21);
  a21=(a26*a1);
  a22=(a0*a30);
  a27=(a13*a22);
  a21=(a21-a27);
  a27=(a20*a19);
  a3=(a31*a30);
  a27=(a27-a3);
  a3=(a12*a27);
  a21=(a21+a3);
  a3=(a10*a21);
  a8=(a0*a19);
  a24=(a13*a8);
  a4=(a18*a30);
  a29=(a28*a19);
  a4=(a4+a29);
  a29=(a26*a4);
  a24=(a24+a29);
  a30=(a20*a30);
  a19=(a31*a19);
  a30=(a30+a19);
  a19=(a12*a30);
  a24=(a24+a19);
  a19=(a25*a24);
  a3=(a3-a19);
  a8=(a16*a8);
  a4=(a11*a4);
  a8=(a8+a4);
  a30=(a15*a30);
  a8=(a8+a30);
  a30=(a23*a8);
  a3=(a3+a30);
  a1=(a11*a1);
  a22=(a16*a22);
  a1=(a1-a22);
  a27=(a15*a27);
  a1=(a1+a27);
  a27=(a6*a1);
  a3=(a3-a27);
  a5=(a5*a3);
  a3=sin(a7);
  a24=(a23*a24);
  a21=(a6*a21);
  a24=(a24-a21);
  a8=(a25*a8);
  a24=(a24+a8);
  a1=(a10*a1);
  a24=(a24-a1);
  a3=(a3*a24);
  a5=(a5-a3);
  if (res[0]!=0) res[0][1]=a5;
  a5=cos(a17);
  a3=cos(a7);
  a24=(a23*a3);
  a1=sin(a7);
  a8=(a25*a1);
  a24=(a24-a8);
  a8=(a16*a24);
  a21=(a23*a1);
  a27=(a25*a3);
  a21=(a21+a27);
  a27=(a13*a21);
  a8=(a8-a27);
  a8=(a0*a8);
  a27=(a11*a24);
  a22=(a26*a21);
  a27=(a27-a22);
  a22=(a28*a27);
  a8=(a8+a22);
  a22=(a6*a3);
  a30=(a10*a1);
  a22=(a22-a30);
  a30=(a11*a22);
  a1=(a6*a1);
  a3=(a10*a3);
  a1=(a1+a3);
  a3=(a26*a1);
  a30=(a30-a3);
  a3=(a18*a30);
  a8=(a8-a3);
  a24=(a15*a24);
  a21=(a12*a21);
  a24=(a24-a21);
  a21=(a31*a24);
  a8=(a8+a21);
  a21=(a15*a22);
  a3=(a12*a1);
  a21=(a21-a3);
  a3=(a20*a21);
  a8=(a8-a3);
  a5=(a5*a8);
  a8=sin(a17);
  a22=(a16*a22);
  a1=(a13*a1);
  a22=(a22-a1);
  a22=(a0*a22);
  a27=(a18*a27);
  a22=(a22+a27);
  a30=(a28*a30);
  a22=(a22+a30);
  a24=(a20*a24);
  a22=(a22+a24);
  a21=(a31*a21);
  a22=(a22+a21);
  a8=(a8*a22);
  a5=(a5+a8);
  if (res[0]!=0) res[0][2]=a5;
  a5=-1.0724256777678894e-01;
  a8=9.9826135519388559e-01;
  a22=(a8*a13);
  a21=2.2999999889266845e-03;
  a24=(a21*a26);
  a22=(a22+a24);
  a24=-5.8898019716436364e-02;
  a30=(a24*a12);
  a22=(a22+a30);
  a30=(a5*a22);
  a27=sin(a17);
  a1=(a0*a27);
  a3=(a1*a13);
  a17=cos(a17);
  a4=(a18*a17);
  a19=(a28*a27);
  a4=(a4-a19);
  a19=(a4*a26);
  a3=(a3-a19);
  a19=(a20*a17);
  a29=(a31*a27);
  a19=(a19-a29);
  a29=(a19*a12);
  a3=(a3-a29);
  a29=(a25*a3);
  a30=(a30-a29);
  a0=(a0*a17);
  a13=(a0*a13);
  a28=(a28*a17);
  a18=(a18*a27);
  a28=(a28+a18);
  a26=(a28*a26);
  a13=(a13+a26);
  a31=(a31*a17);
  a20=(a20*a27);
  a31=(a31+a20);
  a12=(a31*a12);
  a13=(a13+a12);
  a12=(a10*a13);
  a30=(a30+a12);
  a12=3.6910356433321918e-02;
  a8=(a8*a16);
  a21=(a21*a11);
  a8=(a8+a21);
  a24=(a24*a15);
  a8=(a8+a24);
  a24=(a12*a8);
  a30=(a30-a24);
  a1=(a1*a16);
  a4=(a4*a11);
  a1=(a1-a4);
  a19=(a19*a15);
  a1=(a1-a19);
  a19=(a23*a1);
  a30=(a30+a19);
  a0=(a0*a16);
  a28=(a28*a11);
  a0=(a0+a28);
  a31=(a31*a15);
  a0=(a0+a31);
  a31=(a6*a0);
  a30=(a30-a31);
  a31=sin(a7);
  a30=(a30*a31);
  a23=(a23*a3);
  a12=(a12*a22);
  a23=(a23-a12);
  a6=(a6*a13);
  a23=(a23-a6);
  a5=(a5*a8);
  a23=(a23-a5);
  a25=(a25*a1);
  a23=(a23+a25);
  a10=(a10*a0);
  a23=(a23-a10);
  a7=cos(a7);
  a23=(a23*a7);
  a30=(a30+a23);
  a30=(-a30);
  if (res[0]!=0) res[0][3]=a30;
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_02_05_12495939_cost_y_e_hess(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int model_2021_02_05_12495939_cost_y_e_hess_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_02_05_12495939_cost_y_e_hess_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_02_05_12495939_cost_y_e_hess_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int model_2021_02_05_12495939_cost_y_e_hess_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_02_05_12495939_cost_y_e_hess_release(int mem) {
}

CASADI_SYMBOL_EXPORT void model_2021_02_05_12495939_cost_y_e_hess_incref(void) {
}

CASADI_SYMBOL_EXPORT void model_2021_02_05_12495939_cost_y_e_hess_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int model_2021_02_05_12495939_cost_y_e_hess_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int model_2021_02_05_12495939_cost_y_e_hess_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real model_2021_02_05_12495939_cost_y_e_hess_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_02_05_12495939_cost_y_e_hess_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_02_05_12495939_cost_y_e_hess_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_02_05_12495939_cost_y_e_hess_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_02_05_12495939_cost_y_e_hess_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int model_2021_02_05_12495939_cost_y_e_hess_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
