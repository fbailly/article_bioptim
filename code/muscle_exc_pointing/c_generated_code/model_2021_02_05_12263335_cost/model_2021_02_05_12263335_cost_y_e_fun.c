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
  #define CASADI_PREFIX(ID) model_2021_02_05_12263335_cost_y_e_fun_ ## ID
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

static const casadi_int casadi_s0[14] = {10, 1, 0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
static const casadi_int casadi_s1[3] = {0, 0, 0};
static const casadi_int casadi_s2[17] = {13, 1, 0, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

/* model_2021_02_05_12263335_cost_y_e_fun:(i0[10],i1[],i2[])->(o0[13]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a3, a4, a5, a6, a7, a8, a9;
  a0=arg[0]? arg[0][0] : 0;
  if (res[0]!=0) res[0][0]=a0;
  a1=arg[0]? arg[0][1] : 0;
  if (res[0]!=0) res[0][1]=a1;
  a2=arg[0]? arg[0][2] : 0;
  if (res[0]!=0) res[0][2]=a2;
  a2=arg[0]? arg[0][3] : 0;
  if (res[0]!=0) res[0][3]=a2;
  a2=arg[0]? arg[0][4] : 0;
  if (res[0]!=0) res[0][4]=a2;
  a2=arg[0]? arg[0][5] : 0;
  if (res[0]!=0) res[0][5]=a2;
  a2=arg[0]? arg[0][6] : 0;
  if (res[0]!=0) res[0][6]=a2;
  a2=arg[0]? arg[0][7] : 0;
  if (res[0]!=0) res[0][7]=a2;
  a2=arg[0]? arg[0][8] : 0;
  if (res[0]!=0) res[0][8]=a2;
  a2=arg[0]? arg[0][9] : 0;
  if (res[0]!=0) res[0][9]=a2;
  a2=-1.4999999999999999e-01;
  a3=-1.7545000000000002e-02;
  a4=1.6671699644008754e-02;
  a5=9.9750107761097473e-01;
  a6=cos(a0);
  a7=(a5*a6);
  a8=3.9020807762349584e-02;
  a0=sin(a0);
  a9=(a8*a0);
  a7=(a7+a9);
  a9=(a4*a7);
  a10=-2.8994080392798782e-01;
  a8=(a8*a6);
  a5=(a5*a0);
  a8=(a8-a5);
  a5=(a10*a8);
  a9=(a9+a5);
  a5=7.8368601202986399e-04;
  a9=(a9+a5);
  a3=(a3+a9);
  a9=-1.3071942448447652e-01;
  a5=7.7320652313482108e-01;
  a11=cos(a1);
  a12=(a5*a11);
  a13=-6.2511643845589493e-01;
  a1=sin(a1);
  a14=(a13*a1);
  a12=(a12+a14);
  a14=(a12*a7);
  a15=6.2502056305245079e-01;
  a16=(a15*a11);
  a17=7.7965829948778420e-01;
  a18=(a17*a1);
  a16=(a16+a18);
  a18=(a16*a8);
  a14=(a14+a18);
  a18=-5.8898019716436364e-02;
  a19=-1.0724256777678894e-01;
  a20=(a19*a11);
  a21=3.6910356433321918e-02;
  a22=(a21*a1);
  a20=(a20+a22);
  a22=(a18*a20);
  a14=(a14+a22);
  a14=(a9*a14);
  a22=-1.7676989936934065e-01;
  a13=(a13*a11);
  a5=(a5*a1);
  a13=(a13-a5);
  a5=(a13*a7);
  a17=(a17*a11);
  a15=(a15*a1);
  a17=(a17-a15);
  a15=(a17*a8);
  a5=(a5+a15);
  a21=(a21*a11);
  a19=(a19*a1);
  a21=(a21-a19);
  a18=(a18*a21);
  a5=(a5+a18);
  a5=(a22*a5);
  a14=(a14+a5);
  a5=-8.0520021982363552e-03;
  a18=1.0668228978597614e-01;
  a7=(a18*a7);
  a19=3.8499763654015404e-02;
  a8=(a19*a8);
  a7=(a7+a8);
  a8=-5.8517980910776868e-02;
  a7=(a7+a8);
  a7=(a5*a7);
  a14=(a14+a7);
  a3=(a3+a14);
  a2=(a2-a3);
  if (res[0]!=0) res[0][10]=a2;
  a2=8.0000000000000002e-02;
  a3=-7.0000000000000001e-03;
  a14=-3.8952964437603196e-02;
  a7=(a14*a6);
  a8=9.9923839826218319e-01;
  a1=(a8*a0);
  a7=(a7+a1);
  a1=(a4*a7);
  a8=(a8*a6);
  a14=(a14*a0);
  a8=(a8-a14);
  a14=(a10*a8);
  a1=(a1+a14);
  a14=-3.0603368800321084e-05;
  a1=(a1+a14);
  a3=(a3+a1);
  a1=(a12*a7);
  a14=(a16*a8);
  a1=(a1+a14);
  a14=2.2999999889266845e-03;
  a11=(a14*a20);
  a1=(a1+a11);
  a1=(a9*a1);
  a11=(a13*a7);
  a15=(a17*a8);
  a11=(a11+a15);
  a14=(a14*a21);
  a11=(a11+a14);
  a11=(a22*a11);
  a1=(a1+a11);
  a7=(a18*a7);
  a8=(a19*a8);
  a7=(a7+a8);
  a8=2.2851592650277005e-03;
  a7=(a7+a8);
  a7=(a5*a7);
  a1=(a1+a7);
  a3=(a3+a1);
  a2=(a2-a3);
  if (res[0]!=0) res[0][11]=a2;
  a2=1.2000000000000000e-01;
  a3=1.7000000000000001e-01;
  a1=5.8942910739687680e-02;
  a6=(a1*a6);
  a4=(a4*a6);
  a1=(a1*a0);
  a10=(a10*a1);
  a4=(a4-a10);
  a10=-1.3282678503995692e-02;
  a4=(a4+a10);
  a3=(a3+a4);
  a12=(a12*a6);
  a16=(a16*a1);
  a12=(a12-a16);
  a16=9.9826135519388559e-01;
  a20=(a16*a20);
  a12=(a12+a20);
  a9=(a9*a12);
  a13=(a13*a6);
  a17=(a17*a1);
  a13=(a13-a17);
  a16=(a16*a21);
  a13=(a13+a16);
  a22=(a22*a13);
  a9=(a9+a22);
  a18=(a18*a6);
  a19=(a19*a1);
  a18=(a18-a19);
  a19=9.9182008509702291e-01;
  a18=(a18+a19);
  a5=(a5*a18);
  a9=(a9+a5);
  a3=(a3+a9);
  a2=(a2-a3);
  if (res[0]!=0) res[0][12]=a2;
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_02_05_12263335_cost_y_e_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int model_2021_02_05_12263335_cost_y_e_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_02_05_12263335_cost_y_e_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_02_05_12263335_cost_y_e_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int model_2021_02_05_12263335_cost_y_e_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_02_05_12263335_cost_y_e_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void model_2021_02_05_12263335_cost_y_e_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void model_2021_02_05_12263335_cost_y_e_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int model_2021_02_05_12263335_cost_y_e_fun_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int model_2021_02_05_12263335_cost_y_e_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real model_2021_02_05_12263335_cost_y_e_fun_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_02_05_12263335_cost_y_e_fun_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_02_05_12263335_cost_y_e_fun_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_02_05_12263335_cost_y_e_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_02_05_12263335_cost_y_e_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int model_2021_02_05_12263335_cost_y_e_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
