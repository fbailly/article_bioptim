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
  #define CASADI_PREFIX(ID) model_2021_02_05_12254308_cost_y_e_fun_jac_ut_xt_ ## ID
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

static const casadi_int casadi_s0[14] = {10, 1, 0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
static const casadi_int casadi_s1[3] = {0, 0, 0};
static const casadi_int casadi_s2[17] = {13, 1, 0, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
static const casadi_int casadi_s3[32] = {10, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 0, 1, 0, 1};

/* model_2021_02_05_12254308_cost_y_e_fun_jac_ut_xt:(i0[10],i1[],i2[])->(o0[13],o1[10x13,16nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a4, a5, a6, a7, a8, a9;
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
  a9=sin(a0);
  a10=(a8*a9);
  a7=(a7+a10);
  a10=(a4*a7);
  a11=-2.8994080392798782e-01;
  a12=(a8*a6);
  a13=(a5*a9);
  a12=(a12-a13);
  a13=(a11*a12);
  a10=(a10+a13);
  a13=7.8368601202986399e-04;
  a10=(a10+a13);
  a3=(a3+a10);
  a10=-1.3071942448447652e-01;
  a13=7.7320652313482108e-01;
  a14=cos(a1);
  a15=(a13*a14);
  a16=-6.2511643845589493e-01;
  a17=sin(a1);
  a18=(a16*a17);
  a15=(a15+a18);
  a18=(a15*a7);
  a19=6.2502056305245079e-01;
  a20=(a19*a14);
  a21=7.7965829948778420e-01;
  a22=(a21*a17);
  a20=(a20+a22);
  a22=(a20*a12);
  a18=(a18+a22);
  a22=-5.8898019716436364e-02;
  a23=-1.0724256777678894e-01;
  a24=(a23*a14);
  a25=3.6910356433321918e-02;
  a26=(a25*a17);
  a24=(a24+a26);
  a26=(a22*a24);
  a18=(a18+a26);
  a18=(a10*a18);
  a26=-1.7676989936934065e-01;
  a27=(a16*a14);
  a28=(a13*a17);
  a27=(a27-a28);
  a28=(a27*a7);
  a29=(a21*a14);
  a30=(a19*a17);
  a29=(a29-a30);
  a30=(a29*a12);
  a28=(a28+a30);
  a14=(a25*a14);
  a17=(a23*a17);
  a14=(a14-a17);
  a17=(a22*a14);
  a28=(a28+a17);
  a28=(a26*a28);
  a18=(a18+a28);
  a28=-8.0520021982363552e-03;
  a17=1.0668228978597614e-01;
  a30=(a17*a7);
  a31=3.8499763654015404e-02;
  a32=(a31*a12);
  a30=(a30+a32);
  a32=-5.8517980910776868e-02;
  a30=(a30+a32);
  a30=(a28*a30);
  a18=(a18+a30);
  a3=(a3+a18);
  a2=(a2-a3);
  if (res[0]!=0) res[0][10]=a2;
  a2=8.0000000000000002e-02;
  a3=-7.0000000000000001e-03;
  a18=-3.8952964437603196e-02;
  a30=(a18*a6);
  a32=9.9923839826218319e-01;
  a33=(a32*a9);
  a30=(a30+a33);
  a33=(a4*a30);
  a34=(a32*a6);
  a35=(a18*a9);
  a34=(a34-a35);
  a35=(a11*a34);
  a33=(a33+a35);
  a35=-3.0603368800321084e-05;
  a33=(a33+a35);
  a3=(a3+a33);
  a33=(a15*a30);
  a35=(a20*a34);
  a33=(a33+a35);
  a35=2.2999999889266845e-03;
  a36=(a35*a24);
  a33=(a33+a36);
  a33=(a10*a33);
  a36=(a27*a30);
  a37=(a29*a34);
  a36=(a36+a37);
  a37=(a35*a14);
  a36=(a36+a37);
  a36=(a26*a36);
  a33=(a33+a36);
  a36=(a17*a30);
  a37=(a31*a34);
  a36=(a36+a37);
  a37=2.2851592650277005e-03;
  a36=(a36+a37);
  a36=(a28*a36);
  a33=(a33+a36);
  a3=(a3+a33);
  a2=(a2-a3);
  if (res[0]!=0) res[0][11]=a2;
  a2=1.2000000000000000e-01;
  a3=1.7000000000000001e-01;
  a33=5.8942910739687680e-02;
  a6=(a33*a6);
  a36=(a4*a6);
  a9=(a33*a9);
  a37=(a11*a9);
  a36=(a36-a37);
  a37=-1.3282678503995692e-02;
  a36=(a36+a37);
  a3=(a3+a36);
  a36=(a15*a6);
  a37=(a20*a9);
  a36=(a36-a37);
  a37=9.9826135519388559e-01;
  a24=(a37*a24);
  a36=(a36+a24);
  a36=(a10*a36);
  a24=(a27*a6);
  a38=(a29*a9);
  a24=(a24-a38);
  a14=(a37*a14);
  a24=(a24+a14);
  a24=(a26*a24);
  a36=(a36+a24);
  a24=(a17*a6);
  a14=(a31*a9);
  a24=(a24-a14);
  a14=9.9182008509702291e-01;
  a24=(a24+a14);
  a24=(a28*a24);
  a36=(a36+a24);
  a3=(a3+a36);
  a2=(a2-a3);
  if (res[0]!=0) res[0][12]=a2;
  a2=1.;
  if (res[1]!=0) res[1][0]=a2;
  if (res[1]!=0) res[1][1]=a2;
  if (res[1]!=0) res[1][2]=a2;
  if (res[1]!=0) res[1][3]=a2;
  if (res[1]!=0) res[1][4]=a2;
  if (res[1]!=0) res[1][5]=a2;
  if (res[1]!=0) res[1][6]=a2;
  if (res[1]!=0) res[1][7]=a2;
  if (res[1]!=0) res[1][8]=a2;
  if (res[1]!=0) res[1][9]=a2;
  a2=cos(a0);
  a3=(a8*a2);
  a0=sin(a0);
  a36=(a5*a0);
  a3=(a3-a36);
  a36=(a4*a3);
  a8=(a8*a0);
  a5=(a5*a2);
  a8=(a8+a5);
  a5=(a11*a8);
  a36=(a36-a5);
  a5=(a15*a3);
  a24=(a20*a8);
  a5=(a5-a24);
  a5=(a10*a5);
  a24=(a27*a3);
  a14=(a29*a8);
  a24=(a24-a14);
  a24=(a26*a24);
  a5=(a5+a24);
  a3=(a17*a3);
  a8=(a31*a8);
  a3=(a3-a8);
  a3=(a28*a3);
  a5=(a5+a3);
  a36=(a36+a5);
  a36=(-a36);
  if (res[1]!=0) res[1][10]=a36;
  a36=cos(a1);
  a5=(a16*a36);
  a1=sin(a1);
  a3=(a13*a1);
  a5=(a5-a3);
  a3=(a7*a5);
  a8=(a21*a36);
  a24=(a19*a1);
  a8=(a8-a24);
  a24=(a12*a8);
  a3=(a3+a24);
  a24=(a25*a36);
  a14=(a23*a1);
  a24=(a24-a14);
  a14=(a22*a24);
  a3=(a3+a14);
  a3=(a10*a3);
  a16=(a16*a1);
  a13=(a13*a36);
  a16=(a16+a13);
  a7=(a7*a16);
  a21=(a21*a1);
  a19=(a19*a36);
  a21=(a21+a19);
  a12=(a12*a21);
  a7=(a7+a12);
  a25=(a25*a1);
  a23=(a23*a36);
  a25=(a25+a23);
  a22=(a22*a25);
  a7=(a7+a22);
  a7=(a26*a7);
  a3=(a3-a7);
  a3=(-a3);
  if (res[1]!=0) res[1][11]=a3;
  a3=(a32*a2);
  a7=(a18*a0);
  a3=(a3-a7);
  a7=(a4*a3);
  a32=(a32*a0);
  a18=(a18*a2);
  a32=(a32+a18);
  a18=(a11*a32);
  a7=(a7-a18);
  a18=(a15*a3);
  a22=(a20*a32);
  a18=(a18-a22);
  a18=(a10*a18);
  a22=(a27*a3);
  a23=(a29*a32);
  a22=(a22-a23);
  a22=(a26*a22);
  a18=(a18+a22);
  a3=(a17*a3);
  a32=(a31*a32);
  a3=(a3-a32);
  a3=(a28*a3);
  a18=(a18+a3);
  a7=(a7+a18);
  a7=(-a7);
  if (res[1]!=0) res[1][12]=a7;
  a7=(a30*a5);
  a18=(a34*a8);
  a7=(a7+a18);
  a18=(a35*a24);
  a7=(a7+a18);
  a7=(a10*a7);
  a30=(a30*a16);
  a34=(a34*a21);
  a30=(a30+a34);
  a35=(a35*a25);
  a30=(a30+a35);
  a30=(a26*a30);
  a7=(a7-a30);
  a7=(-a7);
  if (res[1]!=0) res[1][13]=a7;
  a0=(a33*a0);
  a4=(a4*a0);
  a33=(a33*a2);
  a11=(a11*a33);
  a4=(a4+a11);
  a15=(a15*a0);
  a20=(a20*a33);
  a15=(a15+a20);
  a15=(a10*a15);
  a27=(a27*a0);
  a29=(a29*a33);
  a27=(a27+a29);
  a27=(a26*a27);
  a15=(a15+a27);
  a17=(a17*a0);
  a31=(a31*a33);
  a17=(a17+a31);
  a28=(a28*a17);
  a15=(a15+a28);
  a4=(a4+a15);
  if (res[1]!=0) res[1][14]=a4;
  a5=(a6*a5);
  a8=(a9*a8);
  a5=(a5-a8);
  a24=(a37*a24);
  a5=(a5+a24);
  a10=(a10*a5);
  a9=(a9*a21);
  a6=(a6*a16);
  a9=(a9-a6);
  a37=(a37*a25);
  a9=(a9-a37);
  a26=(a26*a9);
  a10=(a10+a26);
  a10=(-a10);
  if (res[1]!=0) res[1][15]=a10;
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_02_05_12254308_cost_y_e_fun_jac_ut_xt(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int model_2021_02_05_12254308_cost_y_e_fun_jac_ut_xt_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_02_05_12254308_cost_y_e_fun_jac_ut_xt_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_02_05_12254308_cost_y_e_fun_jac_ut_xt_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int model_2021_02_05_12254308_cost_y_e_fun_jac_ut_xt_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_02_05_12254308_cost_y_e_fun_jac_ut_xt_release(int mem) {
}

CASADI_SYMBOL_EXPORT void model_2021_02_05_12254308_cost_y_e_fun_jac_ut_xt_incref(void) {
}

CASADI_SYMBOL_EXPORT void model_2021_02_05_12254308_cost_y_e_fun_jac_ut_xt_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int model_2021_02_05_12254308_cost_y_e_fun_jac_ut_xt_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int model_2021_02_05_12254308_cost_y_e_fun_jac_ut_xt_n_out(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_real model_2021_02_05_12254308_cost_y_e_fun_jac_ut_xt_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_02_05_12254308_cost_y_e_fun_jac_ut_xt_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_02_05_12254308_cost_y_e_fun_jac_ut_xt_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_02_05_12254308_cost_y_e_fun_jac_ut_xt_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_02_05_12254308_cost_y_e_fun_jac_ut_xt_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    case 1: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int model_2021_02_05_12254308_cost_y_e_fun_jac_ut_xt_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
