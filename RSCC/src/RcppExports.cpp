// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// onedTVw_c
arma::vec onedTVw_c(arma::vec y, arma::vec lambda);
RcppExport SEXP _RSCC_onedTVw_c(SEXP ySEXP, SEXP lambdaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    rcpp_result_gen = Rcpp::wrap(onedTVw_c(y, lambda));
    return rcpp_result_gen;
END_RCPP
}
// ProxG_dfs_w
List ProxG_dfs_w(arma::vec& y, arma::mat& X, arma::vec& Xty, arma::mat& sample_order_list, arma::mat& edge_weight_list, arma::vec& var_weight, int& intercept, arma::mat& B_initial, double& lambda_row, double& lambda_col, double& lambda_tau, arma::mat& B_save, double& GIC_out, double& MSE_out, double& var_num, double step_size, double tol, double MaxIt);
RcppExport SEXP _RSCC_ProxG_dfs_w(SEXP ySEXP, SEXP XSEXP, SEXP XtySEXP, SEXP sample_order_listSEXP, SEXP edge_weight_listSEXP, SEXP var_weightSEXP, SEXP interceptSEXP, SEXP B_initialSEXP, SEXP lambda_rowSEXP, SEXP lambda_colSEXP, SEXP lambda_tauSEXP, SEXP B_saveSEXP, SEXP GIC_outSEXP, SEXP MSE_outSEXP, SEXP var_numSEXP, SEXP step_sizeSEXP, SEXP tolSEXP, SEXP MaxItSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type Xty(XtySEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type sample_order_list(sample_order_listSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type edge_weight_list(edge_weight_listSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type var_weight(var_weightSEXP);
    Rcpp::traits::input_parameter< int& >::type intercept(interceptSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type B_initial(B_initialSEXP);
    Rcpp::traits::input_parameter< double& >::type lambda_row(lambda_rowSEXP);
    Rcpp::traits::input_parameter< double& >::type lambda_col(lambda_colSEXP);
    Rcpp::traits::input_parameter< double& >::type lambda_tau(lambda_tauSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type B_save(B_saveSEXP);
    Rcpp::traits::input_parameter< double& >::type GIC_out(GIC_outSEXP);
    Rcpp::traits::input_parameter< double& >::type MSE_out(MSE_outSEXP);
    Rcpp::traits::input_parameter< double& >::type var_num(var_numSEXP);
    Rcpp::traits::input_parameter< double >::type step_size(step_sizeSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< double >::type MaxIt(MaxItSEXP);
    rcpp_result_gen = Rcpp::wrap(ProxG_dfs_w(y, X, Xty, sample_order_list, edge_weight_list, var_weight, intercept, B_initial, lambda_row, lambda_col, lambda_tau, B_save, GIC_out, MSE_out, var_num, step_size, tol, MaxIt));
    return rcpp_result_gen;
END_RCPP
}
// GIC_dpsw_c
List GIC_dpsw_c(arma::vec& y, arma::mat& X, arma::mat& sample_order_list, arma::mat& edge_weight_list, arma::vec& var_weight, int& intercept, arma::vec& lambda_row_list, arma::vec& lambda_col_list, arma::vec& lambda_tau_list, double step_size, double tol, double MaxIt);
RcppExport SEXP _RSCC_GIC_dpsw_c(SEXP ySEXP, SEXP XSEXP, SEXP sample_order_listSEXP, SEXP edge_weight_listSEXP, SEXP var_weightSEXP, SEXP interceptSEXP, SEXP lambda_row_listSEXP, SEXP lambda_col_listSEXP, SEXP lambda_tau_listSEXP, SEXP step_sizeSEXP, SEXP tolSEXP, SEXP MaxItSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type sample_order_list(sample_order_listSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type edge_weight_list(edge_weight_listSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type var_weight(var_weightSEXP);
    Rcpp::traits::input_parameter< int& >::type intercept(interceptSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type lambda_row_list(lambda_row_listSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type lambda_col_list(lambda_col_listSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type lambda_tau_list(lambda_tau_listSEXP);
    Rcpp::traits::input_parameter< double >::type step_size(step_sizeSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< double >::type MaxIt(MaxItSEXP);
    rcpp_result_gen = Rcpp::wrap(GIC_dpsw_c(y, X, sample_order_list, edge_weight_list, var_weight, intercept, lambda_row_list, lambda_col_list, lambda_tau_list, step_size, tol, MaxIt));
    return rcpp_result_gen;
END_RCPP
}
// onedTV_c
arma::vec onedTV_c(arma::vec y, double lambda);
RcppExport SEXP _RSCC_onedTV_c(SEXP ySEXP, SEXP lambdaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    rcpp_result_gen = Rcpp::wrap(onedTV_c(y, lambda));
    return rcpp_result_gen;
END_RCPP
}
// ProxG_dfs
List ProxG_dfs(arma::vec& y, arma::mat& X, arma::vec& Xty, arma::vec& sample_order_list0, arma::vec& penalty_factor, int intercept, arma::mat& B_initial, double& lambda_row, double& lambda_col, double& lambda_tau, arma::mat& B_save, double& GIC_out, double& MSE_out, arma::vec& var_num, double step_size, double tol, double MaxIt);
RcppExport SEXP _RSCC_ProxG_dfs(SEXP ySEXP, SEXP XSEXP, SEXP XtySEXP, SEXP sample_order_list0SEXP, SEXP penalty_factorSEXP, SEXP interceptSEXP, SEXP B_initialSEXP, SEXP lambda_rowSEXP, SEXP lambda_colSEXP, SEXP lambda_tauSEXP, SEXP B_saveSEXP, SEXP GIC_outSEXP, SEXP MSE_outSEXP, SEXP var_numSEXP, SEXP step_sizeSEXP, SEXP tolSEXP, SEXP MaxItSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type Xty(XtySEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type sample_order_list0(sample_order_list0SEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type penalty_factor(penalty_factorSEXP);
    Rcpp::traits::input_parameter< int >::type intercept(interceptSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type B_initial(B_initialSEXP);
    Rcpp::traits::input_parameter< double& >::type lambda_row(lambda_rowSEXP);
    Rcpp::traits::input_parameter< double& >::type lambda_col(lambda_colSEXP);
    Rcpp::traits::input_parameter< double& >::type lambda_tau(lambda_tauSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type B_save(B_saveSEXP);
    Rcpp::traits::input_parameter< double& >::type GIC_out(GIC_outSEXP);
    Rcpp::traits::input_parameter< double& >::type MSE_out(MSE_outSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type var_num(var_numSEXP);
    Rcpp::traits::input_parameter< double >::type step_size(step_sizeSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< double >::type MaxIt(MaxItSEXP);
    rcpp_result_gen = Rcpp::wrap(ProxG_dfs(y, X, Xty, sample_order_list0, penalty_factor, intercept, B_initial, lambda_row, lambda_col, lambda_tau, B_save, GIC_out, MSE_out, var_num, step_size, tol, MaxIt));
    return rcpp_result_gen;
END_RCPP
}
// MSE_dfs_c
List MSE_dfs_c(arma::vec& y, arma::mat& X, arma::vec& sample_order_list0, arma::vec& penalty_factor, int intercept, arma::vec& lambda_row_list, arma::vec& lambda_col_list, arma::vec& lambda_tau_list, int min_var, double step_size, double tol, double MaxIt);
RcppExport SEXP _RSCC_MSE_dfs_c(SEXP ySEXP, SEXP XSEXP, SEXP sample_order_list0SEXP, SEXP penalty_factorSEXP, SEXP interceptSEXP, SEXP lambda_row_listSEXP, SEXP lambda_col_listSEXP, SEXP lambda_tau_listSEXP, SEXP min_varSEXP, SEXP step_sizeSEXP, SEXP tolSEXP, SEXP MaxItSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type sample_order_list0(sample_order_list0SEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type penalty_factor(penalty_factorSEXP);
    Rcpp::traits::input_parameter< int >::type intercept(interceptSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type lambda_row_list(lambda_row_listSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type lambda_col_list(lambda_col_listSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type lambda_tau_list(lambda_tau_listSEXP);
    Rcpp::traits::input_parameter< int >::type min_var(min_varSEXP);
    Rcpp::traits::input_parameter< double >::type step_size(step_sizeSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< double >::type MaxIt(MaxItSEXP);
    rcpp_result_gen = Rcpp::wrap(MSE_dfs_c(y, X, sample_order_list0, penalty_factor, intercept, lambda_row_list, lambda_col_list, lambda_tau_list, min_var, step_size, tol, MaxIt));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_RSCC_onedTVw_c", (DL_FUNC) &_RSCC_onedTVw_c, 2},
    {"_RSCC_ProxG_dfs_w", (DL_FUNC) &_RSCC_ProxG_dfs_w, 18},
    {"_RSCC_GIC_dpsw_c", (DL_FUNC) &_RSCC_GIC_dpsw_c, 12},
    {"_RSCC_onedTV_c", (DL_FUNC) &_RSCC_onedTV_c, 2},
    {"_RSCC_ProxG_dfs", (DL_FUNC) &_RSCC_ProxG_dfs, 17},
    {"_RSCC_MSE_dfs_c", (DL_FUNC) &_RSCC_MSE_dfs_c, 12},
    {NULL, NULL, 0}
};

RcppExport void R_init_RSCC(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
