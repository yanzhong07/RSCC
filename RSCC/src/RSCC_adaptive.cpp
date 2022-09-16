//
//  convex_ncwvs.cpp
//  
//
//  Created by yanzhong on 10/11/19.
//
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

//#define ARMA_DONT_PRINT_ERRORS
//#define ARMA_64BIT_WORD 1
#include <RcppArmadillo.h>
#include <stdio.h>
//#include <ctime>


//#include <RcppArmadilloExtensions/sample.h>


using namespace Rcpp;
using namespace arma;

// proximal fused lasso with weight. lambda is diferent for each componenet. size(lambda) = len(y)-1
// [[Rcpp::export]]
arma::vec onedTVw_c(arma::vec y, arma::vec lambda){
    //revise weight
    int sz = lambda.size();
    lambda.resize(sz+1);
    lambda(sz) = 0;
    
    int n = y.n_elem;
    int k = 1;
    int k0 = 1;
    int kmin = 1;
    int kplu = 1;
    double vmin = y(0) - lambda(0);
    double vmax = y(0) + lambda(0);
    double mumin = lambda(0);
    double mumax = -lambda(0);
    arma::vec x(n,1, fill::zeros);
    while(TRUE){ // line 2
        if(k == n){
            x(n-1) = vmin + mumin;
            break;
        }
        
        while(k < n){ // line 3
            if( (y(k+1-1) + mumin) < (vmin - lambda(k))){
                //x(span(k0-1,kmin-1)) = vmin;
                x(span(k0-1,kmin-1)).fill(vmin);
                k = kmin + 1;
                k0 = kmin + 1;
                kplu = kmin + 1;
                kmin = kmin + 1;
                vmin = y(k-1) - lambda(k-1) + lambda(k-2);
                vmax = y(k-1) + lambda(k-1) + lambda(k-2);
                mumin = lambda(k-1);
                mumax = -lambda(k-1);
            }else if( (y(k+1-1) + mumax) > (vmax + lambda(k))){
                //x(span(k0-1,kplu-1)) = vmax;
                x(span(k0-1,kplu-1)).fill(vmax);
                k = kplu + 1;
                k0 = kplu + 1;
                kmin = kplu + 1;
                kplu = kplu + 1;
                vmin = y(k-1) - lambda(k-1) - lambda(k-2);
                vmax = y(k-1) + lambda(k-1) - lambda(k-2);
                mumin = lambda(k-1);
                mumax = -lambda(k-1);
            }else{
                k = k + 1;
                mumin = mumin + y(k-1) -vmin;
                mumax = mumax + y(k-1) -vmax;
                if(mumin >= lambda(k-1)){
                    vmin = vmin + (mumin - lambda(k-1)) / (k-k0+1);
                    mumin = lambda(k-1);
                    kmin = k;
                }
                if(mumax <= -lambda(k-1)){
                    vmax = vmax + (mumax  +lambda(k-1)) / (k-k0+1);
                    mumax = - lambda(k-1);
                    kplu = k;
                }
            }
        }
        
        if(mumin <0){
            //x(span(k0-1,kmin-1)) = vmin;
            x(span(k0-1,kmin-1)).fill(vmin);
            k = kmin + 1;
            k0 = kmin + 1;
            kmin = kmin + 1;
            vmin = y(k-1);
            mumin = lambda(k-1);
            mumax = y(k-1) + lambda(k-1) -vmax;
        }else if(mumax > 0){
            
            //x(span(k0-1,kplu-1)) = vmax;
            x(span(k0-1,kplu-1)).fill(vmax);
            k = kplu + 1;
            k0 = kplu + 1;
            kplu = kplu + 1;
            vmax = y(k-1);
            mumax = -lambda(k-1);
            mumin = y(k-1) - lambda(k-1) - vmin;
        }else{
            x(span(k0-1,n-1)).fill(vmin + mumin/(k - k0 + 1));
            break;
        }
    }
    
    return x;
}


arma::mat grplassoprox_matrix_col_c(arma::mat& X, arma::vec lambda_list){
    int n = X.n_rows;
    int p = X.n_cols;
    arma::mat beta(n,p,fill::zeros);
    arma::vec tmp0(p,fill::zeros);
    arma::vec tmp1 = sqrt(sum(pow(X,2),0)).t();
    arma::vec tmp2 = (tmp1.elem(find(tmp1 > lambda_list)) - lambda_list.elem(find(tmp1 > lambda_list))) / tmp1.elem(find(tmp1 > lambda_list));
    arma::mat Y = X.cols(find(tmp1 > lambda_list));
    beta.cols(find(tmp1 > lambda_list)) =  Y.each_row() % tmp2.t();
    //tmp0.elem(find(tmp1 > lambda_list)) = tmp2;
    //beta = X.each_row() % tmp0.t();
    return beta;
}

// order the index of matrix
arma::mat orderMatrix1(arma::mat& B, arma::mat & sample_order_list){
    int n = B.n_rows;
    int p = B.n_cols;
    arma::mat B_order(n,p, fill::zeros);
    arma::vec tmp(n, fill::zeros);
    for(int i =0; i<p; i++){
        tmp(sort_index(sample_order_list.col(i))) = B.col(i);
        B_order.col(i) = tmp;
    }
    return B_order;
}

// order back the index of matrix
arma::mat orderMatrix2(arma::mat& B_order, arma::mat & sample_order_list){
    int p = B_order.n_cols;
    arma::mat B = B_order;
    arma::vec tmp;
    for(int i =0; i<p; i++){
        tmp = B_order.col(i);
        B.col(i) = tmp(sort_index(sample_order_list.col(i)));
    }
    return B;
}



// proximal gradient for depth first searching for weight edge
// [[Rcpp::export]]
List ProxG_dfs_w(arma::vec& y, arma::mat& X, arma::vec& Xty, arma::mat& sample_order_list, arma::mat& edge_weight_list, arma::vec& var_weight, int& intercept, arma::mat& B_initial, double& lambda_row, double& lambda_col, double& lambda_tau, arma::mat& B_save, double& GIC_out, double& MSE_out, double& var_num, double step_size = 1, double tol = 0.0000001, double MaxIt = 100){
    
    // initialization
    int n = X.n_rows;
    int p = X.n_cols;
    //arma::mat Bnow(n,p,fill::zeros);
    arma::mat Bk = B_initial;
    double ak = (mean(y - sum(X % Bk,1))) * intercept;
    
    // Build overall design matrix for calculating gradient
    //arma::vec Xty = vectorise(X.each_col() % y);
    
    // setup for Fista
    double ak_1 = ak;
    double anow = ak;  // for transformed a
    arma::mat Bnow = Bk;
    arma::mat Bk_1 = Bk; // for transformed b
    double tk = 1;
    double tk_1 = 1;
    
    // step size
    double iLuse = n / ( max(sum(pow(X,2),1)) * (1 + pow(10.0,-3)) ) * step_size; // step_size between 0 to 1.
    
    // setup for the stop rule of iteration
    double PS1 = sum(sqrt(sum(pow(Bk,2),0)) * lambda_col * var_weight);
    double PS2 = 0;
    
    for(int i=0; i<p; i++){
        for(int j=0; j<n-1; j++){
            PS2 = PS2 + fabs(Bk(sample_order_list(j,i),i) - Bk(sample_order_list(j+1,i),i)) * edge_weight_list(j,i);
        }
    }
    PS2 = PS2 * lambda_row;
    //PS2 = PS2 + sum(fabs(Bnow(sample_order_list(span(0,n-2),i),i) - Bnow(sample_order_list(span(1,n-1),i),i) )) * lambda_row;
    double PS3 = sum(sum(pow(Bk,2))) * lambda_tau / 2 * lambda_col ;
    double SST = sum(pow(y - ak - sum(X % Bk,1),2)) / 2.0 / n; // notice that our loss function have 1/n
    double TS0 = SST + PS1 + PS2 + PS3;
    double TS1 = 0;
    double diff = 1;
    //arma::mat Bpre = Bnow;
    arma::mat Btmp = Bnow;
    double this_tol  = 2 * tol;
    double this_iter = 0;
    arma::vec gradient1;
    arma::mat tmp_g(n,p,fill::zeros);
    arma::mat tmp_g_col;
    
    // elastic net adjust
    double ilambda_col2 = 1.0 / (1+2 * lambda_tau / 2 * lambda_col * iLuse);
    
    while(fabs(diff) > tol && this_iter < MaxIt){
        //calculate gradient
        gradient1 = vectorise(Bnow) + iLuse * (Xty - vectorise(X) * anow * intercept - vectorise(X.each_col() % sum(X % Bnow,1))) / n;
        
        // proximal of fused lasso
        tmp_g = mat(gradient1);
        tmp_g.reshape(n,p);
        tmp_g = orderMatrix1(tmp_g, sample_order_list);
        
        for(uint i=0; i<p; i++){
            tmp_g.col(i);
            Btmp.col(i) = onedTVw_c(vectorise(tmp_g.col(i)) * ilambda_col2, edge_weight_list.col(i) * lambda_row * iLuse * ilambda_col2);
        }
        
        Btmp = orderMatrix2(Btmp, sample_order_list);
        Bk = grplassoprox_matrix_col_c(Btmp, lambda_col * iLuse * var_weight * ilambda_col2);
        
        tk = (sqrt(1.0+tk_1 * tk_1 * 4) + 1)/2;
        Bnow = Bk + (tk_1 - 1)/ tk * (Bk - Bk_1);
        ak = mean(y-  sum(X % Bk,1)) * intercept;
        anow = ak;
        
        PS1 = sum(sqrt(sum(pow(Bk,2),0)) * lambda_col * var_weight);
        PS2 = 0;
        
        for(int i=0; i<p; i++){
            for(int j=0; j<n-1; j++){
                PS2 = PS2 + fabs(Bk(sample_order_list(j,i),i) - Bk(sample_order_list(j+1,i),i)) * edge_weight_list(j,i);
            }
        }
        PS2 = PS2 * lambda_row;
        PS3 = sum(sum(pow(Bk,2))) * lambda_tau / 2.0 * lambda_col;
        SST = sum(pow(y - ak - sum(X % Bk,1),2.0)) / 2.0 / n;
        
        TS1 = SST + PS1 + PS2 + PS3;
        diff = TS0 - TS1;
        
        this_tol  = sqrt(mean(mean( pow(Bk - Bk_1,2))));
        this_iter = this_iter + 1;
        TS0 = TS1;
        tk_1 = tk;
        Bk_1 = Bk;
        ak_1 = ak;
        //printf("%f,%f,%f\n",this_iter, TS1, diff);
    }
    
    double df1 = sum(sum(abs(Bk),0) != 0);
    arma::vec dfcol(p,fill::zeros);
    for(int i=0; i<p; i++){
        for(int j=0; j<n-1; j++){
            dfcol(i) = dfcol(i) + ((Bk(sample_order_list(j,i),i) - Bk(sample_order_list(j+1,i),i)) != 0);
        }
    }
    
    SST = sum(pow(y - ak - sum(X % Bk,1),2)) / 2.0 / n;
    //double df_adjust = 1;
    //double df = (df1 *df_adjust + sum(dfcol)) / (p*df_adjust + (n-1)*p) * n*p;
    double df = df1 + sum(dfcol) + 1 * intercept;
    double df_constant = log(n) * log(log(n));  // GIC use this
    double GIC = log(2 * SST) + df * df_constant / n;
    double MSE = log(2 * SST);
    //print(paste0("GIC: ",as.character(GIC),", diff:" ,as.character(this_tol),", SST:", SST, ", df:", df))
    //printf("GIC:%f, df:%f\n",GIC, df);
    B_save = Bk;
    GIC_out = GIC;
    MSE_out = MSE;
    var_num = df1 + sum(dfcol);
    List list1;
    list1.push_back(Bnow);
    list1.push_back(sqrt(SST * 2)); //mse
    list1.push_back(GIC);
    list1.push_back(lambda_row);
    list1.push_back(lambda_col);
    list1.push_back(lambda_tau);
    list1.push_back(df1+sum(dfcol)+ 1 * intercept);
    list1.push_back(ak);
    
    return list1;
}




// Use GIC to select best model
// ini_method ==1 use lasso with gic selection
// ini_method ==2 use lasso with mse selection
// [[Rcpp::export]]
List GIC_dpsw_c(arma::vec& y, arma::mat& X, arma::mat & sample_order_list, arma::mat & edge_weight_list, arma::vec& var_weight, int& intercept, arma::vec& lambda_row_list, arma::vec& lambda_col_list, arma::vec& lambda_tau_list, double step_size = 1, double tol = 0.0000001, double MaxIt = 100){  // adap_begin, use for the initial of adaptive, may need to revise.
    int m1 = lambda_row_list.n_elem;
    int m2 = lambda_col_list.n_elem;
    int m3 = lambda_tau_list.n_elem;
    double nowValue = datum::inf;
    double thisGIC = 0;
    double thisMSE = 0;
    double var_num = 0;
    List minmodel;
    List tmpmodel;
    int ibest = 1;
    int jbest = 1;
    int qbest = 1;
    int n = y.n_elem;
    int p = X.n_cols;
    arma::vec Xty = vectorise(X.each_col() % y);
    //clock_t t1,t2;
    
    for(int q = 0; q < m3; q++){
        arma::mat B_initial(n,p,fill::zeros);
        arma::mat B_save1(n,p,fill::zeros);
        arma::mat B_save2(n,p,fill::zeros);
        for(int i = 0 ; i < m1  ; i++){
            for(int j = 0 ; j < m2  ; j++){
                if(j ==0){
                    B_initial = B_save1;
                }

                //t1 = clock();
                tmpmodel = ProxG_dfs_w(y, X, Xty, sample_order_list, edge_weight_list, var_weight, intercept, B_initial, lambda_row_list(i), lambda_col_list(j), lambda_tau_list(q), B_save2, thisGIC, thisMSE, var_num, step_size, tol, MaxIt);
                //printf("%f,%f\n", thisGIC,var_num);                
                //t2 = clock();
                //printf("time elapse: %f seconds=>", (double)(t2 - t1)); 
                
                
                if(nowValue > thisGIC){
                    minmodel = tmpmodel;
                    nowValue = thisGIC;
                    ibest = i;
                    jbest = j;
                    qbest = q;
                }
                
                B_initial = B_save2;
                if(j ==0){
                    B_save1 = B_save2;
                }
                //printf("%d%d%d=>", q,i,j);
                
                //printf("(%f,%f,%f): GIC: %f, Var: %f",lambda_row_list(i), lambda_col_list(j), lambda_tau_list(q), thisGIC, var_num);
            }
            printf("%d%d=>", q,i);
        }
        printf("\nBestnow(%f,%f,%f): GIC: %f\n",lambda_row_list(ibest), lambda_col_list(jbest), lambda_tau_list(qbest), nowValue);
    }
    printf("Finish.");
    return minmodel;
}
