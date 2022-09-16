#' RSCC with model selection by GIC.
#' 
#' @param y The response variable for n observations.
#' @param X An n*p matrix for n observations with p covariates. (Do not include intercept)
#' @param Graph0 A list that contains the information for graph. Graph0$graph is a igraph object that contains the whole graph. Graph0$edge is a three-column matrix includes edge indexes in the first two columns and the distance between nodes of edges in the third column. For unweighted graph, set the third column of Graph0$edge to all 1s. An example of this object is the output of RSCC::buildKNN(). The input graph should be connected.
#' @param penalty_factor A vector of length p that indicates the variables that are pre-included into the model. 0 means that the vairable must be included in the model.
#' @param intercept include fixed intercept or not.
#' @param lambda_row_list A list of lambda_row for clustering.
#' @param lambda_col_list A list of lambda_col for variable selection.
#' @param lambda_row_list_ini,lambda_col_list_ini Lists of parameters for model initialization.
#' @param min_var_ini An integer requires that the number of variables remained by model initialization should not below this value.
#' @param step_size A tuning parameter to adjust step_size. It should be in (0,1].
#' @param tol Tolerance for convergence.
#' @param MaxIt Max number of iterations.
#' @return Return a list with the estimator for B, mse, gic, the values of three parameters, the estimated degree of freedom, and the fixed intercept term a.
#'   
RSCC_gic = function(y, X, Graph0, penalty_factor = NULL, intercept = T, 
                    lambda_row_list = exp(seq(0,-8,length.out = 10)), lambda_col_list= exp(seq(0,-8,length.out = 10)),
                    lambda_row_list_ini = exp(seq(0,-7,length.out = 10)), lambda_col_list_ini= exp(seq(-3,-8,length.out = 10)), min_var_ini = dim(X)[2]/2, step_size = 1, tol = 10^(-7), MaxIt = 1000){
  cat("000\n")
  
  ## initialization
  n <- dim(X)[1]
  p <- dim(X)[2]
  if(intercept){
    intercept = 1
  }else{
    intercept = 0
  }
  
  stopifnot(n == length(y))
  lambda_col_list_ini = sort(lambda_col_list_ini, decreasing = T)
  lambda_row_list_ini = sort(lambda_row_list_ini, decreasing = T)
  lambda_col_list = sort(lambda_col_list, decreasing = T)
  lambda_row_list = sort(lambda_row_list, decreasing = T)
  stopifnot(is_connected(Graph0$graph))
  ## first part, build initial model.
  g_mst <- mst(Graph0$graph, weights = Graph0$edge[,3])
  sample_order = as.vector(dfs(g_mst, root = 1)$order)
  sample_order_list_ini = sample_order -1 ## this -1 is to adjust the index in C++
  
  if(is.null(penalty_factor)){
    penalty_factor = rep(1,p)
  }
  stopifnot(p == length(penalty_factor), sum(penalty_factor %in% c(0,1)) == p)

  ## selection 1: use lasso with very small lambda
  # use gic to select a best lasso model: The unweighted version is faster in the first step.
  model1 = MSE_dfs_c(y = y, X = X, sample_order_list0 = sample_order_list_ini, penalty_factor = penalty_factor, intercept = intercept,
                     lambda_row_list = lambda_row_list_ini, lambda_col_list =  lambda_col_list_ini, lambda_tau_list =  0, min_var = min_var_ini,
                     step_size = step_size, tol = tol, MaxIt = MaxIt)
  
  ## second part, adaptive model
  # build adaptive graph
  group_weight = 1/(sqrt(colSums(model1[[1]]^2))^2)
  var_select1 = group_weight !=Inf # only remain variable selected in the first step.
  if(sum(var_select1) == 0){
    cat("Initialization estimator select no varaibles. Try to use smaller lambda_col.")
    names(model1) = c("Bnow", "MSE", "GIC", "lambda_row", "lambda_col", "lambda_tau", "num_var")
    return(list(model_lasso = model1, model_ad = model1, model_ts = model1))
  }
  
  group_weight[penalty_factor==0] = 0
  sample_order_list = matrix(0,ncol = sum(var_select1), nrow = n)
  edge_weight_list = matrix(0,ncol = sum(var_select1), nrow = n-1)
  
  
  Bsub = as.matrix(model1[[1]][, var_select1], nrow = n)
  for(j in 1:dim(Bsub)[2]){
    edge_weight = abs(Bsub[Graph0$edge[,1],j] - Bsub[Graph0$edge[,2],j])
    g_mstj <- mst(Graph0$graph, weights = edge_weight)
    sample_order_list[,j] = as.vector(dfs(g_mstj, root = sample(1:n))$order)
    dif_tmp = abs(Bsub[sample_order_list[-n,j], j] - Bsub[sample_order_list[-1,j], j])
    edge_weight_list[,j] =  1 /(dif_tmp) # not too many penalty 
  }
  
  if(min(edge_weight_list) == Inf){
    edge_weight_list[,] = 1
  }else{
    edge_weight_list[edge_weight_list == Inf] = max(edge_weight_list[edge_weight_list != Inf]) * 2  # not too many penalty
  }
  # 
  sample_order_list = sample_order_list - 1
  
  new_lambda_row_list = exp(seq(log(max(lambda_row_list) / min(edge_weight_list)), log(min(lambda_row_list) / max(edge_weight_list)), length.out = length(lambda_row_list)))
  
  if(sum(group_weight[var_select1] != 0) ==0){
    new_lambda_col_list = 0
    cat("Initialization estimator select no additional varaibles. Try to use smaller lambda_col.")
  }else{
    new_lambda_col_list = exp(seq(log(max(lambda_col_list) / min(group_weight[var_select1][group_weight[var_select1] != 0])^2), log(min(lambda_col_list) / max(group_weight[var_select1])^2), length.out = length(lambda_col_list)))
  }
  model_ad = GIC_dpsw_c(y = y, X = as.matrix(X[,var_select1], nrow = n), sample_order_list = sample_order_list, edge_weight_list = edge_weight_list, var_weight = group_weight[var_select1]^2, intercept = intercept, lambda_row_list = new_lambda_row_list, lambda_col_list = new_lambda_col_list, lambda_tau_list =  0, step_size = step_size, tol = tol, MaxIt = MaxIt)
  cat(length(var_select1))
  Bfinal = matrix(0,nrow = n,ncol = p)
  
  Bfinal[, var_select1] = model_ad[[1]]
  
  model_ad[[1]] = Bfinal
  
  model_ad[[9]] = sample_order_list
  
  names(model_ad) = c("B", "MSE", "GIC", "lambda_row", "lambda_col", "lambda_tau", "df", "a", "sample_graph_order")
  
  cat("Finish.")
  # names(model1) = c("B", "MSE", "GIC", "lambda_row", "lambda_col", "lambda_tau", "df")
  return(model_ad)
}
