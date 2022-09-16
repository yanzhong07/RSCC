#' Build a undirected KNN network.
#' 
#' @param dd A n*n distance matrix of n observations.
#' @param k The number of neighborhoods.
#' @return graph: The graph made by make_graph(igraph).
#' @return edge: The three-column matrix includes edge indexes in the first two columns and the distance between nodes of edges in the third column.
#' @examples
#' 
#' x = matrix(rnorm(400), nrow = 20,ncol = 20)
#' x = matrix(rnorm(400), nrow = 50,ncol = 8)
#' dd = as.matrix(dist(x, diag = T, upper = T))
#' knn_graph = buildKNN(dd, 5)
buildKNN = function(dd, k){
  n = dim(dd)[1]
  stopifnot(dim(dd)[1] != dim(dd[2]))
  diag(dd) = 0
  edge = cbind(NA,NA, NA)
  for(i in 1:n){
    edge_index = order(dd[i,])[2:(k+1)]
    edge = rbind(edge,cbind(i,edge_index, dd[i, edge_index]))
  }
  edge = edge[-1,]
  edge = rbind(edge, edge[,c(2,1,3)])
  edge = edge[!duplicated(edge[, 1:2]),]
  edge = edge[edge[,1] < edge[,2],]
  colnames(edge) = c("i", "j", "distance")
  Ge = make_graph(c(t(edge[,1:2])), directed = F)
  return(list(graph = Ge, edge = edge))
}
