% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/buildKNN.R
\name{buildKNN}
\alias{buildKNN}
\title{Build a undirected KNN network.}
\usage{
buildKNN(dd, k)
}
\arguments{
\item{dd}{A n*n distance matrix of n observations.}

\item{k}{The number of neighborhoods.}
}
\value{
graph: The graph made by make_graph{igraph}.

edge: The three-column matrix includes edge indexes in the first two columns and the distance between nodes of edges in the third column.
}
\description{
Build a undirected KNN network.
}
\examples{
# set.seed(123)
# x = matrix(rnorm(400), nrow = 50,ncol = 8)
# dd = dist(x, diag = T, upper = T)
# dd = as.matrix(dd)
# knn_graph = buildKNN(dd, 5)
}
