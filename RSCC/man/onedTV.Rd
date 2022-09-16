\name{onedTV_c}
\alias{onedTV_c}
\alias{onedTVw_c}
\title{One Dimension Total Variance Proximity (fused lasso)}
\description{
	Solve the proximal operator of the (adjusted) fused lasso penalty, which is also (wegihted) one dimension total variance proximity problem.
	The used method is Condat's method.
	onedTV_c is for the same penalty.
	onedTV_TVw_c is for the weighted penalty.
}
\usage{
onedTV_c(y, lambda)
onedTVw_c(y, lambda)
}
\arguments{
  \item{y}{A length-n vector.}
  \item{lambda}{A scala for onedTV_c and a length-(n-1) vector for onedTVw_c}
}
\value{
  An n*1 matrix.
}

\references{
Barbero, Alvaro, and Suvrit Sra. "Modular proximal optimization for multidimensional total-variation regularization." The Journal of Machine Learning Research 19.1 (2018): 2232-2313.
Condat, Laurent. "A direct algorithm for 1-D total variation denoising."IEEE Signal Processing Letters 20.11 (2013): 1054-1057.	
}
\examples{
set.seed(12345)
x = rnorm(19)
lambda = 1
lambda_list = c(rep(1,6),rep(0.1,6), rep(5,6))
onedTV_c(x, lambda)
onedTVw_c(x, lambda_list)
}

