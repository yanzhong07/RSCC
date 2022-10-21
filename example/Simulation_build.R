library(mvtnorm)
library(Matrix)

set_beta1 = function(x,y){
  beta10 = c(1.5,-0.75,0.75,-1.5)
  ind = x-y
  if(ind < -0.5){
    beta_use = beta10[1]
  }else if(ind >= -0.5 & ind < 0){
    beta_use = beta10[2]
  }else if(ind >= 0 & ind < 0.5){
    beta_use = beta10[3]
  }else{
    beta_use = beta10[4]
  }
  return(beta_use)
}

set_beta2 = function(x,y){
  beta20 =  c(-0.5,1,-1.5,-1, 0.5)
  if(x < 0.5 & y < 0.5){
    beta_use = beta20[1]
  }else if(x < 0.5 & y > 0.5){
    beta_use = beta20[2]
  }else if(x > 0.5 & y < 0.5){
    beta_use = beta20[3]
  }else{
    beta_use = beta20[4]
  }
  if( sqrt(((x-0.5)^2 + (y-0.5)^2)) < 0.25){
    beta_use = beta20[5]
  }
  
  return(beta_use)
}

ff1 = function(x){
  return(0.094 / 0.09 * (x-0.7)^2+0.15)
}

ff2r = function(x){
  return(-0.406 / (0.45^2) * (x-0.55)^2+0.65)
}

ff2l = function(x){
  return(-9 * (x-0.55)^2+0.65)
}

ff3r = function(x){
  return(12 * (x-0.55)^2+0.65)
}

ff3l = function(x){
  return( (x-0.55)^2+0.65)
}

set_beta3 = function(x,y){
  beta30 =  c(-0.5,-1.5,1,-1, 1.5,0.5)
  if(x <= 0.55){
    tmp1 = ff1(x)
    tmp2 = ff2l(x)
    tmp3 = ff3l(x)
    if(y >= tmp3){
      beta_use = beta30[1]
    }else if(y >= tmp1){
      if(y >= tmp2){
        beta_use = beta30[2]
      }else{
        beta_use = beta30[5]
      }
    }else{
      if(y >= tmp2){
        beta_use = beta30[3]
      }else{
        beta_use = beta30[6]
      }
    }
  }else{
    tmp1 = ff1(x)
    tmp2 = ff2r(x)
    tmp3 = ff3r(x)
    if(y >= tmp3){
      beta_use = beta30[1]
    }else if(y >= tmp1){
      if(y >= tmp2){
        beta_use = beta30[4]
      }else{
        beta_use = beta30[5]
      }
    }else{
      beta_use = beta30[6]
    }
  }
  return(beta_use)
}

simulation = function(n = 1000, p = 20, phi = 0.3, cor.index = 0.5){ 
  r = 3
  l1 = runif(n)
  l2 = runif(n)
  loc = cbind(l1,l2)
  dd = rdist(loc)
  d1 = rdist(l1)
  d2 = rdist(l2)
  
  ## generate D
  beta1 = rep(0,n)
  beta2 = rep(0,n)
  beta3 = rep(0,n)
  for(i in 1:n){
    beta1[i] = set_beta1(l1[i],l2[i])
    beta2[i] = set_beta2(l1[i],l2[i])
    beta3[i] = set_beta3(l1[i],l2[i])
  }
  Dtrue = cbind(beta1,beta2,beta3)
  
  ## generate A, link function
  Atrue = matrix(0, nrow = p, ncol = r)
  
  Aselect = sample(1:p,r)
  for(i in 1:r){
    Atrue[Aselect[i],i] = 1
  }
  
  ## generate X
  covm = exp(-dd/phi)
  Z = NA
  for(i in 1:p){
    tmp = mvtnorm::rmvnorm(1,mean = rep(0,n), sigma = covm)
    Z = cbind(Z, c(tmp))
  }
  Z = as.matrix(Z[,-1])
  
  Sigma = matrix(0,p,p)
  for(i in 1:p){
    for(j in 1:p){
      Sigma[i,j] = cor.index^(abs(i-j))  # this is an important parameter
    }
  }
  choS =chol(Sigma)
  X =Z %*% choS 
  
  ## construct y
  Et = rnorm(n,sd = 0.1)
  ypure = rowSums( (X %*% Atrue) * Dtrue)
  aintercept =  runif(1,-0.2,0.2)
  y = ypure + Et + aintercept
  return(list(dd = dd, X = X, y = y, Atrue = Atrue, Dtrue = Dtrue, loc = loc, aintercept = aintercept))
}

construct_H = function(dd, k =5){
  n = dim(dd)[1]
  edge = cbind(NA,NA, NA)
  for(i in 1:n){
    edge_index = order(dd[i,])[2:(k+1)]
    edge = rbind(edge,cbind(i,edge_index, dd[i, edge_index]))
  }
  edge = edge[-1,]
  edge = rbind(edge, edge[,c(2,1,3)])
  edge = edge[!duplicated(edge[, 1:2]),]
  edge = edge[edge[,1] < edge[,2],]
  
  Ge = make_graph(c(t(edge[,1:2])), directed = F)
  g_mst <- mst(Ge, weights = edge[,3])
  sample_order = as.vector(dfs(g_mst, root = 1)$order)
  
  zhong = g_mst[1:n,1:n]
  zhong = t(zhong) + zhong
  edge_mst = summary(zhong)
  edge_mst  = edge_mst[edge_mst[,1] < edge_mst[,2],]
  
  ## construct H matrix for mst
  H = Matrix(0, ncol =n,nrow = n-1)
  for(i in 1:dim(edge_mst)[1]){
    H[i,edge_mst[i,1]] = 1
    H[i,edge_mst[i,2]] = -1
  }  
  return(list(H = H, Ge = Ge, edge = edge, sample_order = sample_order))
}