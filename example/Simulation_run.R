## This is an example simulation for 2 true related variables.
library(RColorBrewer)
library(ggplot2)
library(RSCC)
library(fields)
library(mvtnorm)
library(igraph)
library(Matrix)
library(glmnet)
source("example/Simulation_build.R")

set.seed(123)
n = 1000
p = 20
sim_data = simulation(n,p) # take about half a minute to generate data
X = sim_data$X
y = sim_data$y
Atrue = sim_data$Atrue
Dtrue = sim_data$Dtrue
dd = sim_data$dd
loc = sim_data$loc

# build KNN network and construct network.
k = 5 - 1
bad_graph = T
while(bad_graph){
  k = k + 1
  Graph0 = buildKNN(dd, k = k)
  bad_graph = !igraph::is.connected(Graph0$graph)
}

## run our model, take 3~5 minutes to search lambda and build model.
system.time({model1 = RSCC_gic(y, X, Graph0, penalty_factor =NULL, lambda_row_list = exp(seq(0,-10,length.out = 10)),
                  lambda_col_list = exp(seq(0,-10,length.out = 10)), lambda_row_list_ini = exp(seq(0,-10,length.out = 10)), lambda_col_list_ini = exp(seq(0,-10,length.out = 10)), intercept = T)})

RSCC_select_var = c(1:p)[colSums(model1$B !=0) !=0]
true_var = c(1:p)[rowSums(Atrue) !=0]
RSCC_select_var # variables selected by RSCC
true_var # true related variables
Bpre = round(model1$B, 2)

dat = data.frame(loc1 = loc[,1], loc2 = loc[,2], 
                 b1t = Dtrue[,1], b2t = Dtrue[,2], b3t = Dtrue[,3],
                 b1p = model1$B[,Atrue[,1] ==1], b2p = model1$B[,Atrue[,2] ==1], b3p = model1$B[,Atrue[,3] ==1])

cor(dat$b1t, dat$b1p)
cor(dat$b2t, dat$b2p)
cor(dat$b3t, dat$b3p)

# true beta1
p1t = ggplot(data = dat,aes(x = loc1,y = loc2)) + 
  geom_point(aes(colour = b1t), size = 1.5) + 
  scale_color_gradientn(colours = brewer.pal(3,"Spectral")) + 
  xlab(NULL) + ylab(NULL)+ theme_linedraw()+
  theme(panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(), panel.grid.major.y = element_blank(), panel.grid.minor.y = element_blank(),
        legend.position="right",legend.title = element_blank())+ ggtitle("beta1_true")
p1t


p1p = ggplot(data = dat,aes(x = loc1,y = loc2)) + 
  geom_point(aes(colour = b1p), size = 1.5) + 
  scale_color_gradientn(colours = brewer.pal(3,"Spectral")) + 
  xlab(NULL) + ylab(NULL)+ theme_linedraw()+
  theme(panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(), panel.grid.major.y = element_blank(), panel.grid.minor.y = element_blank(),
        legend.position="right",legend.title = element_blank())+ ggtitle("beta1_predict")
#geom_path(aes(x = path1, y = path2)) +
p1p


## compared with glm
model.glmnet.cv = cv.glmnet(X, y, family = "gaussian", intercept = T, gamma = 0)
model.glmnet = glmnet(X, y, family = "gaussian",lambda = model.glmnet.cv$lambda.1se, gamma = 0)
glmnet_select_var = c(1:p)[as.numeric(model.glmnet$beta) !=0]
glmnet_select_var
true_var
