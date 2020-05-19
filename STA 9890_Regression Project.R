## Author: Tanay Mukherjee
## Date: 12th May, 2020

## Project Name: Predicting Financial Distress
## Class: STA 9760

## Mentor: Prof. Kamiar Rahnama Rad



# Libraries for the exercise
library(dplyr)
library(randomForest)
library(ggplot2)
library(gridExtra)
library(glmnet)

# Read the file
fd <- read.csv("C:\\Users\\its_t\\Documents\\CUNY\\Spring 2020\\9890 - Statistical Learning for Data Mining\\Project\\data.csv")

# Take a summary look at the original dataset
# glimpse(fd)

# Filtering for the relevant features that we are going to use for analysis
fd_filtered <- fd %>% select(3:69)


## Fixing categorical variables

# One way is to put a threshold and see if we have any feature which has
# less than 10 unique values. If yes, then we need to manually analyze it
# to check for the column
which(sapply(fd_filtered, function(x) {length(unique(x)) < 10}))
## Output --> named integer(0)
## Therefore we have no categorical variables


## Imputing for missing values:

# I want to not only check if there are any 0 values but also not defined ones 
# +INF or -INF n my dataset for the resgressor variables.
check_index <- fd_filtered %>% 
  select_if(function(x) any(is.na(x) | is.infinite(x))) %>% 
  summarise_each(funs(sum(is.na(.),sum(is.infinite(.)))))

# It should return 0 columns and 1 row with the row number
# if there were any NaN, NAs or INF values
check_index
## Output --> For this dataset we get: data frame with 0 columns and 1 row


## Take a summary look at the filtered dataset
glimpse(fd_filtered)
# Output --> Observations: 3672, Variables: 67


# Picking regressors and predictors
X <- data.matrix(fd_filtered %>% select(2:67))
y <- data.matrix((fd_filtered %>% select(1)))

a <- 1 - min(y)
a
y <- y + a
y

y <- data.matrix(log(y))
# y <- data.matrix(log(fd_filtered %>% select(1)))


is.nan.data.frame <- function(x) do.call(cbind, lapply(x, is.nan))
y[is.nan(y)] <- 0

n <- as.integer(nrow(fd_filtered) - 1)
p <- as.integer(ncol(fd_filtered) - 1 )
n;p;


# Scaling the dataset
scaled_x <- scale(X)

# check that we get mean of 0 and sd of 1
apply(scaled_x, 2, mean)
apply(scaled_x, 2, sd)

scaled_y <- scale(y)
apply(scaled_y, 2, mean)
apply(scaled_y, 2, sd)

## Data split
# Determine row to split on: split
split <- round(n * 0.80)

# Create train
n.train <- nrow(fd_filtered[1:split, ])
# Create test
n.test <- nrow(fd_filtered[(1 + split):n, ])


# n.train <- floor(0.8*n)
# n.test <- n-n.train

# Looping sequence
M = 5

# lr = Lasso Regression
Rsq.test.lr <- rep(0,M)  
Rsq.train.lr <- rep(0,M)
# rr = Ridge Regression
Rsq.test.rr <- rep(0,M)  
Rsq.train.rr <- rep(0,M)
# rf = Random Forest
Rsq.test.rf <- rep(0,M)  
Rsq.train.rf <- rep(0,M)
# en = Elastic Net
Rsq.test.en <- rep(0,M)
Rsq.train.en <- rep(0,M)


#store residuals of train and test sets for each method when running loop 1 time
train.res.lr = rep(0,n.train)
test.res.lr = rep(0,n.test)
train.res.rr = rep(0,n.train)
test.res.rr = rep(0,n.test)
train.res.rf = rep(0,n.train)
test.res.rf = rep(0,n.test)
train.res.en = rep(0,n.train)
test.res.en = rep(0,n.test)


set.seed(1)


for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
  a = 0 #ridge
  cv.fit.rr        =     cv.glmnet(X.train,y.train,alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train, alpha = a, lambda = cv.fit.rr$lambda.min)
  y.train.hat      =     predict(fit, newx = X.train, type = "response") 
  y.test.hat       =     predict(fit, newx = X.test, type = "response") 
  Rsq.test.rr[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.rr[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  #run this only when M = 1, M>1, comment 3 lines below
  if (m<2){
    train.res.rr        =     as.vector(y.train - y.train.hat)
    test.res.rr         =     as.vector(y.test - y.test.hat)
    boxplot(train.res.rr,test.res.rr, horizontal = TRUE, at= c(1,2), 
            names=c("Train","Test"), main = "Ridge Regression")}
  
  a=0.5# elastic-net
  cv.fit.en        =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train, alpha = a, lambda = cv.fit.en$lambda.min)
  y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.en[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.en[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  #only when M = 1
  if(m<2){
    train.res.en     =     y.train - as.vector(y.train.hat)
    test.res.en      =     y.test - as.vector(y.test.hat)
    boxplot(train.res.en,test.res.en, horizontal = TRUE,at= c(1,2), 
            names=c("Train","Test"), main = "Elastic Net") }
  a=1# lasso
  cv.fit.lr        =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train, alpha = a, lambda = cv.fit.lr$lambda.min)
  y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.lr[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.lr[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  #only when M = 1
  if(m<2){
    train.res.lr     =     y.train - as.vector(y.train.hat)
    test.res.lr      =     y.test - as.vector(y.test.hat)
    boxplot(train.res.lr,test.res.lr, horizontal = TRUE,at= c(1,2),names=c("Train","Test"), main = "Lasso")}
  
  
  # fit RF and calculate and record the train and test R squares 
  rf               =     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
  y.test.hat       =     predict(rf, X.test)
  y.train.hat      =     predict(rf, X.train)
  Rsq.test.rf[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.rf[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  #only when M = 1
  if(m<2){
    train.res.rf        =     y.train - as.vector(y.train.hat)
    test.res.rf         =     y.test  - as.vector(y.test.hat)
    boxplot(train.res.rf, test.res.rf,horizontal = TRUE,at= c(1,2), 
            names=c("Train","Test"), main = "Random Forest")}
  
  
  cat(sprintf("m=%3.f| Rsq.test.rf=%.2f,  Rsq.test.en=%.2f,  
              Rsq.test.lr=%.2f, Rsq.test.rr=%.2f| Rsq.train.rf=%.2f,  Rsq.train.en=%.2f, 
              Rsq.train.lr=%.2f, Rsq.train.rr=%.2f| \n", m,  Rsq.test.rf[m], Rsq.test.en[m], 
              Rsq.test.lr[m],  Rsq.test.rr[m],  Rsq.train.rf[m], Rsq.train.en[m], 
              Rsq.train.lr[m], Rsq.train.rr[m]))
}



# # ----
# 
# library(reshape)
# 
# w.plot <- melt(fd_filtered$FinancialDistress) 
# 
# boxplot(fd_filtered$FinancialDistress, horizontal=TRUE, main="y")
# 
# 
# newdata <- log(fd_filtered$FinancialDistress)
# hist(newdata$FinancialDistress)
# 
# # -----

#Boxplots of the test and train R-squares
boxplot(Rsq.test.rf,Rsq.test.en,Rsq.test.lr,Rsq.test.rr, main = "Test R-square",
        font.main = 2, cex.main = 1, at = c(1,2,3,4), names = c("RF","EN","LS","RD"),
        col = "orange",horizontal = FALSE)


boxplot(Rsq.train.rf,Rsq.train.en,Rsq.train.lr,Rsq.train.rr, main = "Train R-square",
        font.main = 2, cex.main = 1, at = c(1,2,3,4), names = c("RF","EN","LS","RD"),
        col = "light blue",horizontal = FALSE)

# Plot 10 fold cross validation curves
plot(cv.fit.en,sub = "Elastic Net", cex.sub = 1) #elasticnet
plot(cv.fit.lr,sub = "Lasso", cex.sub = 1) #lasso
plot(cv.fit.rr,sub = "Ridge", cex.sub = 1) #ridge


#Create bootstraped samples
bootstrapSamples =     5

#Store the importance of each coefficient in RF and the betas in LS,EN,RD
beta.rf.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.en.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)
beta.lr.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)   
beta.rr.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)   

for (m in 1:bootstrapSamples){
  bs_indexes       =     sample(n, replace=T)
  X.bs             =     X[bs_indexes, ]
  y.bs             =     y[bs_indexes]
  
  # fit bs rf
  rf               =     randomForest(X.bs, y.bs, mtry = sqrt(p), importance = TRUE)
  beta.rf.bs[,m]   =     as.vector(rf$importance[,1])
  # fit bs en  
  a                =     0.5
  cv.fit           =     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)  
  beta.en.bs[,m]   =     as.vector(fit$beta)
  #fit bs ls
  a                =      1
  cv.fit           =     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)  
  beta.lr.bs[,m]   =     as.vector(fit$beta)
  #fit bs rd
  a                =     0
  cv.fit           =     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)  
  beta.rr.bs[,m]   =     as.vector(fit$beta)
  
  cat(sprintf("Bootstrap Sample %3.f \n", m))
  
}

# calculate bootstrapped standard errors 
rf.bs.sd    = apply(beta.rf.bs, 1, "sd")
en.bs.sd    = apply(beta.en.bs, 1, "sd")
ls.bs.sd    = apply(beta.lr.bs, 1, "sd")
rd.bs.sd    = apply(beta.rr.bs, 1, "sd")


# fit rf to the whole data
rf.whole         =     randomForest(X, y, mtry = sqrt(p), importance = TRUE)

# fit en,rd,ls to the whole data
a=0.5 # elastic-net
cv.en            =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.en           =     glmnet(X, y, alpha = a, lambda = cv.en$lambda.min)
a=0 # ridge
cv.rr            =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.rr           =     glmnet(X, y, alpha = a, lambda = cv.rr$lambda.min)
a=1 #lasso
cv.lr            =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.lr           =     glmnet(X, y, alpha = a, lambda = cv.lr$lambda.min)

#store the importance of the coefficients with its error (2sd) in a data frame
betaS.rf               =     data.frame(c(1:p), as.vector(rf.whole$importance[,1]), 2*rf.bs.sd)
colnames(betaS.rf)     =     c( "feature", "value", "err")
#store betas in en,rd, ls with its error (2sd) in a data frame
betaS.en               =     data.frame(c(1:p), as.vector(fit.en$beta), 2*en.bs.sd)
colnames(betaS.en)     =     c( "feature", "value", "err")

betaS.rr               =     data.frame(c(1:p), as.vector(fit.rr$beta), 2*rd.bs.sd)
colnames(betaS.rr)     =     c( "feature", "value", "err")

betaS.lr               =     data.frame(c(1:p), as.vector(fit.lr$beta), 2*ls.bs.sd)
colnames(betaS.lr)     =     c( "feature", "value", "err")

#barplots with bootstrapped error bars for rf, en
rfPlot =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + ggtitle("Random Forest")


enPlot =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + ggtitle("Elastic Net")

grid.arrange(rfPlot, enPlot, nrow = 2)

#barplots with bootstrapped error bars for rd, ls
rdPlot =  ggplot(betaS.rr, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)+ ggtitle("Ridge")

lsPlot =  ggplot(betaS.lr, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + ggtitle("Lasso")
grid.arrange(rdPlot, lsPlot, nrow = 2)

#rearrange the order of betas according to the order of the importance of betas in rf
betaS.rf$feature     =  factor(betaS.rf$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.en$feature     =  factor(betaS.en$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.rr$feature     =  factor(betaS.rr$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.lr$feature     =  factor(betaS.lr$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])

rfPlot =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + ggtitle("Random Forest")

enPlot =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + ggtitle("Elastic Net")

grid.arrange(rfPlot, enPlot, nrow = 2)

rdPlot =  ggplot(betaS.rr, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)+ ggtitle("Ridge")

lsPlot =  ggplot(betaS.lr, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)+ ggtitle("Lasso")

grid.arrange(rdPlot, lsPlot, nrow = 2)
