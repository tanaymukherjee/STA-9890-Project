## Author: Tanay Mukherjee
## Date: 12th May, 2020

## Project Name: Predicting Financial Distress
## Class: STA 9760

## Mentor: Prof. Kamiar Rahnama Rad

#------------------------------------------------------------------

# Libraries for the exercise
library(dplyr)
library(tidyr)
library(randomForest)
library(ggplot2)
library(gridExtra)
library(glmnet)
library(tidyverse)

# Set the working directory on your local machine to replicate it. The file is under data folder in this repository.
# Read the file
fd <- read.csv("C:\\Users\\its_t\\Documents\\CUNY\\Spring 2020\\9890 - Statistical Learning for Data Mining\\Project\\Financial Distress.csv")

# Take a summary look at the original dataset
glimpse(fd)

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


# Picking regressors and predictors
X <- data.matrix(fd_filtered %>% select(2:67))
y <- data.matrix((fd_filtered %>% select(1)))


# Transforming our y-variable to normalize the skweness seen in the distribution
y <- y + 1 - min(y)
y <- data.matrix(log(y))


# Setting the value of rows and columns
n <- as.integer(nrow(fd_filtered))
p <- as.integer(ncol(fd_filtered) - 1)
n;p;


# Scaling the dataset
scaled_x <- scale(X)

# check that we get mean of 0 and sd of 1
apply(scaled_x, 2, mean)
apply(scaled_x, 2, sd)

## Data split
# Determine row to split on: split
split <- round(n * 0.80)

# Create training set
n.train <- nrow(fd_filtered[1:split, ])
# Create test set
n.test <- nrow(fd_filtered[(1 + split):n, ])


# Looping sequence
M = 100

# lr = Lasso Regression
Rsq.test.lr  <- rep(0,M)
Rsq.train.lr <- rep(0,M)

# rr = Ridge Regression
Rsq.test.rr  <- rep(0,M)  
Rsq.train.rr <- rep(0,M)

# rf = Random Forest
Rsq.test.rf  <- rep(0,M)  
Rsq.train.rf <- rep(0,M)

# en = Elastic Net
Rsq.test.en  <- rep(0,M)
Rsq.train.en <- rep(0,M)


# Store residuals of train and test sets for each method when running loop 1 time
train.res.lr <- rep(0,n.train)
test.res.lr  <- rep(0,n.test)
train.res.rr <- rep(0,n.train)
test.res.rr  <- rep(0,n.test)
train.res.rf <- rep(0,n.train)
test.res.rf  <- rep(0,n.test)
train.res.en <- rep(0,n.train)
test.res.en  <- rep(0,n.test)


# Set the seed of R‘s random number generator, which is useful for
# creating simulations or random objects that can be reproduced.
set.seed(1)


for (m in c(1:M)) {
  shuffled_indexes <-     sample(n)
  train            <-     shuffled_indexes[1:n.train]
  test             <-     shuffled_indexes[(1+n.train):n]
  X.train          <-     X[train, ]
  y.train          <-     y[train]
  X.test           <-     X[test, ]
  y.test           <-     y[test]
  
  
  # Fitting Lasso and doing calculations for R^2
  a=1 # Lasso
  lr_start         <-     Sys.time()
  cv.fit.lr        <-     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
  fit              <-     glmnet(X.train, y.train, alpha = a, lambda = cv.fit.lr$lambda.min)
  y.train.hat      <-     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       <-     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.lr[m]   <-     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.lr[m]  <-     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  resid.test.lr    <-     as.vector(y.test - y.test.hat)
  resid.train.lr   <-     as.vector(y.train - y.train.hat)
  lr_end           <-     Sys.time()
  lr_time          <-     lr_end - lr_start
  
  
  
  # Fitting Elastic-net and  doing calculations for R^2
  a=0.5 # Elastic-net
  en_start         <-     Sys.time()
  cv.fit.en        <-     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
  fit              <-     glmnet(X.train, y.train, alpha = a, lambda = cv.fit.en$lambda.min)
  y.train.hat      <-     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       <-     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.en[m]   <-     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.en[m]  <-     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  resid.test.en    <-     as.vector(y.test - y.test.hat)
  resid.train.en   <-     as.vector(y.train - y.train.hat)
  en_end           <-     Sys.time()
  en_time          <-     en_end - en_start
  
  
  # Fitting Ridge and doing calculations for R^2
  a=0 # Ridge
  rr_start         <-     Sys.time()
  cv.fit.rr        <-     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
  fit              <-     glmnet(X.train, y.train, alpha = a, lambda = cv.fit.rr$lambda.min)
  y.train.hat      <-     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       <-     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.rr[m]   <-     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.rr[m]  <-     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  resid.test.rr    <-     as.vector(y.test - y.test.hat)
  resid.train.rr   <-     as.vector(y.train - y.train.hat)
  rr_end           <-     Sys.time()
  rr_time          <-     rr_end - rr_start
  
  
  # Fitting Random Forest and doing calculations for R^2
  rf_start         <-     Sys.time()
  rf               <-     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
  y.test.hat       <-     predict(rf, X.test)
  y.train.hat      <-     predict(rf, X.train)
  Rsq.test.rf[m]   <-     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.rf[m]  <-     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  resid.test.rf    <-     as.vector(y.test - y.test.hat)
  resid.train.rf   <-     as.vector(y.train - y.train.hat)
  rf_end           <-     Sys.time()
  rf_time          <-     rf_end - rf_start
  
  
  # Printing All the Train R-square:
  cat(sprintf("m =%3.f| Rsq.train.rf=%.3f, Rsq.train.rr=%.3f,  Rsq.train.en=%.3f, Rsq.train.lr=%.3f \n", 
              m,  Rsq.train.rf[m], Rsq.train.rr[m], Rsq.train.en[m], Rsq.train.lr[m]))
  
  # Printing All the Test R-square:
  cat(sprintf("m =%3.f| Rsq.test.rf=%.3f,  Rsq.test.rr=%.3f,   Rsq.test.en=%.3f,  Rsq.test.lr=%.3f \n", 
              m,  Rsq.test.rf[m], Rsq.test.rr[m], Rsq.test.en[m], Rsq.test.lr[m]))
}



#------------------------------------------------------------------
## Que - 4

#  Distribution of y
hist_y <- fd_filtered %>%
  ggplot(aes(x=FinancialDistress))+
  geom_histogram(bins = 25) + ggtitle("Financial Distress with no transformation") +
  theme(axis.title.x=element_blank())

#  Distribution of transformed y
hist_log_y <- fd_filtered %>%
  ggplot(aes(x=log(sqrt(FinancialDistress*FinancialDistress))+0.5)) +
  geom_histogram(bins = 25) + ggtitle("Financial Distress with Log transformation")+
  theme(axis.title.x=element_blank())

grid.arrange(hist_y, hist_log_y, nrow=2) 


#Boxplots of the test and train R-squares
rsq_train_df <- data.frame(Rsq.train.lr,Rsq.train.en,Rsq.train.rr,Rsq.train.rf)
rsq_train_data <- rsq_train_df %>% gather(variable, RSquare) %>%
  separate(variable, c("Measure", "Category","Model"), sep = "\\.")

rsq_test_df <- data.frame(Rsq.test.lr,Rsq.test.en,Rsq.test.rr,Rsq.test.rf)
rsq_test_data <- rsq_test_df %>% gather(variable, RSquare) %>%
  separate(variable, c("Measure", "Category","Model"), sep = "\\.")

rsq_data <- rbind(rsq_train_data,rsq_test_data)
rsq_data$Model[grepl("lr", rsq_data$Model, ignore.case=T)] <- "Lasso"
rsq_data$Model[grepl("en", rsq_data$Model, ignore.case=T)] <- "Elastic Net"
rsq_data$Model[grepl("rr", rsq_data$Model, ignore.case=T)] <- "Ridge"
rsq_data$Model[grepl("rf", rsq_data$Model, ignore.case=T)] <- "Random Forest"
rsq_data$Category <- factor(rsq_data$Category, levels = c("train", "test"))


rsq_plot <- rsq_data %>%
  ggplot(aes(x=Model, y=RSquare, fill=Model)) + geom_boxplot() +
  facet_wrap(~Category)
rsq_plot


# Plot 10 fold cross validation curves
par(mfrow=c(3,1))
plot(cv.fit.en,sub = "CV for Elastic Net", cex.sub = 1) #elasticnet
plot(cv.fit.lr,sub = "CV for Lasso", cex.sub = 1) #lasso
plot(cv.fit.rr,sub = "CV for Ridge", cex.sub = 1) #ridge
par(mfrow=c(1,1))
            
# Calculating minimum lambdas:
cv.fit.rr$lambda.min
cv.fit.en$lambda.min
cv.fit.lr$lambda.min


# Residual Plots
residual_train_df <- data.frame(resid.train.lr,resid.train.en,resid.train.rr,resid.train.rf)
residual_train_data <- residual_train_df %>% gather(variable, Residual) %>%
  separate(variable, c("Measure", "Category","Model"), sep = "\\.")

residual_test_df <- data.frame(resid.test.lr,resid.test.en,resid.test.rr,resid.test.rf)
residual_test_data <- residual_test_df %>%
  gather(variable, Residual) %>%
  separate(variable, c("Measure", "Category","Model"), sep = "\\.")

residual_data <- rbind(residual_train_data,residual_test_data)
residual_data$Model[grepl("lr", residual_data$Model, ignore.case=T)] <- "Lasso"
residual_data$Model[grepl("en", residual_data$Model, ignore.case=T)] <- "Elastic Net"
residual_data$Model[grepl("rr", residual_data$Model, ignore.case=T)] <- "Ridge"
residual_data$Model[grepl("rf", residual_data$Model, ignore.case=T)] <- "Random Forest"
residual_data$Category <- factor(residual_data$Category, levels = c("train", "test"))

residual_plot <- residual_data %>%
  ggplot(aes(x=Model, y=Residual, color=Model)) + geom_boxplot() +
  facet_wrap(~Category)
residual_plot


#------------------------------------------------------------------
#Create bootstraped samples
bootstrapSamples = 100

#Store the importance of each coefficient in RF and the betas in LS,EN,RD
beta.rf.bs       <-    matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.en.bs       <-    matrix(0, nrow = p, ncol = bootstrapSamples)
beta.lr.bs       <-    matrix(0, nrow = p, ncol = bootstrapSamples)   
beta.rr.bs       <-    matrix(0, nrow = p, ncol = bootstrapSamples)   


bs_start           <-     Sys.time()
for (m in 1:bootstrapSamples){
  bs_indexes       <-    sample(n, replace=T)
  X.bs             <-    X[bs_indexes, ]
  y.bs             <-    y[bs_indexes]
  
  # Fit Bootstrap on Random Forest
  rf               <-     randomForest(X.bs, y.bs, mtry = sqrt(p),ntree=15, importance = TRUE)
  beta.rf.bs[,m]   <-     as.vector(rf$importance[,1])
  
  # Fit Bootstrap on Elastic-net
  a = 0.5
  cv.fit           <-     cv.glmnet(X.bs, y.bs, intercept = T, alpha = a, nfolds = 10)
  fit              <-     glmnet(X.bs, y.bs, intercept = T, alpha = a, lambda = cv.fit$lambda.min)  
  beta.en.bs[,m]   <-     as.vector(fit$beta)
  
  # Fit Bootstrap on Lasso Regression
  a = 1
  cv.fit           <-     cv.glmnet(X.bs, y.bs, intercept = T, alpha = a, nfolds = 10)
  fit              <-     glmnet(X.bs, y.bs, intercept = T, alpha = a, lambda = cv.fit$lambda.min)  
  beta.lr.bs[,m]   <-     as.vector(fit$beta)
  
  # Fit Bootstrap on Ridge Regression
  a = 0
  cv.fit           <-     cv.glmnet(X.bs, y.bs, intercept = T, alpha = a, nfolds = 10)
  fit              <-     glmnet(X.bs, y.bs, intercept = T, alpha = a, lambda = cv.fit$lambda.min)  
  beta.rr.bs[,m]   <-     as.vector(fit$beta)
  
  cat(sprintf("Bootstrap Sample %3.f \n", m))
  
}
bs_end             <-     Sys.time()
bs_time            <-     bs_end - bs_start

# calculate bootstrapped standard errors 
rf.bs.sd    <-   apply(beta.rf.bs, 1, "sd")
en.bs.sd    <-   apply(beta.en.bs, 1, "sd")
lr.bs.sd    <-   apply(beta.lr.bs, 1, "sd")
rr.bs.sd    <-   apply(beta.rr.bs, 1, "sd")


# Fit Random Forest to the whole data
rf.whole               <-     randomForest(X, as.vector(y), mtry = floor(sqrt(p)),ntree=15, importance = TRUE)
betaS.rf               <-     data.frame(names(X[1,]), as.vector(rf$importance[,1]), 2*rf.bs.sd)
colnames(betaS.rf)     <-     c( "feature", "value", "err")

# Fit Elastic-net, Lasso, Ridge to the whole data
# Elastic Net
a = 0.5 
cv.en                  <-     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.en                 <-     glmnet(X, y, alpha = a, lambda = cv.en$lambda.min)
betaS.en               <-     data.frame(names(X[1,]), as.vector(fit$beta), 2*en.bs.sd)
colnames(betaS.en)     <-     c( "feature", "value", "err")

# Ridge Regression
a = 0 
cv.rr                  <-     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.rr                 <-     glmnet(X, y, alpha = a, lambda = cv.rr$lambda.min)
betaS.rr               <-     data.frame(names(X[1,]), as.vector(fit$beta), 2*rr.bs.sd)
colnames(betaS.rr)     <-     c( "feature", "value", "err")

# Lasso Regression
a = 1
cv.lr                  <-     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.lr                 <-     glmnet(X, y, alpha = a, lambda = cv.lr$lambda.min)
betaS.lr               <-     data.frame(names(X[1,]), as.vector(fit$beta), 2*lr.bs.sd)
colnames(betaS.lr)     <-     c( "feature", "value", "err")


#rearrange the order of betas according to the order of the importance of betas in rf
betaS.rf$feature     <-  factor(betaS.rf$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.en$feature     <-  factor(betaS.en$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.rr$feature     <-  factor(betaS.rr$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.lr$feature     <- factor(betaS.lr$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])

enPlot <-  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + ggtitle("Elastic Net")+
  theme(axis.title.x=element_blank())

lrPlot <-  ggplot(betaS.lr, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)+ ggtitle("Lasso")+
  theme(axis.title.x=element_blank())

rrPlot <-  ggplot(betaS.rr, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)+ ggtitle("Ridge")

rfPlot <-  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + ggtitle("Random Forest")+
  theme(axis.title.x=element_blank())

grid.arrange(rfPlot, enPlot, lrPlot, rrPlot, nrow = 4)
#------------------------------------------------------------------

# Measure of time for tuning
time_tuning <- data.frame(en_time, lr_time, rf_time, rr_time, bs_time)
time_tuning <- as.data.frame(t(as.matrix(time_tuning)))
time_tuning
