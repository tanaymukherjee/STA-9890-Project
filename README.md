# Financial Distress
## STA-9890-Project
In this exercise we are attempting at running various regression models through namely Lasso, Ridge, Elastic Net and Random Forrest to predict on a financial dataset from multiple firms to see if we can predict distress.


## Understanding the models
### Elastic Net:
In statistics and, in particular, in the fitting of linear or logistic regression models, the elastic net is a regularized regression method that linearly combines the L1 and L2 penalties of the lasso and ridge methods.

The elastic net method overcomes the limitations of the LASSO (least absolute shrinkage and selection operator) method which uses a penalty function. In addition to setting and choosing a lambda value elastic net also allows us to tune the alpha parameter where 𝞪 = 0 corresponds to ridge and 𝞪 = 1 to lasso. Simply put, if you plug in 0 for alpha, the penalty function reduces to the L1 (ridge) term and if we set alpha to 1 we get the L2 (lasso) term. Therefore, we can choose an alpha value between 0 and 1 to optimize the elastic net. Effectively this will shrink some coefficients and set some to 0 for sparse selection. In this exercise we will have 𝞪 = 0.5

### Lasso:
Lasso regression uses the L1 penalty term and stands for Least Absolute Shrinkage and Selection Operator. The penalty applied for L2 is equal to the absolute value of the magnitude of the coefficients.

Similar to ridge regression, a lambda value of zero spits out the basic OLS equation, however given a suitable lambda value lasso regression can drive some coefficients to zero. The larger the value of lambda the more features are shrunk to zero. This can eliminate some features entirely and give us a subset of predictors that helps mitigate multi-collinearity and model complexity. Predictors not shrunk towards zero signify that they are important and thus L1 regularization allows for feature selection (sparse selection).

### Ridge:
Ridge regression uses L2 regularization which adds the penalty term to the OLS equation.

The L2 term is equal to the square of the magnitude of the coefficients. In this case if lambda(λ) is zero then the equation is the basic OLS but if it is greater than zero then we add a constraint to the coefficients. This constraint results in minimized coefficients (aka shrinkage) that trend towards zero the larger the value of lambda. Shrinking the coefficients leads to a lower variance and in turn a lower error value. Therefore, Ridge regression decreases the complexity of a model but does not reduce the number of variables, it rather just shrinks their effect.

### Random Forest:
A Random Forest is an ensemble technique capable of performing both regression and classification tasks with the use of multiple decision trees and a technique called Bootstrap Aggregation, commonly known as bagging. What is bagging you may ask? Bagging, in the Random Forest method, involves training each decision tree on a different data sample where sampling is done with replacement.

The basic idea behind this is to combine multiple decision trees in determining the final output rather than relying on individual decision trees.


## Knowing the dataset
We have a total of 66 features and 3761 observation. Our y-variable is called 'financial distress'. We are trying to know how stable firms are when compared to all other financial institutions.

Basically, if the value of financial distress is less than -1 for any firm it means they are not doing well and thus have a high risk of filing for bankruptcy. Similarly, if the value of distress is between -1 to 0 then it seems like the firm is unstable. Lastly, anything over 1 means the firm is safe and is doing well in open markets. That way, we can do a classification exercise out of this data as well but it is beyong our scope for this project.

Right now, we focus on identifying the variables that are useful for our regression, and then run the above 4 models and see how well can they predict the financial distress, and which are the most prominent features that help us achieve better prediction.


## Procedure
- Identify the predictors and re-name them to X1 to X66 for easier representation. For complete details check slide 2 [here](https://github.com/tanaymukherjee/STA-9890-Project/blob/master/Video%20Presentation/STA%209890_Project%20Presentation.pdf)
- Standardize the x-variables
- Check out for skewness in y - variable. If the skew is high then transform.
- The y-variable here was highly skewed and also had negative values so had to transform using log:
```
y = log(y + 1- min(y))
```
- Break the dataset into two parts. Training set has 80% of the records and test set has remianing 20%
- Run the simulation for all 4 regression models discussed earlier for 100 times.
- For Elastic Net, Lasso and Ridge we use 10-fold cross validation method to tune the lambdas.
- Then we use the R-square test equation given to us as part of the project guideline in part 3.d [here](https://github.com/tanaymukherjee/STA-9890-Project/blob/master/Guidelines/Project%20Tasks.pdf)
- We plot the box-plot of train and test sets and then also plot the box-plot for the residuals once the simulation is over.
- Also, we plot the 10-fold CV curves for Elastic Net, Lasso, Ridge.
- All the plots can be seen in the final report on the root directory - "STA 9890_Project Report_Tanay Mukherjee.pptx"
features and how are they coming for 4 regression models we discussed above in brief.
- Lastly, we also measure the time needed to tune each of these models and compare it with others to see the efficiency.


## Steps to replicate
- Download the file from [data](https://github.com/tanaymukherjee/STA-9890-Project/tree/master/Data) folder in this repo.
- Open R Studio and set your working directory to the place where you have the file from previous step saved.
- Load the code available on the root directory [here](https://github.com/tanaymukherjee/STA-9890-Project/blob/master/STA%209890_Regression%20Project.R)
- Run the code, and see the analysis unfold as each model tries to regress the feature variables to predict the best distress for financial institutions.


## Appendix
1. Folder: Guidelines, has all the tasks mentioned with the submission template
2. Folder: Plots - it has all the plots as png files
3. Folder: Output has the result of all R-squares saved from the console
4. Proposal: It has the initial proposal submitted on the assignment of this project confirming the chosen dataset.
5. Video Presentation: It has the pdf file used in the video recording
6. Video recording: https://vimeo.com/420873137. It is password protected and is available for view to only Prof. Rad.

## Submission to
- Prof. Kamiar Rahnama Rad
```
Subject: 9890 - Statistical Learning for Data Mining
Session: Fall, 2020
```
