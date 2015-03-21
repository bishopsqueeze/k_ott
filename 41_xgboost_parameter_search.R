##------------------------------------------------------------------
## Process Raw Data Files
##------------------------------------------------------------------

##------------------------------------------------------------------
## Load libraries
##------------------------------------------------------------------
require(xgboost)
require(methods)

##------------------------------------------------------------------
## Clear the workspace
##------------------------------------------------------------------
rm(list=ls())

##------------------------------------------------------------------
## Set the working directory
##------------------------------------------------------------------
setwd("/Users/alexstephens/Development/kaggle/otto/data")

##------------------------------------------------------------------
## Read in the various files
##------------------------------------------------------------------
load("/Users/alexstephens/Development/kaggle/otto/data/02_ProcessData_RawData.Rdata")
#load("/Users/alexstephens/Development/kaggle/otto/data/02_ProcessData_Base2Data.Rdata")

##------------------------------------------------------------------
## Modify the target
##------------------------------------------------------------------
y = train.target
y = gsub('L','',y)
y = as.integer(y)-1         ## xgboost take features in [0,numOfClass)

##------------------------------------------------------------------
## Combine train/test
##------------------------------------------------------------------
x = rbind(train.feat, test.feat)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))

## indicators for each
trind = 1:length(y)
teind = (nrow(train.feat)+1):nrow(x)

##------------------------------------------------------------------
## Cross-Validation Tests of Parameter Space
##------------------------------------------------------------------

##------------------------------------------------------------------
## Test #1:  Alter max_depth
##------------------------------------------------------------------

## define paramter to test
tryDepth        <- 2:20
tryDepthList    <- list()
cv.nround       <- 100
num.fold        <- 5

## placeholder for cross-validation results
tryDepthTrainRes    <- matrix(, nrow=cv.nround, ncol=length(tryDepthList))
tryDepthTestRes     <- matrix(, nrow=cv.nround, ncol=length(tryDepthList))

## define the parameter list
for (i in 1:length(tryDepth))
{
    
    tryDepthList[[i]] <- list( "objective" = "multi:softprob",
                            "eval_metric" = "mlogloss",
                            "num_class" = 9,
                            "nthread" = 8,
                            "eta" = 0.3,
                            "max_depth" = tryDepth[i],
                            "min_child_weight" = 1,
                            "subsample" = 1,
                            "colsample_bytree" = 1)
}

## loop over each variant and collect results
for (i in 1:length(tryDepthList))
{
    bst.cv = xgb.cv(param=tryDepthList[[i]], data = x[trind,], label = y, nfold = num.fold, nrounds=cv.nround, verbose=2)
    
    tryDepthTrainRes[,i] <- as.numeric(bst.cv[,train.mlogloss.mean])
    tryDepthTestRes[,i]  <- as.numeric(bst.cv[,test.mlogloss.mean])
}

## save the results
save(   tryDepthTrainRes, tryDepthTestRes,
        file="/Users/alexstephens/Development/kaggle/otto/data/41_xgboost_parameter_search_tryDepth.Rdata")

## plot the minimum achieved
pdf(file="/Users/alexstephens/Development/kaggle/otto/figs/41_xgboost_parameter_search_tryDepth.pdf")
    plot(tryDepth, apply(tryDepthTestRes, 2, min), type="b", col="blue", xlab="max_depth", ylab="Test CV mLogLoss", main="Test #1")
dev.off()



##------------------------------------------------------------------
## Test #2:  subsample
##------------------------------------------------------------------


## define paramter to test
trySubsamp      <- seq(0.5,1,0.1)
trySubsampList  <- list()
cv.nround       <- 100
num.fold        <- 5

## placeholder for cross-validation results
trySubsampTrainRes    <- matrix(, nrow=cv.nround, ncol=length(trySubsamp))
trySubsampTestRes     <- matrix(, nrow=cv.nround, ncol=length(trySubsamp))

## define the parameter list
for (i in 1:length(trySubsamp))
{
    
    trySubsampList[[i]] <- list( "objective" = "multi:softprob",
                                 "eval_metric" = "mlogloss",
                                 "num_class" = 9,
                                 "nthread" = 8,
                                 "eta" = 0.3,
                                 "max_depth" = 6,
                                 "min_child_weight" = 1,
                                 "subsample" = trySubsamp[i],
                                 "colsample_bytree" = 1)
}

## loop over each variant and collect results
for (i in 1:length(trySubsampList))
{
    bst.cv = xgb.cv(param=trySubsampList[[i]], data = x[trind,], label = y, nfold = num.fold, nrounds=cv.nround, verbose=2)
    
    trySubsampTrainRes[,i] <- as.numeric(bst.cv[,train.mlogloss.mean])
    trySubsampTestRes[,i]  <- as.numeric(bst.cv[,test.mlogloss.mean])
}


