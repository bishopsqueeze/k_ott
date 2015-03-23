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

## save the results
save(   trySubsampTrainRes, trySubsampTestRes,
file="/Users/alexstephens/Development/kaggle/otto/data/41_xgboost_parameter_search_trySubsamp.Rdata")

## plot the minimum achieved
pdf(file="/Users/alexstephens/Development/kaggle/otto/figs/41_xgboost_parameter_search_trySubsamp.pdf")
plot(trySubsamp, apply(trySubsampTestRes, 2, min), type="b", col="blue", xlab="subsamp %", ylab="Test CV mLogLoss", main="Test #1")
dev.off()





##------------------------------------------------------------------
## Test #3:  colsample_bytree
##------------------------------------------------------------------

## define paramter to test
tryColsamp      <- seq(0.5,1,0.1)
tryColsampList  <- list()
cv.nround       <- 100
num.fold        <- 5

## placeholder for cross-validation results
tryColsampTrainRes    <- matrix(, nrow=cv.nround, ncol=length(tryColsamp))
tryColsampTestRes     <- matrix(, nrow=cv.nround, ncol=length(tryColsamp))

## define the parameter list
for (i in 1:length(tryColsamp))
{
    
    tryColsampList[[i]] <- list( "objective" = "multi:softprob",
    "eval_metric" = "mlogloss",
    "num_class" = 9,
    "nthread" = 8,
    "eta" = 0.3,
    "max_depth" = 7,
    "min_child_weight" = 1,
    "subsample" = 0.9,
    "colsample_bytree" = tryColsamp[i])
}

## loop over each variant and collect results
for (i in 1:length(tryColsampList))
{
    bst.cv = xgb.cv(param=tryColsampList[[i]], data = x[trind,], label = y, nfold = num.fold, nrounds=cv.nround, verbose=2)
    
    tryColsampTrainRes[,i] <- as.numeric(bst.cv[,train.mlogloss.mean])
    tryColsampTestRes[,i]  <- as.numeric(bst.cv[,test.mlogloss.mean])
}

## save the results
save(   tryColsampTrainRes, tryColsampTestRes,
file="/Users/alexstephens/Development/kaggle/otto/data/41_xgboost_parameter_search_tryColsamp.Rdata")

## plot the minimum achieved
pdf(file="/Users/alexstephens/Development/kaggle/otto/figs/41_xgboost_parameter_search_tryColsamp.pdf")
plot(tryColsamp, apply(tryColsampTestRes, 2, min), type="b", col="blue", xlab="colsamp %", ylab="Test CV mLogLoss", main="Test #1")
dev.off()


##------------------------------------------------------------------
## Test #4:  eta
##------------------------------------------------------------------

## define paramter to test
tryEta          <- c(1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005)
tryEtaList      <- list()
cv.nround       <- 1000
cv.nround.2     <- 5000
num.fold        <- 5

## placeholder for cross-validation results
tryEtaTrainRes    <- matrix(, nrow=cv.nround, ncol=length(tryEta))
tryEtaTestRes     <- matrix(, nrow=cv.nround, ncol=length(tryEta))

tryEtaTrainRes.2  <- matrix(, nrow=cv.nround.2, ncol=length(tryEta))
tryEtaTestRes.2   <- matrix(, nrow=cv.nround.2, ncol=length(tryEta))

## define the parameter list
for (i in 1:length(tryEta))
{
    
    tryEtaList[[i]] <- list( "objective" = "multi:softprob",
                                 "eval_metric" = "mlogloss",
                                 "num_class" = 9,
                                 "nthread" = 8,
                                 "eta" = tryEta[i],
                                 "max_depth" = 7,
                                 "min_child_weight" = 1,
                                 "subsample" = 0.9,
                                 "colsample_bytree" = 0.9)
}

## loop over each variant and collect results
#for (i in 1:length(tryEtaList))
for (i in 7:8)
{
    #bst.cv = xgb.cv(param=tryEtaList[[i]], data = x[trind,], label = y, nfold = num.fold, nrounds=cv.nround, verbose=2)
    bst.cv = xgb.cv(param=tryEtaList[[i]], data = x[trind,], label = y, nfold = num.fold, nrounds=cv.nround.2, verbose=2)
    
    tryEtaTrainRes.2[,i] <- as.numeric(bst.cv[,train.mlogloss.mean])
    tryEtaTestRes.2[,i]  <- as.numeric(bst.cv[,test.mlogloss.mean])
}


## save the results
#save(   tryEtaTrainRes, tryEtaTestRes,
#file="/Users/alexstephens/Development/kaggle/otto/data/41_xgboost_parameter_search_tryEta.Rdata")
save(   tryEtaTrainRes.2, tryEtaTestRes.2,
file="/Users/alexstephens/Development/kaggle/otto/data/41_xgboost_parameter_search_tryEta2.Rdata")

## plot the minimum achieved
#pdf(file="/Users/alexstephens/Development/kaggle/otto/figs/41_xgboost_parameter_search_tryEta.pdf")
tmp.1 <- apply(tryEtaTestRes, 2, min)
tmp.2 <- apply(tryEtaTestRes.2, 2, min)

plot(tryEta, c(apply(tryEtaTestRes, 2, min)+apply(tryEtaTestRes.2, 2, min)), type="b", col="blue", xlab="Eta", ylab="Test CV mLogLoss", main="Test #1")
#dev.off()



##------------------------------------------------------------------
## Test #5: Confirm initial selection with 10x CV
##------------------------------------------------------------------
##  - min logloss =
##------------------------------------------------------------------

## define paramter to test
tryParam        <- c(0.05)
tryParamList    <- list()
cv.nround       <- 1000
num.fold        <- 10

## placeholder for cross-validation results
tryParamTrainRes    <- matrix(, nrow=cv.nround, ncol=length(tryParam))
tryParamTestRes     <- matrix(, nrow=cv.nround, ncol=length(tryParam))
tryParamTrainErr    <- matrix(, nrow=cv.nround, ncol=length(tryParam))
tryParamTestErr     <- matrix(, nrow=cv.nround, ncol=length(tryParam))

## define the parameter list
for (i in 1:length(tryParam))
{
    
    tryParamList[[i]] <- list( "objective" = "multi:softprob",
                                "eval_metric" = "mlogloss",
                                "num_class" = 9,
                                "nthread" = 8,
                                "eta" = tryParam[i],
                                "max_depth" = 7,
                                "min_child_weight" = 1,
                                "subsample" = 0.9,
                                "colsample_bytree" = 0.9)
}

## loop over each variant and collect results
for (i in 1:length(tryParamList))
{
    bst.cv = xgb.cv(param=tryParamList[[i]], data = x[trind,], label = y, nfold = num.fold, nrounds=cv.nround, verbose=2)
    
    tryParamTrainRes[,i] <- as.numeric(bst.cv[,train.mlogloss.mean])
    tryParamTrainErr[,i] <- as.numeric(bst.cv[,train.mlogloss.stdn])
    tryParamTestRes[,i]  <- as.numeric(bst.cv[,test.mlogloss.mean])
    tryParamTestErr[,i]  <- as.numeric(bst.cv[,test.mlogloss.std])
}


##------------------------------------------------------------------
## Test #5: Test as #5 with re-scaled raw data
##------------------------------------------------------------------




