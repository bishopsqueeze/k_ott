##------------------------------------------------------------------
## Process Raw Data Files
##------------------------------------------------------------------

##------------------------------------------------------------------
## Load libraries
##------------------------------------------------------------------
require(caret)
require(randomForest)
library(car)

##------------------------------------------------------------------
## Define the parallel flag
##------------------------------------------------------------------
DOPARALLEL  <- TRUE

##------------------------------------------------------------------
## Register the clusters
##------------------------------------------------------------------
if (DOPARALLEL) {
    library(foreach)
    library(doMC)
    registerDoMC(3)
}

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
## Temporary utility functions
##------------------------------------------------------------------

## log loss for <caret>
logLoss <- function(data, lev = NULL, model = NULL)
{
    eps         <- 1e-10
    nr          <- nrow(data)
    
    tmpObs      <- as.integer(gsub("L", "", data[,c("obs")]))
    tmpProbs    <- data[,c(paste0("L",1:9))]
    
    tmpProbs[tmpProbs < eps]     <- eps
    tmpProbs[tmpProbs > 1 - eps] <- 1 - eps
    
    matObs                      <- matrix(0, nrow=nr, ncol=9, dimnames=list(NULL, c(paste0("L",1:9))))
    matObs[cbind(1:nr, tmpObs)] <- 1
    
    ll          <- sum(matObs*tmpProbs)/nr
    nll         <- -ll
    out         <- c(ll, nll)
    names(out)  <- c("logloss", "neglogloss")
    out
}

## general los loss
evalMetric <- function(myPred, myTarget)
{
    nc  <- 9
    
    if (class(myPred) %in% c("matrix", "data.frame")) {
        
        nr          <- nrow(myPred)
        tmpTarget   <- as.numeric(gsub("L", "", myTarget))
        tmpPred     <- as.matrix(myPred)
        
        return(-sum(expandTarget.cmp(tmpTarget)*log(floorMatrix.cmp(tmpPred)))/nr)
        
        
    } else if (class(myPred) == "factor") {
        
        nr          <- length(myPred)
        tmpTarget   <- as.numeric(gsub("L", "", myTarget))
        tmpPred     <- as.numeric(gsub("L", "", myPred))
        
        return(-sum(expandTarget.cmp(tmpTarget)*log(floorMatrix.cmp(expandTarget.cmp(tmpPred))))/nr)
        
    } else {
        return(-999)
    }
    
}

#tmp2 <- predict(gbmFit1, newdata=train.os, type="prob")evalMetric(tmp1, class.os)
#evalMetric(tmp2, class.os)

##------------------------------------------------------------------
## Simple tests of various datasets using a vanilla GBM
##------------------------------------------------------------------

## define a split index
#idx7030         <- createDataPartition(train.target, p=0.7, list=FALSE, times=1)

##------------------------------------------------------------------
## Test #1 :: Use the vanilla features (logloss = 0.6338822 +- 0.002349236)
##------------------------------------------------------------------

train           <- cbind(train.feat)
class           <- train.target
#train.is        <- train[ idx7030, ]
#train.os        <- train[-idx7030, ]
#class.is        <- train.target[ idx7030]
#class.os        <- train.target[-idx7030]
#class.os.mat    <- expandTarget.cmp(class.os)

fitControl <- trainControl( method = "cv",
                            number = 5,
                            classProbs = TRUE,
                            allowParallel = TRUE,
                            summaryFunction = logLoss)


rfGrid  <- expand.grid(mtry = 2:10)


set.seed(4321)
rfFit1 <- train(    x=train,
                    y=class,
                    method = "rf",
                    tuneGrid=rfGrid,
                    trControl = fitControl,
                    metric = "neglogloss",
                    ## rf params
                    ntree=500,
                    importance=TRUE)


#tmp.summary <- list()
#tmp.summary <- foreach (h=1:length(unq.h)) %dopar% {
#tmp.summary[[h]] <- data.frame(
#)
#}
