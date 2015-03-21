##------------------------------------------------------------------
## Process Raw Data Files
##------------------------------------------------------------------

##------------------------------------------------------------------
## Load libraries
##------------------------------------------------------------------
library(compiler)

##------------------------------------------------------------------
## Set the working directory
##------------------------------------------------------------------
setwd("/Users/alexstephens/Development/kaggle/otto/data/raw")


preProcessTrain <- function(myTrainFile) {
    load(paste0(getwd(),"/01_trainRaw.Rdata")
    
    trainId     <- train.raw[,1]
    train       <- data.frame(t=as.factor(gsub("Class_", "", train.raw[,95])), train.raw[,2:94])
    
    
    
}



testSub     <- matrix(0, nrow=length(trainId), ncol=9, dimnames=list(NULL, c(paste0("Class_",1:9))))
testTarget  <- as.integer(train[,c("t")])
unifSub     <- (1/9) + testSub
all2Sub     <- testSub
all2Sub[,2] <- 1



floorMatrix <- function(myPred, eps=1e-10)
{
    return(pmax(pmin(myPred, 1-eps), eps))
}
##a <- floorSub(unifSub)
floorMatrix.cmp <- cmpfun(floorMatrix)


expandTarget <- function(myTarget)
{
    nr          <- length(myTarget)
    tmpTarget   <- as.numeric(gsub("L", "", myTarget))
    expTarget   <- matrix(0, nrow=nr, ncol=9, dimnames=list(NULL, c(paste0("L",1:9))))
    
    expTarget[cbind(1:nr, tmpTarget)] <- 1
    
    return(expTarget)
}
expandTarget.cmp <- cmpfun(expandTarget)
##b <- expandTarget(testTarget)



##------------------------------------------------------------------
## <function> evalMetric
##------------------------------------------------------------------
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




tmp1 <- evalMetric(unifSub, testTarget) ## res = 2.197225 (consistent with leaderboard)
tmp2 <- evalMetric(all2Sub, testTarget) ## res = 25.53987
tmp3 <- evalMetric(testSub, testTarget) ## res = 34.53878 (maximum value)


0.70*144368

## max value on leaderboard = 33.47073 ~ 2.90969e-15

