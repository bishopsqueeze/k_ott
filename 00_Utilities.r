##------------------------------------------------------------------
## Process Raw Data Files
##------------------------------------------------------------------

##------------------------------------------------------------------
## Load libraries
##------------------------------------------------------------------

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

floorSub    <- function(mySub)
{
    if ( class(mySub) == "matrix" ) {
        return(pmax(pmin(mySub, (1 - 1e-15)), 1e-15))
    } else {
        stop("floorSub:: Submission file not a matrix")
    }
}
a <- floorSub(unifSub)


expandTarget <- function(myTarget)
{
    if (class(myTarget) == "factor") {
        myTarget   <- as.integer(myTarget)
    }
    
    nr  <- length(myTarget)
    nc  <- 9
    
    expTarget   <- matrix(0, nrow=nr, ncol=nc, dimnames=list(NULL, c(paste0("Class_",1:nc))))
    
    expTarget[cbind(1:nr,myTarget)] <- 1
    
    return(expTarget)
}
b <- expandTarget(testTarget)



##------------------------------------------------------------------
## <function> evalMetric
##------------------------------------------------------------------
evalMetric <- function(mySub, myTarget)
{
    return(-(1/nrow(mySub))*sum(expandTarget(myTarget)*log(floorSub(mySub))))
}

tmp1 <- evalMetric(unifSub, testTarget) ## res = 2.197225 (consistent with leaderboard)
tmp2 <- evalMetric(all2Sub, testTarget) ## res = 25.53987
tmp3 <- evalMetric(testSub, testTarget) ## res = 34.53878 (maximum value)


0.70*144368

## max value on leaderboard = 33.47073 ~ 2.90969e-15

