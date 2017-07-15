# Set a seed for reproducibility
set.seed(0)
library(waveslim)
library(data.table)
# Load the libs required for the analysis
library(class)
library(pls)

saveRDS(alldata,file="/home/josh/Desktop/MINST_data/32by32/alldata.rds")
alldata <- readRDS(file="/home/josh/Desktop/MINST_data/32by32/alldata.rds")
browseURL("https://s3.amazonaws.com/assets.datacamp.com/img/blog/data+table+cheat+sheet.pdf")
# setwd("/home/josh/R/data/MINST")
setwd("/home/josh/Desktop/MINST_data/32by32")

# Load the training and test datasets
#train <- read.csv("./train.csv")
#test <- read.csv("./test.csv")
LoadOriginalData <- function() {
  train <- matrix(readBin(con="train-images.idx3-ubyte.raw", "numeric", n = 32*32*60000, size = 4, endian = "little"),
                  nrow= 60000, ncol = 32*32, byrow = TRUE)  
  
  test <- matrix(readBin(con="t10k-images.idx3-ubyte.raw", "numeric", n = 32*32*10000, size = 4, endian = "little"),
                 nrow = 10000,  ncol = 32*32, byrow = TRUE)  
  
  trainlabel <- matrix(readBin(con="train-labels.idx1-ubyte_8bit.raw", "integer", n = 60000, size = 1, endian = "little"),
                       nrow= 1, ncol = 60000, byrow = TRUE) 
  testlabel <- matrix(readBin(con="t10k-labels.idx1-ubyte_8bit.raw", "integer", n = 10000, size = 1, endian = "little"),
                      nrow= 1, ncol = 10000, byrow = TRUE) 
  # print(image(test_1))
  # For training data
  # Convert to lists
  train <- t(train)
  train_df <- lapply(seq_len(ncol(train)), function(i) train[,i])
  trainlabel_df <- lapply(seq_len(ncol(trainlabel)), function(i) trainlabel[,i])
  # convert lists to data table
  trainlabel_vect <- as.vector(trainlabel)
  train <- data.table(train_df,trainlabel_vect,"train")
  names(train) <- c("image","label","train_or_test")
  setkey(train,label)
  ## For test data
  # Convert to lists
  test <- t(test)
  test_df <- lapply(seq_len(ncol(test)), function(i) test[,i])
  testlabel_vect <- as.vector(testlabel)
  # convert lists to data table
  test <- data.table(test_df,testlabel_vect,"test")
  names(test) <- c("image","label","train_or_test")
  setkey(test,label)
  
  # Make one big Data table!
  # combine all rows
  #test_0_v2 <- alldata2[train_or_test=="train",label==1] # this was really slow (when label was a list)
  alldata <- rbind(test,train)
  setkey(alldata,label,train_or_test)
}

tables()
###  we can dynamically add and subtract columns using ":=" ###
# Test using a column added so we can count the number of rows
alldata[, num := 1L]  # Add a column of 1's
sum(alldata[train, sum(num), by=label]$V1)  # should equal 60000 - the V1 variable is the return column that we are suming
## This is the standard way to count though - using .N  notation
alldata[train_or_test=="train", .N, by=label] # count number of each label in training set using the .N inbuilt
######
label    N
1:     0 5923
2:     1 6742
3:     2 5958
4:     3 6131
5:     4 5842
6:     5 5421
7:     6 5918
8:     7 6265
9:     8 5851
10:     9 5949
######
sum(alldata[train_or_test=="train", .N, by=label]$N)  # should equal 60000 - count number of each label in training set using the .N inbuilt
alldata[, num := NULL] # remove that column

alldata[,.N, by=train_or_test]  # counts how many rows
tables()  # list all the data.tables in memory

alldata[train_or_test=="test", .N, by=label]
#####
label    N
1:     0  980
2:     1 1135
3:     2 1032
4:     3 1010
5:     4  982
6:     5  892
7:     6  958
8:     7 1028
9:     8  974
10:     9 1009
####


# FFT in 2D
test_fft_2D <- alldata[1, image]
head(test_fft_2D)
test_fft_2D_fft <- fft(matrix(test_fft_2D[[1]],ncol=32,nrow=32))
print(image(abs(test_fft_2D_fft)))  # displa power spectrum

# FFT in 1D
test_fft_1D <- alldata[label==0L &train_or_test=="train", image]
head(test_fft_1D)
test_fft_1D_fft <-  lapply(test_fft_1D,fft)
test_fft_1D_fft_matrix <- do.call(rbind,test_fft_1D_fft)


# SVD of FFT
test_fft_1D_fft_matrix_svd <- svd(test_fft_1D_fft_matrix)
plot(test_fft_1D_fft_matrix_svd$d)
d_cpy <- test_fft_1D_fft_matrix_svd$d
d_cpy[20:length(d_cpy)] = 0.0 
D = diag(d_cpy)
result <- test_fft_1D_fft_matrix_svd$u %*% D %*% t(test_fft_1D_fft_matrix_svd$v)
res_invfft <- mvfft(t(result),inverse=TRUE)
print(image(matrix(abs(res_invfft[1,]),ncol=32,nrow=32)))


# FUNCTION TO NORMALISE THE LENGTH OF INPUT VECTOR
vnorm <- function(v1) {
  length <- sqrt(t(v1)%*%(v1))
  v1_norm <- v1 / length
}

# Function to scale by the max range and remove offset
scale_to_one <- function(matrix_in) {
  mat_min <- min(matrix_in)
  mat_max <- max(matrix_in)
  scaled_in <-(matrix_in-mat_min)/(mat_max-mat_min)
}

# Function to force anything under limit value to 0.0 and over limit to 1.0
force_to_one <- function(matrix_in, limit= 0.01) {
  #res <- ifelse(matrix_in > limit, 1.0, 0.0)
  res <- rep(0.0,length(matrix_in))
  res[matrix_in > limit] <- 1.0
  return(res)
}

# Function to return the power spectrum of the fft of input
fft_power <- function(matrix_in) {
  matrix_in <- fft(matrix_in)
  matrix_in <- abs(matrix_in)
  matrix_in <- vnorm(matrix_in)
  return(matrix_in)
}

row_or_col_ave <- function(vector_in, row_or_col) {
  mat_res <- matrix(vector_in,ncol=32,nrow=32)
  mat_res <- apply(mat_res,row_or_col,sum)
  return(mat_res)
}

join_vect <- function(vector_1, vector_2) {
  mat_res <- c(vector_1,vector_2)
  return(mat_res)
}

# alldata[, fft:=NULL]
# apply FFT the whole set of image vectors (1D)
# alldata[, image:=lapply(image,vnorm)]
# Add a column of FFT power spectrum versions of the data to the alldata DT
alldata[, fft_lab:=lapply(image,fft_power)]
# Compute the row and column averages 
alldata[, row_ave:=lapply(image,row_or_col_ave,row_or_col=1)]
alldata[, col_ave:=lapply(image,row_or_col_ave,row_or_col=2)]
#alldata[, row_col_ave:=.(row_ave,col_ave)]
#alldata[, row_col_ave:=NULL]
#alldata[1, row_col_ave][[1]]
# tables()
# test_fft_1D <- abs(alldata[2000, fft_lab][[1]])
# plot(test_fft_1D)

r1 <- alldata[20000,row_ave][[1]]
r2 <- alldata[20000,col_ave][[1]]
r1r2 <- c(r1,r2)

alldata[20000,row_col_ave][[1]]

ave_results_vn <- list()
input_column <-  "image"
# This just uses the average of the data as a model for prediction of class
for (i in 0L:9L) {
  train_1D <- alldata[label==as.integer(i) & train_or_test=="train", eval(parse(text=input_column))]
  # train_1D <- lapply(train_1D,vnorm)
  # train_1D <- lapply(train_1D,force_to_one)
  train_1D <- lapply(train_1D,vnorm)
  train_1D_matrix <- do.call(rbind,train_1D)
  v1_ave_v2 <- apply(train_1D_matrix,2,mean)
  v1_ave_v2_norm <- vnorm(v1_ave_v2)
  ave_results_vn[[paste0("ave_",i)]] <- v1_ave_v2_norm
}


######################################################
# PLSR all labels as factor version (v1)
# get training data set
input_column <-  "image"
train_1D <- alldata[train_or_test=="train", eval(parse(text=input_column))]
train_1D <- lapply(train_1D,vnorm)
train_1D_matrix <- do.call(rbind,train_1D)
train_1D_lab <- alldata[train_or_test=="train", label]
train_DF <- data.frame(label=train_1D_lab, train=I(train_1D_matrix))
trained <- plsr(label ~ train, data=train_DF, ncomp=10)

saveRDS(trained, file="/home/josh/Desktop/MINST_data/32by32/trained_plsr_image_30pc.rds") # save model
trained <- readRDS( file="/home/josh/Desktop/MINST_data/32by32/trained_plsr_image.rds") # save model
# get test data 
test_1D <- alldata[train_or_test=="test", eval(parse(text=input_column))]
test_1D <- lapply(test_1D,vnorm)
test_1D_matrix <- do.call(rbind,test_1D)
test_1D_lab <- alldata[train_or_test=="test", label]
test_DF <- data.frame(label=test_1D_lab, train=I(test_1D_matrix))

reval <- alldata[train_or_test=="test", (.N), by=label] # print number of each label
reval_cuml <- cumsum(reval)
i <- 2
# Calculate the average predicted value or print the histogram 
# of values predictedd for each label class.
ncomp <- 7  # the number of compenents to iclude for the prediction model
start <- 1
res_list <- list()
for (i in 1:10) {
  end <- reval_cuml[["V1"]][i]
  num <- reval[["V1"]][i]
  # res_list[paste(i)] <- sum(predict(trained, ncomp = ncomp, newdata = test_DF[start:end,])) / num
  hist(predict(trained, ncomp = ncomp, newdata = test_DF[start:end,]), breaks=20, main=paste("Label:",i), xlim=c(-5,10))
  start <- end + 1
}
###################################################### 



input_column <-  "row_ave"  # other option is fft_lab or image  row_ave  col_ave
train_1D_rave <- alldata[ eval(parse(text=input_column))]
input_column <-  "col_ave"  # other option is fft_lab or image  row_ave  col_ave
train_1D_cave <- alldata[ eval(parse(text=input_column))]

vres <-mapply(c, train_1D_rave, train_1D_cave, SIMPLIFY=FALSE) # if SIMPLIFY == TRUE then an array would be returned
alldata[,row_col_ave:=vres]  alldata[,row_col_ave:=NULL]
train_1D <- lapply(vres,vnorm)
train_1D_matrix <- do.call(rbind,train_1D)
# str(train_1D_matrix)

######################################################
# PLSR Model of individual labels (as 1 / 0)
# get training data set
input_column <-  "row_ave"  # other option is fft_lab or image  row_ave  col_ave
train_1D <- alldata[train_or_test=="train", eval(parse(text=input_column))]
# train_1D <- lapply(train_1D,scale_to_one)
# train_1D <- lapply(train_1D,force_to_one)
train_1D <- lapply(train_1D,vnorm)
train_1D_matrix <- do.call(rbind,train_1D)
i <- 0
train_1D_lab <- alldata[train_or_test=="train", label]
trained_list <- list() 
for (i in 0:9) {
  # train_1D_lab <- alldata[train_or_test=="train", label]
  label_binary <-  ifelse(train_1D_lab == i, 1, -1)
  # sum(label_binary)
  train_DF <- data.frame(label=label_binary, train=I(train_1D_matrix))
  #trained_list[[paste(i)]] <- plsr(label ~ train, data=train_DF, ncomp=10)
  trained <- plsr(label ~ train, data=train_DF, ncomp=10)
  saveRDS(trained, file=paste0("/home/josh/Desktop/MINST_data/32by32/trained_plsr_label_",i,"col_ave_10pc.rds")) 
  rm(trained)
  gc()
}

# get test data 
test_1D <- alldata[train_or_test=="test", eval(parse(text=input_column))]
#train_1D <- lapply(train_1D,force_to_one)
test_1D <- lapply(test_1D,vnorm)
test_1D_matrix <- do.call(rbind,test_1D)
reval <- alldata[train_or_test=="test", (.N), by=label] # print number of each label
test_1D_lab <- alldata[train_or_test=="test", label]
test_DF <- data.frame(label=test_1D_lab, train=I(test_1D_matrix))
# Calculate the average predicted value or print the histogram 
# of values predictedd for each label class.
ncomp <- 10 # the number of compenents to iclude for the prediction model
res_list <- list()
for (i in 0:9) {
  num <- reval[["V1"]][i+1]  # used to get the ratio of correct predictions (this is the total number of a particular label)
  trained <- readRDS(file=paste0("/home/josh/Desktop/MINST_data/32by32/trained_plsr_label_",i,"row_ave_10pc.rds")) 
  # res_list[paste(i)] <- sum(predict(trained, ncomp = ncomp, newdata = test_DF[start:end,])) / num
  # hist(predict(trained, ncomp = ncomp, newdata = test_DF), breaks=20, main=paste("Label:",i), xlim=c(-2,2))
  preds <- predict(trained, ncomp = ncomp, newdata = test_DF)
  pred_label <- preds > 0.0
  hist(preds, breaks=20, main=paste("Label:",i), xlim=c(-2,2))
  test_label_vect <- test_1D_lab == i
  res <- ifelse((pred_label==test_label_vect) & (test_1D_lab==i),1,0)
  res_list[paste(i)] <- sum(res/num)  # sum the cases where the prediction is in the same position as the label of interest
  rm(trained)
  gc()
}

### using image data force_to_one and vnorm  model: trained_plsr_label_[0-9]_10pc.rds
0 0.8551020
1 0.8995595
2 0.6027132
3 0.5336634
4 0.6985743
5 0.4428251
6 0.7881002
7 0.7538911
8 0.2854209
9 0.4142716
###
# Using fft_ data and vnorm model: trained_plsr_label_[0-9]fft_10pc.rds
0 0.761224490
1 0.971806167
2 0.328488372
3 0.330693069
4 0.340122200
5 0.470852018
6 0.124217119
7 0.363813230
8 0.351129363
9 0.002973241

res_vect <- do.call(rbind,res_list)
num <- reval[["V1"]][i]

# row_ave_10pc.rds
0 0.235714286
1 0.584140969
2 0.000000000
3 0.000000000
4 0.006109980
5 0.003363229
6 0.000000000
7 0.008754864
8 0.000000000
9 0.000000000

# col_ave_10pc.rds
0 0.00000000
1 0.01409692
2 0.37015504
3 0.03366337
4 0.27902240
5 0.01008969
6 0.55845511
7 0.61770428
8 0.00000000
9 0.08126858

################################################################################




# preprocess
# train_1D <- lapply(train_1D,vnorm)
# train_1D <- lapply(train_1D,force_to_one)
train_1D <- lapply(train_1D,vnorm)

# Create into a matrix
train_1D_matrix <- do.call(rbind,train_1D)
# preprocess more - remove VN'd mean
# train_1D_matrix_scaled <-scale(train_1D_matrix,center=TRUE,scale=FALSE)
# Do svd on training images
train_1D_matrix_svd <- svd(train_1D_matrix)
plot(train_1D_matrix_svd$d)
# N.B. U matrix are the scores and V matrix are the vectors
twohundred_scores_u <- train_1D_matrix_svd$u[, 1:400]
str(twohundred_scores_u)
twohundred_scores_u_scaled <-scale(twohundred_scores_u,center=TRUE,scale=TRUE)


train_1D_0label <- alldata[train_or_test=="train", label]
train_1D_0label <- ifelse(train_1D_0label == 1, 5,-1)
lm_400 <- lm(train_1D_0label ~ twohundred_scores_u_scaled[,1:200])
plot(lm_400)

summary(lm_400)
#plot(train_1D_matrix_scaled_svd$u[, 200], train_1D_matrix_scaled_svd$u[, 70], main = "SVD", xlab = "U1", ylab = "U2")
# d_cpy <- train_1D_matrix_svd$d 
# d_cpy[20:length(d_cpy)] = 0.0 
# D = diag(d_cpy)
# result_1D <- train_1D_matrix_svd$u %*% D %*% t(train_1D_matrix_svd$v)
# str(result_1D)
# vectors <- D %*% t(train_1D_matrix_svd$v)
#  pca_1D <- princomp(train_1D_matrix)
#  print(image(pca_1D$loadings))
#  str(pca_1D$loadings)
#  min(pca_1D$loadings)
#  v1_ave <- apply(pca_1D$loadings,2, mean)
#  v1_norm <- vnorm(v1_ave)


ave_results_vn <- list()
input_column <-  "image"
for (i in 0L:9L) {
  # SVD of stretched vectors and then reconstitute from to 20 vectors  
  # can use image of fft_lab as inputs
  # train_1D <- alldata[label==as.integer(i) & train_or_test=="train", eval(parse(text=input_column))]
  
  
  v1_ave_v2 <- apply(train_1D_matrix,2,mean)
  v1_ave_v2_norm <- vnorm(v1_ave_v2)
  # plot(v1_ave_v2)
  # plot(v1_norm)
  ave_results_vn[[paste0("ave_",i)]] <- v1_ave_v2_norm
}



ave_results_matrix <- do.call(rbind,ave_results_vn)
# get all test data and their labels
test_1D <- alldata[train_or_test=="test", eval(parse(text=input_column))]
test_1D_label <- alldata[train_or_test=="test", label]
## test_1D[,image:=vnorm(image)]
train_1D <- lapply(train_1D,force_to_one)
test_1D <- lapply(test_1D,vnorm)
# test_1D_matrix <- do.call(rbind,test_1D)
# str(test_1D_matrix)
# res_mod_apply <-  test_1D_matrix  %*% v1_ave_v2_norm

# create a data table out of resulting prediction
# test_1D_normed_dt <- data.table(image = test_1D, label = test_1D_label)
# head(test_1D_normed_dt)
# im1 <- test_1D_normed_dt[1,image]

# res <- sweep(ave_results_matrix,MARGIN=2,im1,'*')
get_model_best_guess <- function(x,model_mat) {
  res <-  model_mat %*% diag(x,)
  res_v <- apply(res,1,sum)
  return(which.max(res_v)-1)
}
test_1D_pred <- lapply(test_1D, get_model_best_guess, model_mat=ave_results_matrix)
# add the results to a data table
test_dt_vn <- data.table(ave_results_vn = test_1D_pred, label = test_1D_label)
# fft_predictions_accuracy <- test_dt_01[ ,.(fft_mean = sum(mean_predict_fft==label)/(sum(mean_predict_fft!=label)+sum(mean_predict_fft==label))), by=label]
# predictions_accuracy_01 <- test_dt_01[ ,.(a01_mean = sum(mean_predict_01==label)/(sum(mean_predict_01!=label)+sum(mean_predict_01==label))), by=label]
predictions_accuracy_vn <- test_dt_vn[ ,.(vn_mean = sum(ave_results_vn==label)/(sum(ave_results_vn!=label)+sum(ave_results_vn==label))), by=label]
prediction_accuracy_list <- c(predictions_accuracy_vn,predictions_accuracy_01,fft_predictions_accuracy)
saveRDS(prediction_accuracy_list,file="prediction_accuracy.rds")

#head(test_1D_normed_dt,20)



##########################################
# do some wavelet analysis
library(waveslim)
test_2 <- t(matrix(test[4,],ncol=32,nrow=32))
print(image(test_2))
xbox.dwt <- dwt.2d(test_2, "haar", 4)

plot.dwt.2d(xbox.dwt)

xbox.dwt <- dwt(test[4,], "haar", n.levels=4)
len <- 8
wid <- len
print(image(matrix(xbox.dwt$s4,ncol=len,nrow=wid)))
plot(xbox.dwt$s4)

##########################################


##########################################
# Original KNN vs RandomForest comparison:
##########################################

train <- alldata[train_or_test=="train", image]
train <- do.call(rbind,train)
trainlabel <- as.vector(alldata[train_or_test=="train", label])
test <- alldata[train_or_test=="test", image]
test <- do.call(rbind,test)
testlabel <- as.vector(alldata[train_or_test=="test", label])

# Validate the loadingrn
cat(sprintf("Training set has %d rows and %d columns\n", nrow(train), ncol(train)))
cat(sprintf("Test set has %d rows and %d columns\n", nrow(test), ncol(test)))

# Extract the training set labels
#trainlabel<-train[,1]
# testlabel<-test[,1]




train.x <- scale_to_one(train)
train.c<-scale(train.x,center=TRUE,scale=FALSE)

str(train.x)
# Identify the feature means for the training data
# This can then be used to center the validation data
trainMeans<-colMeans(train.x)
trainMeansMatrix<-do.call("rbind",replicate(nrow(test),trainMeans,simplif=FALSE))
str(trainMeansMatrix)
# Generate a covariance matrix for the centered training data
train.cov<-cov(train.x)

# Run a principal component analysis using the training correlation matrix
pca.train<-prcomp(train.cov)

# Identify the amount of variance explained by the PCs
varEx<-as.data.frame(pca.train$sdev^2/sum(pca.train$sdev^2))
str(varEx)
varEx<-cbind(c(1:ncol(train.cov)),cumsum(varEx[,1]))
colnames(varEx)<-c("Nmbr PCs","Cum Var")
VarianceExplanation<-varEx[seq(0,200,20),]

# Because we can capture 95+% of the variation in the training data
# using only the first 20 PCs, we extract these for use in the KNN classifier
rotate<-pca.train$rotation[,1:20]

# Create the loading matrix based on the original training data
# This is the dimension reduction phase where we take the data
# matrix with 784 cols and convert it to a matrix with only 20 cols
trainFinal<-as.matrix(train.c)%*%(rotate)

# We then create a loading matrix for the testing data after applying
# the same centering and scaling convention as we did for training set
# test.x<-test/255
test.x<- scale_to_one(test)
testFinal<-as.matrix(test.x-trainMeansMatrix)%*%(rotate)

# Run the KNN predictor on the dim reduced datasets
predict<-knn(train=trainFinal,test=testFinal,cl=trainlabel,k=10)

##########
#RUN A RANDOM FOREST BENCHMARK FOR COMPARISON
library(randomForest)
set.seed(0)

numTrain <- 60000
numTrees <- 25
rows <- sample(1:nrow(train), numTrain)
labels <- as.factor(trainlabel)
# train <- train[rows,-1]

rf <- randomForest(train, labels, xtest=test, ntree=numTrees)
RF_predictions <- data.frame(ImageId=1:nrow(test), Label=levels(labels)[rf$test$predicted])
##########

# Create a data table for comparison of the random
# forest and PCA+KNN predictors
res_dt <- data.table(RF = RF_predictions$Label, knn = predict, label =testlabel)
setkey(res_dt,label)
# Summarise % of correct for RF and knn
res_dt[ ,sum(RF==label)/(sum(RF!=label)+sum(RF==label)), by=label]
res_dt[ ,sum(knn==label)/(sum(knn!=label)+sum(knn==label)), by=label]




# Output the results
cat("The PCA+KNN results are: ",sum(results)/length(results)," of the random forest results.","\n")

PCA_KNN_Predictions<-data.frame(ImageID=1:nrow(test),Label=predict)
#write.csv(PCA_KNN_Predictions,file="PCA_KNN_Predictions.csv",row.names=FALSE)