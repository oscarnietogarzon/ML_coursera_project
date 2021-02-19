#Course Project
pks <- c("ElemStatLearn","caret","gbm", "lubridate", "forecast","tidyverse","randomForest","plot.matrix") #load packages
lapply(pks, require, character.only = TRUE)

#Data
UrlTrain <- c("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
UrlTest <- c("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
#Download
download.file(UrlTrain, destfile = "/Users/oscarnietogarzon/Desktop/R Projects/ML_coursera_project/TrainHAR.csv")
download.file(UrlTest, destfile = "/Users/oscarnietogarzon/Desktop/R Projects/ML_coursera_project/TestHAR.csv")

TrainHAR <- read.csv("./TrainHAR.csv")
TestHAR <- read.csv("./TestHAR.csv")

#Predict the manner in which they did the exercise (Classe variable)
TrainHAR$classe<-as.factor(TrainHAR$classe)
#Crossvalidation
set.seed(12345)
createDataPartition(TrainHAR$classe, p=0.7, list=FALSE) -> inCV

TrainHAR[inCV, ] -> trainCV
TrainHAR[-inCV, ] -> valCV
grep("_belt", colnames(trainCV), value=TRUE)
#selected variables to predict in the train set
subset(trainCV, select = c("classe","roll_belt", "accel_belt_x",  "accel_belt_y" ,
  "accel_belt_z", "gyros_belt_x","gyros_belt_y","gyros_belt_z" ,
  "magnet_belt_x", "magnet_belt_y", "magnet_belt_z" ,
  "accel_arm_x","accel_arm_y", "accel_arm_z",
  "magnet_arm_x","magnet_arm_y","magnet_arm_z",
  "accel_dumbbell_x","accel_dumbbell_y","accel_dumbbell_z","magnet_dumbbell_x", "magnet_dumbbell_y","magnet_dumbbell_z",
  "gyros_dumbbell_x" ,"gyros_dumbbell_y","gyros_dumbbell_z",
  "pitch_forearm", "gyros_forearm_x" ,"gyros_forearm_y" , "gyros_forearm_z" ) ) -> trainCV

#selected variables to predict in the test set
subset(valCV, select = c("classe","roll_belt", "accel_belt_x",  "accel_belt_y" ,
                           "accel_belt_z", "gyros_belt_x","gyros_belt_y","gyros_belt_z" ,
                           "magnet_belt_x", "magnet_belt_y", "magnet_belt_z" ,
                           "accel_arm_x","accel_arm_y", "accel_arm_z",
                           "magnet_arm_x","magnet_arm_y","magnet_arm_z",
                           "accel_dumbbell_x","accel_dumbbell_y","accel_dumbbell_z","magnet_dumbbell_x", "magnet_dumbbell_y","magnet_dumbbell_z",
                           "gyros_dumbbell_x" ,"gyros_dumbbell_y","gyros_dumbbell_z",
                           "pitch_forearm", "gyros_forearm_x" ,"gyros_forearm_y" , "gyros_forearm_z" ) ) -> valCV

#models to evaluate
#random forest
randomForest::randomForest(classe~., data=trainCV, ntree=100) -> fit_rf
#boosting with trees (gbm)
train(classe~., data=trainCV, method="gbm", verbose=FALSE) -> fit_gbm
#varImp
order(varImp(fit_rf), decreasing = TRUE) 
varImpPlot(fit_rf)
varImp(fit_gbm)

predict(fit_rf, newdata=valCV) -> pred_rf
predict(fit_gbm, newdata=valCV) -> pred_gbm
#Accurancy
confusionMatrix(pred_rf, valCV$classe)$overall['Accuracy']   #0.9862
fit_rf$err.rate[,1]
plot(fit_rf$err.rate[,1], type = "l")

confusionMatrix(pred_gbm, valCV$classe)$overall['Accuracy']  #0.9233
confusionMatrix(pred_rf, pred_gbm)$overall['Accuracy']       #0.9322
#table
table(pred_rf, valCV$classe) #Random Forest
prop.table(table(pred_rf, valCV$classe), 2) -> rfmatrix  #normalized matrix
matrix(rfmatrix, ncol=5) -> rfmatrix
colnames(rfmatrix) <- c("A","B","C","D","E"); rownames(rfmatrix) <- c("A","B","C","D","E")
plot(rfmatrix,digits=3, cex=0.65, ylab ='', xlab='')

sqrt(mean( (pred_rf-valCV$classe)^2 ))

table(pred_gbm, valCV$classe) #GBM
table(pred_gbm)

prop.table(table(pred_gbm, valCV$classe), 2) #normalized matrix
#Prediction with the rf model with the real test data
predict(fit_rf, newdata=TestHAR)

save.image(file = "projectData.RData")
