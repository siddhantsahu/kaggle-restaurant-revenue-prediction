require(caret)
require(doParallel)
require(Boruta)

# read training and testing set
train <- read.csv("data/train.csv", stringsAsFactors=F)
test <- read.csv("data/test.csv", stringsAsFactors=F)

# combine test and training set for cleaning operations
test$revenue <- 1
data <- rbind(train, test)
n.train <- nrow(train)

# pre-processing data set
data$Open.Date <- strptime(data$Open.Date, "%m/%d/%Y")
data$City <- as.character(data$City)
data$City <- factor(data$City)
data$City.Group <- factor(data$City.Group)
data$Type[data$Type=="DT"] <- "IL"
data$Type[data$Type=="MB"] <- "FC"
data$Type <- factor(data$Type)
data$Age <- difftime(strptime("01/01/2015", format="%m/%d/%Y"), data$Open.Date, units="weeks")
data$Age <- as.numeric(data$Age)

rm(train, test)

# log transform revenue
data$log.revenue <- log(data$revenue)

# leave one core for the OS
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# run Boruta to select best features
important <- Boruta(log.revenue~., data=train[, c(-1, -2, -3, -43)])

# run Random Forests
set.seed(546)
model <- train(log.revenue~.,
               data=data[1:n.train, c(-1, -2, -3, -43)][, c(important$finalDecision != "Rejected", TRUE)],
               method="rf", tuneLength=7, nodesize=7)

prediction <- predict(model, data[-c(1:n.train), ])

# summary of prediction
summary(exp(prediction))
hist(exp(prediction))

# Make Submission
submit <- as.data.frame(cbind(seq(0, length(prediction) - 1, by=1), exp(prediction)))
colnames(submit) <- c("Id", "Prediction")
write.csv(submit, "submission.csv", row.names=FALSE, quote=FALSE)

stopCluster(cl)