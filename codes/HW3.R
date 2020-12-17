
train = read.csv("train.csv",h=T)
test = read.csv("test.csv",h=T)
head(train)
dim(train)
dim(test)
sum(train$SeriousDlqin2yrs)
summary(train$SeriousDlqin2yrs)

### Visualization ###
library(corrplot)
corrplot(cor(train))

attach(train)
par(mfrow=c(ncol(train)%/%3 + 1, 3))
for (i in names(train)) {
	boxplot(get(i), main = i)
}

### Trim ###
y = train$SeriousDlqin2yrs
train$SeriousDlqin2yrs = NULL
testMat = as.matrix(test)
Id = 1:nrow(test)

### Logit ###
library(pROC)

logit  = glm(y ~ . ,  data = train, family = binomial, control = glm.control(maxit=500))
summary(logit)
pred = predict(logit, newdata = test, type = "response" )
# logit_roc = roc(response = y, predictor = pred, plot = T)
# write
logit_sub = data.frame(Id = Id, Probability = pred)
write.csv(logit_sub, "logit.csv", row.names = F)

### Naive Bayes ###
library(e1071)
nb = naiveBayes(y ~ ., data = train)
nbPred = predict(nb, newdata = test, type = "raw")[,2]
summary(nbPred)
# write
nb_sub = data.frame(Id = Id, Probability = nbPred)
write.csv(nb_sub, "nb.csv", row.names = F)

### Ridge ###
library(glmnet)

ridge  = cv.glmnet(x = as.matrix(train), y = as.matrix(y), type.measure = "auc", family = "binomial", alpha = 0)
ridgePred = predict(ridge, newx= testMat, s = ridge$lambda.min, type = "response")
summary(ridgePred)
# write
ridge_sub = data.frame(Id = Id, Probability = as.numeric(ridgePred))
write.csv(ridge_sub, "ridge.csv", row.names = F)

### Lasso ###
#library(glmnet)
lasso  = cv.glmnet(x = as.matrix(train), y = as.matrix(y), type.measure = "auc", family = "binomial", alpha = 1)
lassoPred = predict(lasso, newx = testMat, s = lasso$lambda.min, type = "response")
summary(lassoPred)
# write
lasso_sub = data.frame(Id = Id, Probability = as.numeric(lassoPred))
write.csv(lasso_sub, "lasso.csv", row.names = F)

### Tree ###
library(rpart)

tree = rpart(y ~ ., data = train)
summary(tree)
plot(tree)
text(tree)
treePred = predict(tree, newdata = test)
#Write
tree_sub = data.frame(Id = Id, Probability = treePred)
write.csv(tree_sub, "tree.csv", row.names = F)

### tree = rpart(as.factor(y) ~ ., data = train)
### treePred = predict(tree, newdata = test) # This gives a matrix of Prob. for No and Prob. for Yes.
### tree_sub = data.frame(Id = Id, Probability = treePred[,2]) # only use column for Yes.
### write.csv(tree_sub, "tree1.csv", row.names = F)

#########################################################

### Drop Outliers ###
library(outliers)
train = read.csv("train.csv",h=T)

train.original = train
#num.del = 0 # Num of obs removed
for (i in names(train)[-1]) {
	temp = outlier(get(i))
	train = train[-which(get(i) == temp), ]
	#num.del = num.del + length(temp)
}
dim(train) # 30 are deleted
summary(train)
y = train$SeriousDlqin2yrs
train$SeriousDlqin2yrs = NULL

### Logit with Variable Selection ###
library(bestglm)

trainFull = data.frame(train, y)
head(trainFull)
bestBack = bestglm(Xy = trainFull, family = binomial, method = "backward")
summary(bestBack)
logitBack = bestBack$BestModel

bestForw = bestglm(Xy = trainFull, family = binomial, method = "forward")
summary(bestForw)
bestForw$BestModel

logitBackPred = predict(logitBack, newdata = test, type = "response")
logitBack_sub = data.frame(Id = Id, Probability = logitBackPred)
write.csv(logitBack_sub, "logitBack2.csv", row.names = F)

# Interaction (cannot predict)
#XyInter = model.matrix(formula( ~ (.)^2), data = train)
#trainInter = data.frame(XyInter, y)
#logitInter = bestglm(Xy = trainInter, family = binomial, method = "backward")


### LASSO with Data Cleaning ###
library(glmnet)

lasso  = cv.glmnet(x = as.matrix(train), y = as.matrix(y), type.measure = "auc", family = "binomial", alpha = 1)
lassoPred = predict(lasso, newx = testMat, s = lasso$lambda.min, type = "response")
summary(lassoPred)
# write
lasso_sub = data.frame(Id = Id, Probability = as.numeric(lassoPred))
write.csv(lasso_sub, "lasso1.csv", row.names = F)


