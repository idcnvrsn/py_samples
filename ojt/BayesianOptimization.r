if(0)
{
install.packages("rBayesianOptimization")

	install.packages("drat", repos="https://cran.rstudio.com")
	drat:::addRepo("dmlc")
	install.packages("mxnet")
	install.packages("glmnet")
	install.packages("e1071")
	install.packages("Metrics")
	install.packages("randomForest")
}

train <- read.csv('mnist/short_prac_train.csv')
test <- read.csv('mnist/short_prac_test.csv')
train$label <- as.factor(train$label)
test$label <- as.factor(test$label)

library(e1071)
library(rBayesianOptimization)

svm_holdout <- function(deg, c){
   model <- svm(label~., train, kernel='polynomial', degree=deg, coef0=c)
   t.pred <- predict(model, newdata=test[,-1])
   Pred <- sum(diag(table(test$label, t.pred)))/nrow(test)
   list(Score=Pred, Pred=Pred)
}

opt_svm <- BayesianOptimization(svm_holdout,
                               bounds=list(deg=c(3L,10L),
                                           c=c(0,10)),
                               init_points=10, n_iter=1, acq='ei', kappa=2.576,
                               eps=0.0, verbose=TRUE)


library(randomForest)
rf_holdout <- function(mnum){
     model <- randomForest(label~., train, mtry=mnum)
     t.pred <- predict(model, newdata=test[,-1])
     Pred <- sum(diag(table(test$label, t.pred)))/nrow(test)
     list(Score=Pred, Pred=Pred)
}
opt_rf <- BayesianOptimization(rf_holdout,
                                 bounds=list(mnum=c(3L,40L)),
                                 init_points=10, n_iter=1, acq='ei', kappa=2.576,
                                 eps=0.0, verbose=TRUE)

library(xgboost)
library(Matrix) # データの前処理に必要
train.mx<-sparse.model.matrix(label~., train)
test.mx<-sparse.model.matrix(label~., test)
# データセットをxgboostで扱える形式に直す
dtrain<-xgb.DMatrix(train.mx, label=as.integer(train$label)-1)
dtest<-xgb.DMatrix(test.mx, label=as.integer(test$label)-1)
xgb_holdout <- function(ex, mx, nr){
   model <- xgb.train(params=list(objective="multi:softmax", num_class=10,
                                   eval_metric="mlogloss",
                                   eta=ex/10, max_depth=mx),
                                   data=dtrain, nrounds=nr)
   t.pred <- predict(model, newdata=dtest)
   Pred <- sum(diag(table(test$label, t.pred)))/nrow(test)
   list(Score=Pred, Pred=Pred)
}
opt_xgb <- BayesianOptimization(xgb_holdout,
                               bounds=list(ex=c(2L,5L), mx=c(4L,5L), nr=c(70L,160L)),
                               init_points=20, n_iter=1, acq='ei', kappa=2.576,
                               eps=0.0, verbose=TRUE)

d <- read.csv('OnlineNewsPopularity.csv')
d <- d[,-c(1,2)]
idx <- sample(nrow(d),5000,replace=F)
d_train <- d[-idx,]
d_test <- d[idx,]
library(mxnet)
train <- data.matrix(d_train)
test <- data.matrix(d_test)
train.x <- train[,-59]
train.y <- train[,59]
test.x <- test[,-59]
test.y <- test[,59]
train_means <- apply(train.x, 2, mean)
train_stds <- apply(train.x, 2, sd)
test_means <- apply(test.x, 2, mean)
test_stds <- apply(test.x, 2, sd)
train.x <- t((t(train.x)-train_means)/train_stds)
test.x <- t((t(test.x)-test_means)/test_stds)
train.y <- log(train.y)
test.y <- log(test.y)
mxnet_holdout_bayes <- function(unit1, unit2, unit3, num_r, learn_r){
   data <- mx.symbol.Variable("data")
   fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=unit1)
   act1 <- mx.symbol.Activation(fc1, name="tanh1", act_type="tanh")
   drop1 <- mx.symbol.Dropout(act1, p=0.2)
   fc2 <- mx.symbol.FullyConnected(drop1, name="fc2", num_hidden=unit2)
   act2 <- mx.symbol.Activation(fc2, name="tanh2", act_type="tanh")
   drop2 <- mx.symbol.Dropout(act2, p=0.2)
   fc3 <- mx.symbol.FullyConnected(drop2, name="fc3", num_hidden=unit3)
   act3 <- mx.symbol.Activation(fc3, name="tanh3", act_type="tanh")
   drop3 <- mx.symbol.Dropout(act3, p=0.2)
   fc4 <- mx.symbol.FullyConnected(drop3, name="fc4", num_hidden=1)
   output <- mx.symbol.LinearRegressionOutput(fc4, name="linreg")
   devices <- mx.cpu()
   mx.set.seed(71)
   model <- mx.model.FeedForward.create(output, X=train.x, y=train.y, 
                                        ctx=devices, num.round=num_r, array.batch.size=200,
                                        learning.rate=learn_r, momentum=0.99,
                                        eval.metric=mx.metric.rmse,
                                        initializer=mx.init.uniform(0.5),
                                        array.layout = "rowmajor",
                                        epoch.end.callback=mx.callback.log.train.metric(20),
                                        verbose=FALSE)
   preds <- predict(model, test.x, array.layout='rowmajor')
   holdout_score <- rmse(preds, test.y)
   list(Score=-holdout_score, Pred=-holdout_score)
   # 「最大化」されてしまうので、最小化したい場合はメトリクスを負にする
}
library(Metrics)
library(rBayesianOptimization)
opt_res <- BayesianOptimization(mxnet_holdout_bayes,
                               bounds=list(unit1=c(200L,500L),
                                           unit2=c(10L,80L),
                                           unit3=c(10L,50L),
                                           num_r=c(80L,150L),
                                           learn_r=c(1e-5,1e-3)),
                               init_points=20, n_iter=1, acq='ei',
                               kappa=2.576, eps=0.0, verbose=TRUE)
