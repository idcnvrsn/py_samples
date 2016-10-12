if(0)
{
	install.packages("drat", repos="https://cran.rstudio.com")
	drat:::addRepo("dmlc")
	install.packages("mxnet")
	install.packages("glmnet")
	install.packages("e1071")
	install.packages("Metrics")
	install.packages("randomForest")
}

d <- read.csv('OnlineNewsPopularity.csv')
d <- d[,-c(1,2)]
idx <- sample(nrow(d),5000,replace=F)
d_train <- d[-idx,]
d_test <- d[idx,]

library(glmnet)
d_train.glmnet <- cv.glmnet(as.matrix(d_train[,-59]),log(d_train[,59]),family='gaussian',alpha=1)
plot(d_train.glmnet)
plot(log(d_test$shares), predict(d_train.glmnet, newx=as.matrix(d_test[,-59]), s=d_train.glmnet$lambda.min))
library(Metrics)
rmse(log(d_test$shares), predict(d_train.glmnet, newx=as.matrix(d_test[,-59]), s=d_train.glmnet$lambda.min))

library(randomForest)
d_train.rf <- randomForest(log(shares)~.,d_train)
rmse(log(d_test$shares), predict(d_train.rf,newdata=d_test[,-59]))

# フォーマッティング
library(mxnet)
train <- data.matrix(d_train)
test <- data.matrix(d_test)
train.x <- train[,-59]
train.y <- train[,59]
test.x <- test[,-59]
test.y <- test[,59]
# 正規化
train_means <- apply(train.x, 2, mean)
train_stds <- apply(train.x, 2, sd)
test_means <- apply(test.x, 2, mean)
test_stds <- apply(test.x, 2, sd)
train.x <- t((t(train.x)-train_means)/train_stds)
test.x <- t((t(test.x)-test_means)/test_stds)

# 目的変数を対数変換
train.y <- log(train.y)
test.y <- log(test.y)

data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=220)
act1 <- mx.symbol.Activation(fc1, name="tanh1", act_type="tanh")
drop1 <- mx.symbol.Dropout(act1, p=0.5)
fc2 <- mx.symbol.FullyConnected(drop1, name="fc2", num_hidden=220)
act2 <- mx.symbol.Activation(fc2, name="tanh2", act_type="tanh")
drop2 <- mx.symbol.Dropout(act2, p=0.5)
fc3 <- mx.symbol.FullyConnected(drop2, name="fc3", num_hidden=110)
act3 <- mx.symbol.Activation(fc3, name="tanh3", act_type="tanh")
drop3 <- mx.symbol.Dropout(act3, p=0.5)
fc4 <- mx.symbol.FullyConnected(drop3, name="fc4", num_hidden=1)
output <- mx.symbol.LinearRegressionOutput(fc4, name="linreg")
devices <- mx.cpu()
mx.set.seed(71)

model <- mx.model.FeedForward.create(output, X=train.x, y=train.y, ctx=devices, num.round=100, array.batch.size=100, learning.rate=1e-5, momentum=0.99,  eval.metric=mx.metric.rmse, initializer=mx.init.uniform(0.5), array.layout = "rowmajor", epoch.end.callback=mx.callback.log.train.metric(100))

preds <- predict(model, test.x, array.layout='rowmajor')
rmse(preds, test.y)

data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=360)
act1 <- mx.symbol.Activation(fc1, name="tanh1", act_type="tanh")
drop1 <- mx.symbol.Dropout(act1, p=0.2)
fc2 <- mx.symbol.FullyConnected(drop1, name="fc2", num_hidden=30)
act2 <- mx.symbol.Activation(fc2, name="tanh2", act_type="tanh")
drop2 <- mx.symbol.Dropout(act2, p=0.2)
fc3 <- mx.symbol.FullyConnected(drop2, name="fc3", num_hidden=10)
act3 <- mx.symbol.Activation(fc3, name="tanh3", act_type="tanh")
drop3 <- mx.symbol.Dropout(act3, p=0.2)
fc4 <- mx.symbol.FullyConnected(drop3, name="fc4", num_hidden=1)
output <- mx.symbol.LinearRegressionOutput(fc4, name="linreg")
devices <- mx.cpu()
mx.set.seed(71)
model <- mx.model.FeedForward.create(output, X=train.x, y=train.y, ctx=devices, num.round=250, array.batch.size=200, learning.rate=2e-4, momentum=0.99,  eval.metric=mx.metric.rmse, initializer=mx.init.uniform(0.5), array.layout = "rowmajor", epoch.end.callback=mx.callback.log.train.metric(20))

preds <- predict(model, test.x, array.layout='rowmajor')
rmse(preds, test.y)

