if(0)
{
	install.packages("drat", repos="https://cran.rstudio.com")
	drat:::addRepo("dmlc")
	install.packages("mxnet")
	install.packages("glmnet")
	install.packages("e1071")
}

# 男子
dm<-read.csv('https://github.com/ozt-ca/tjo.hatenablog.samples/raw/master/r_samples/public_lib/jp/exp_uci_datasets/tennis/men.txt',header=T,sep='\t')
# 女子
dw<-read.csv('https://github.com/ozt-ca/tjo.hatenablog.samples/raw/master/r_samples/public_lib/jp/exp_uci_datasets/tennis/women.txt',header=T,sep='\t')
# uninformativeな変数を除外する
dm<-dm[,-c(1,2,6,16,17,18,19,20,21,24,34,35,36,37,38,39)]
dw<-dw[,-c(1,2,6,16,17,18,19,20,21,24,34,35,36,37,38,39)]

library(mxnet)
train <- data.matrix(dm)
test <- data.matrix(dw)
train.x <- train[,-1]
train.y <- train[,1]
test.x <- test[,-1]
test.y <- test[,1]


if(1)
{
	train_means <- apply(train.x, 2, mean)
	train_stds <- apply(train.x, 2, sd)
	test_means <- apply(test.x, 2, mean)
	test_stds <- apply(test.x, 2, sd)
	train.x <- t((t(train.x)-train_means)/train_stds)
	test.x <- t((t(test.x)-test_means)/test_stds)
}

#設定1
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=220)
act1 <- mx.symbol.Activation(fc1, name="tanh1", act_type="tanh")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=220)
act2 <- mx.symbol.Activation(fc2, name="tanh2", act_type="tanh")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=110)
act3 <- mx.symbol.Activation(fc3, name="tanh3", act_type="tanh")
fc4 <- mx.symbol.FullyConnected(act3, name="fc4", num_hidden=2)
softmax <- mx.symbol.SoftmaxOutput(fc4, name="softmax")
devices <- mx.cpu()
mx.set.seed(71)
model <- mx.model.FeedForward.create(softmax, X=train.x, y=train.y, ctx=devices, num.round=100, array.batch.size=100, learning.rate=0.03, momentum=0.99,  eval.metric=mx.metric.accuracy, initializer=mx.init.uniform(0.5), array.layout = "rowmajor", epoch.end.callback=mx.callback.log.train.metric(100))

preds <- predict(model, test.x, array.layout = "rowmajor")
pred.label <- max.col(t(preds)) - 1
table(test.y, pred.label)

#設定2
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=46)
act1 <- mx.symbol.Activation(fc1, name="tanh1", act_type="tanh")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=46)
act2 <- mx.symbol.Activation(fc2, name="tanh2", act_type="tanh")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=23)
act3 <- mx.symbol.Activation(fc3, name="tanh3", act_type="tanh")
fc4 <- mx.symbol.FullyConnected(act3, name="fc4", num_hidden=2)
softmax <- mx.symbol.SoftmaxOutput(fc4, name="softmax")
devices <- mx.cpu()
mx.set.seed(71)
model <- mx.model.FeedForward.create(softmax, X=train.x, y=train.y, ctx=devices, num.round=100, array.batch.size=100, learning.rate=0.03, momentum=0.99,  eval.metric=mx.metric.accuracy, initializer=mx.init.uniform(0.5), array.layout = "rowmajor", epoch.end.callback=mx.callback.log.train.metric(100))

preds <- predict(model, test.x, array.layout = "rowmajor")
pred.label <- max.col(t(preds)) - 1
table(test.y, pred.label)

#設定3
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=46)
act1 <- mx.symbol.Activation(fc1, name="tanh1", act_type="tanh")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=46)
act2 <- mx.symbol.Activation(fc2, name="tanh2", act_type="tanh")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=23)
act3 <- mx.symbol.Activation(fc3, name="tanh3", act_type="tanh")
fc4 <- mx.symbol.FullyConnected(act3, name="fc4", num_hidden=2)
softmax <- mx.symbol.SoftmaxOutput(fc4, name="softmax")
devices <- mx.cpu()
mx.set.seed(71)
model <- mx.model.FeedForward.create(softmax, X=train.x, y=train.y, ctx=devices, num.round=9, array.batch.size=100, learning.rate=0.03, momentum=0.99,  eval.metric=mx.metric.accuracy, initializer=mx.init.uniform(0.5), array.layout = "rowmajor", epoch.end.callback=mx.callback.log.train.metric(100))


preds <- predict(model, test.x, array.layout = "rowmajor")
pred.label <- max.col(t(preds)) - 1
table(test.y, pred.label)
sum(diag(table(test.y, pred.label)))/nrow(test)

# L1正則化ロジスティック回帰
library(glmnet)
dm.l1 <- cv.glmnet(as.matrix(dm[,-1]), as.matrix(dm[,1]), family='binomial', alpha=1)
table(dw$Result, round(predict(dm.l1, newx=as.matrix(dw[,-1]), type='response', s=dm.l1$lambda.min),0))
   
sum(diag(table(dw$Result, round(predict(dm.l1, newx=as.matrix(dw[,-1]), type='response', s=dm.l1$lambda.min),0))))/nrow(dw)

# 線形SVM
library(e1071)
dm.svm.l <- svm(as.factor(Result)~., dm, kernel='linear')
table(dw$Result, predict(dm.svm.l, newdata=dw[,-1]))
   
sum(diag(table(dw$Result, predict(dm.svm.l, newdata=dw[,-1]))))/nrow(dw)

# データをインポートして、MXnetに読ませるための下処理と、プロット周りをグリッド含めて準備しておく
dbi <- read.csv('https://github.com/ozt-ca/tjo.hatenablog.samples/raw/master/r_samples/public_lib/jp/dbi_large.txt', header=T, sep='\t')
dbi$label <- dbi$label-1
px <- seq(-4,4,0.03)
py <- seq(-4,4,0.03)
pgrid <- expand.grid(px,py)
names(pgrid) <- names(dbi)[-3]
dbi_train <- data.matrix(dbi)
dbi_train.x <- dbi_train[,-3]
dbi_train.y <- dbi_train[,3]
dbi_test <- data.matrix(pgrid)
label_grid <- rep(0, nrow(pgrid))
for (i in 1:nrow(pgrid)){
     if (pgrid[i,1]+pgrid[i,2]<0){label_grid[i] <- 1}
}

# MXnetで4-3-3中間層2層のDNNを回す
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=4)
act1 <- mx.symbol.Activation(fc1, name="tanh1", act_type="tanh")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=3)
act2 <- mx.symbol.Activation(fc2, name="tanh2", act_type="tanh")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=3)
act3 <- mx.symbol.Activation(fc3, name="tanh3", act_type="tanh")
fc4 <- mx.symbol.FullyConnected(act3, name="fc4", num_hidden=2)
softmax <- mx.symbol.SoftmaxOutput(fc4, name="softmax")
devices <- mx.cpu()
mx.set.seed(71)
model <- mx.model.FeedForward.create(softmax, X=dbi_train.x, y=dbi_train.y, ctx=devices, num.round=20, array.batch.size=100, learning.rate=0.03, momentum=0.99,  eval.metric=mx.metric.accuracy, initializer=mx.init.uniform(0.5), array.layout = "rowmajor", epoch.end.callback=mx.callback.log.train.metric(100))
Start training with 1 devices

preds <- predict(model, dbi_test, array.layout = "rowmajor")
pred.label <- max.col(t(preds)) - 1
table(label_grid, pred.label)
sum(diag(table(label_grid, pred.label)))/nrow(pgrid)

# 描画する
plot(c(), type='n', xlim=c(-4,4), ylim=c(-4,4), xlab='', ylab='')
par(new=T)
polygon(c(-4,4,4),c(4,-4,4),col='#dddddd')
par(new=T)
polygon(c(-4,-4,4),c(4,-4,-4),col='#ffdddd')
par(new=T)
plot(dbi[,-3], pch=19, cex=0.5, col=dbi$label+1, xlim=c(-4,4), ylim=c(-4,4), xlab='', ylab='')
par(new=T)

contour(px, py, array(pred.label, dim=c(length(px),length(py))), col='purple', lwd=5, levels =0.5, drawlabels=F, xlim=c(-4,4), ylim=c(-4,4))

data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=200)
act1 <- mx.symbol.Activation(fc1, name="tanh1", act_type="tanh")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=200)
act2 <- mx.symbol.Activation(fc2, name="tanh2", act_type="tanh")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=100)
act3 <- mx.symbol.Activation(fc3, name="tanh3", act_type="tanh")
fc4 <- mx.symbol.FullyConnected(act3, name="fc4", num_hidden=2)
softmax <- mx.symbol.SoftmaxOutput(fc4, name="softmax")
devices <- mx.cpu()
mx.set.seed(71)
model <- mx.model.FeedForward.create(softmax, X=dbi_train.x, y=dbi_train.y, ctx=devices, num.round=20, array.batch.size=100, learning.rate=0.03, momentum=0.99,  eval.metric=mx.metric.accuracy, initializer=mx.init.uniform(0.5), array.layout = "rowmajor", epoch.end.callback=mx.callback.log.train.metric(100))

preds <- predict(model, dbi_test, array.layout = "rowmajor")
pred.label <- max.col(t(preds)) - 1
table(label_grid, pred.label)

sum(diag(table(label_grid, pred.label)))/nrow(pgrid)

plot(c(), type='n', xlim=c(-4,4), ylim=c(-4,4), xlab='', ylab='')
par(new=T)
polygon(c(-4,4,4),c(4,-4,4),col='#dddddd')
par(new=T)
polygon(c(-4,-4,4),c(4,-4,-4),col='#ffdddd')
par(new=T)
plot(dbi[,-3], pch=19, cex=0.5, col=dbi$label+1, xlim=c(-4,4), ylim=c(-4,4), xlab='', ylab='')
par(new=T)
contour(px, py, array(pred.label, dim=c(length(px),length(py))), col='purple', lwd=5, levels =0.5, drawlabels=F, xlim=c(-4,4), ylim=c(-4,4))
