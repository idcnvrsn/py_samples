x<-read.table("stat01.txt",header=T,sep=",")
summary(x)
t.test(x$A,x$C,var.equal=T)

#正規性の検定（Kolmogorov-Smirnov（コロモゴロフ・スミノフ）検定）
ks.test(x$A,"pnorm",mean=mean(x$A),sd=sd(x$A))
ks.test(x$C,"pnorm",mean=mean(x$C),sd=sd(x$C))

#等分散性の検定（F検定）
var.test(x$A,x$C)

t.test(x$A,x$C,var.equal=T)   # 等分散
t.test(x$A,x$C,var.equal=F)   # 不等分散

wilcox.test(x$A,x$C)