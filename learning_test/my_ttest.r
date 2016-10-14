x<-read.table("stat01.txt",header=T,sep=",")
summary(x)
t.test(x$A,x$C,var.equal=T)

#³‹K«‚ÌŒŸ’èiKolmogorov-SmirnoviƒRƒƒ‚ƒSƒƒtEƒXƒ~ƒmƒtjŒŸ’èj
ks.test(x$A,"pnorm",mean=mean(x$A),sd=sd(x$A))
ks.test(x$C,"pnorm",mean=mean(x$C),sd=sd(x$C))

#“™•ªŽU«‚ÌŒŸ’èiFŒŸ’èj
var.test(x$A,x$C)

t.test(x$A,x$C,var.equal=T)   # “™•ªŽU
t.test(x$A,x$C,var.equal=F)   # •s“™•ªŽU

wilcox.test(x$A,x$C)