x<-read.table("stat01.txt",header=T,sep=",")
summary(x)
t.test(x$A,x$C,var.equal=T)

#���K���̌���iKolmogorov-Smirnov�i�R�����S���t�E�X�~�m�t�j����j
ks.test(x$A,"pnorm",mean=mean(x$A),sd=sd(x$A))
ks.test(x$C,"pnorm",mean=mean(x$C),sd=sd(x$C))

#�����U���̌���iF����j
var.test(x$A,x$C)

t.test(x$A,x$C,var.equal=T)   # �����U
t.test(x$A,x$C,var.equal=F)   # �s�����U