x<-read.table("stat01.txt",header=T,sep=",")
summary(x)
t.test(x$A,x$C,var.equal=T)