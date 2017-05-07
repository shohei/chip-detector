library(quadprog)
library(ggplot2)
c1<-c(1,1)
c2<-c(5,5)
c1x <- c1[1]
c1y <- c1[2]
c2x <- c2[1]
c2y <- c2[2]
p <- matrix(rep(c(0,0,0),40),nrow=40,byrow=T)
for (i in 1:20) {
  p[i,] <- c(rnorm(1,c1x,1),rnorm(1,c1y,1),1)
}
for (i in 21:40) {
  p[i,] <- c(rnorm(1,c2x,1),rnorm(1,c2y,1),-1)
}
df<-data.frame(p)
names(df)<-c('x','y','label')
ggplot(df,aes(x,y,color=label))+geom_point()

D <- matrix(rep(0,40*40),nrow=40,byrow=T)
for (i in 1:40){
  for (j in 1:40){
    xi <- matrix(p[i,1:2],nrow=2)
    xj <- matrix(p[j,1:2],nrow=2)
    D[i,j] <-  p[i,3]*p[j,3]*t(xi)%*%xj
  } 
}
d <- rep(-1,40)

A <- matrix(c(p[,3],-p[,3]),40,2)
b <- c(0,0)

result <- solve.QP(D,d,A,bvec=b)

result$solution


cat('input any character and hit enter to finish > ')
b <- scan("stdin", character(), n=1)
