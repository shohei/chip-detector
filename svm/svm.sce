N=40;
c1 = [1,1];
c2 = [5,5];

c1x=grand(N/2,1,'nor',c1(1),1);
c1y=grand(N/2,1,'nor',c1(2),1);
label1=ones(N/2,1)*(-1);
c2x=grand(N/2,1,'nor',c2(1),1);
c2y=grand(N/2,1,'nor',c2(2),1);
label2=ones(N/2,1);
s=ones(N,1)*200;
points=[c1x,c1y,label1;c2x,c2y,label2];
gcf().color_map = wintercolormap(64);
scatter(points(:,1),points(:,2),s,points(:,3),"fill");

Q=zeros(N,N)
for idx=1:N
    for jdx=1:N
        Q(idx,jdx) = points(idx,1)*points(jdx,1)*points(idx,2)*points(jdx,2);
    end    
end
p = ones(1,N)*(-1);
C = 
b = 
me = 

[x,iact,iter,f]=qpsolve(Q,p);
x
iact
iter
f



