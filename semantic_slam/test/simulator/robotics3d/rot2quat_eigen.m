function q = rot2quat_eigen(R)
% code from Eigen c++ library

t = trace(R);
q = [0;0;0;1];

if t > 0
    t = sqrt(t + 1);
    q(4) = 0.5 * t;
    t = 0.5/t;
    q(1) = (R(3,2) - R(2,3)) * t;
    q(2) = (R(1,3) - R(3,1)) * t;
    q(3) = (R(2,1) - R(1,2)) * t;
else
    i = 0;
    if R(2,2) > R(1,1)
        i = 1;
    end
    if R(3,3) > R(i+1,i+1)
        i = 2;
    end
    j = mod(i+1, 3);
    k = mod(j+1, 3);
    
    i = i + 1;
    j = j + 1;
    k = k + 1;

    t = sqrt(R(i,i)-R(j,j)-R(k,k) + 1);
    q(i) = 0.5 * t;
    t = 0.5/t;
    q(4) = (R(k,j)-R(j,k))*t;
    q(j) = (R(j,i)+R(i,j))*t;
    q(k) = (R(k,i)+R(i,k))*t;
end