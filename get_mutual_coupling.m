function C = get_mutual_coupling(M,N,mc_coef)
    C = zeros(M*N, M*N);
    for idx1 = 1:M
        for idx2 = 1:N
            C_tmp = zeros(M,N); 
            for idx3 = idx1-3:idx1+3
                for idx4 = idx2-3:idx2+3
                    if idx3==idx1 && idx4 ==idx2
                        C_tmp(idx3, idx4) = 1;
                    else
                        if idx3<1 ||idx3>M || idx4<1 || idx4>N 
                            continue;
                        else
                            C_tmp(idx3, idx4) = rand(1,1)*((mc_coef(2)-mc_coef(1))+mc_coef(1))*exp(1j*2*pi*rand(1,1));
                        end 
                    end
                end
            end
            C(M*(idx2-1)+idx1,:) = vecH(C_tmp);
        end
    end
end