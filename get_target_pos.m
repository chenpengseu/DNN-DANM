function [target_theta, target_phi] = get_target_pos(target_num, theta_range, phi_range, min_space_theta, min_space_phi)
    while true
        target_theta = sort(rand(target_num,1)*(theta_range(2)-theta_range(1))+theta_range(1),'ascend'); % 20~120
        if target_num>1
            tmp = min(abs(target_theta(2:end)-target_theta(1:end-1)));
            if tmp>=min_space_theta
                break;
            end
        else
            break;
        end
    end
    while true
        target_phi = sort(rand(target_num,1)*(phi_range(2)-phi_range(1))+phi_range(1),'ascend'); % -45~45
        if target_num>1
            tmp = min(abs(target_phi(2:end)-target_phi(1:end-1)));
            if tmp>=min_space_phi
                break;
            end
        else
            break;
        end
    end 
end