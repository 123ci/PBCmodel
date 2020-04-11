function [dist, t] = re_rank(par, offlineG, dist, dist_gallery)
 
 
MethodOption = par.MethodOption;
[~, dist_gallery_ID] = sort(dist_gallery);    
g_knn1s = dist_gallery_ID(2:par.k(2)+1, :);

g_knn_w = offlineG.g_knn_w;
dist_gallery_new = offlineG.dist_gallery_new+offlineG.dist_gallery_new';
dist_gallery_new(eye(size(dist_gallery_new))==1) = dist_gallery_new(eye(size(dist_gallery_new))==1)/2;
s_ng1 = offlineG.s_ng1;
s_ng2 = offlineG.s_ng2;
g_knns = offlineG.g_knns;
       

t = [0 0];
% optimization with the second order context
if MethodOption(1) == 1 
    par_gp2.dist_gallery_ID = dist_gallery_ID;
    par_gp2.g_knn_w = g_knn_w;
    par_gp2.dist_gallery_new = dist_gallery_new;
    par_gp2.s_ng2 = s_ng2;
    par_gp2.g_knns = g_knns;
    
    temp = cputime;
    dist = gp2_rerank(par, dist, par_gp2);  
    t(1) = cputime-temp;
end
% optimization with the first order context
if MethodOption(2) == 1  
    par_gp1.dist_gallery_new = dist_gallery_new;
    par_gp1.s_ng1 = s_ng1;
    par_gp1.g_knn1s = g_knn1s; 
    
    temp = cputime;
    dist = gp1_rerank(par, dist, par_gp1);
    t(2) = cputime-temp;
end
t = sum(t);
 
 
 