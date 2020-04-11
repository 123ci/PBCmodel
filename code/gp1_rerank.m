function dist = gp1_rerank(par, dist, par_gp1)
 
 
num_query = par.num_query;
k = par.k(2);
L = par.L;
 
dist_gallery_new = par_gp1.dist_gallery_new;
s_ngs = par_gp1.s_ng1;
g_knns = par_gp1.g_knn1s;
    
[dist_sort, dist_ID] = sort(dist);
dist_sort = dist_sort(1:L, :);
 
for i = 1:num_query
    
    s_ng = s_ngs(:,dist_ID(1:L,i));  
    [~, s_qn] = ismember(g_knns(:, dist_ID(1:L,i)), dist_ID(:,i));    
    s_qn = 1./s_qn;
    sim = sum(s_ng.*s_qn);
    clear g_knn
    
    q_knn = dist_ID(1:k, i);
    s_qm = 1./(1:k)';
    s_mg = 1./(cell2mat(arrayfun(@(x) dist_gallery_new(q_knn,dist_ID(x,i)), 1:L, 'un', 0)));
    sim = sim + s_qm'*s_mg;
    
    [~, sim_ID] = sort(sim, 'descend');
    temp = dist_ID(1:L,i);
    dist(temp(sim_ID), i) = dist_sort(:, i);
       
end
 
 

