function dist = gp2_rerank(par, dist, par_gp2)
 
num_query = par.num_query;
k1 = par.k(1);
k2 = par.k(2);
L = par.L;
 
dist_gallery_ID = par_gp2.dist_gallery_ID;
g_knn_w = par_gp2.g_knn_w;
dist_gallery_new = par_gp2.dist_gallery_new;
s_ngs = par_gp2.s_ng2;
g_knns = par_gp2.g_knns;

[dist_sort, dist_ID] = sort(dist);
dist_sort = dist_sort(1:L, :);
 
for i = 1:num_query
    
    s_ng = s_ngs(:,dist_ID(1:L,i));
    [~, s_qn] = ismember(g_knns(:, dist_ID(1:L,i)), dist_ID(:,i));
    s_qn = 1./s_qn;
    sim = sum(s_ng.*s_qn);    
    
    q_knn = dist_gallery_ID(1:k2, dist_ID(1:k1, i));
    s_qm = repmat(g_knn_w(dist_ID(1:k1, i)), k2, 1);
    s_qm = s_qm(:);
    s_mg = 1./cell2mat(arrayfun(@(x) dist_gallery_new(q_knn,dist_ID(x,i)), 1:L, 'un', 0));
    sim = sim + (s_qm)'*s_mg;
    
    [~, sim_ID] = sort(sim, 'descend');
    temp = dist_ID(1:L,i);
    dist(temp(sim_ID), i) = dist_sort(:, i);     
   
end
 

