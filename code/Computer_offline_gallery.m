function offlineG = Computer_offline_gallery(par, dist_gallery)


num_test = par.num_test;
k1 = par.k(1);
k2 = par.k(2);
[~, dist_gallery_ID] = sort(dist_gallery);


% computing Eq.(10) for all gallery samples
g_knn = dist_gallery_ID(1:k2, :);
g_knn_w = 1./(1:k2);
g_knn_w_temp = zeros(k2, num_test);
for i = 1:num_test
    g_knn_w_temp(:,i) = cell2mat(arrayfun(@(x) g_knn_w*ismember(g_knn(:,g_knn(x,i)), g_knn(:,i)), 1:k2, 'un', 0));
end
g_knn_w = sum(g_knn_w_temp)/(k2*sum(g_knn_w));


% computing Eq.(8) (Eq.(9)) for all gallery samples
[~, dist_gallery_new] = sort(dist_gallery_ID);
dist_gallery_new = dist_gallery_new + dist_gallery_new' + max(dist_gallery_new, dist_gallery_new');


% computing 
% - the first order context sample's weight for all gallery samples
% - the second order context sample's weight for all gallery samples
% - the second order context for all gallery samples 
g_knn1s = dist_gallery_ID(2:k2+1, :); 
g_knn0s = dist_gallery_ID(2:k1+1, :); 

s_ng1 = zeros(k2, num_test);     % the first order context sample's weight
s_ng2 = zeros(k1*k2, num_test);  % the second order context sample's weight
g_knns = s_ng2;                  % the second order context
for i = 1:num_test
    s_ng1(:,i) = 1./(dist_gallery_new(g_knn1s(:,i),i)-3);
    
    g_knn = g_knn0s(:, i); 
    s_ng = g_knn_w(g_knn);
    s_ng = repmat(s_ng, k2, 1);
    s_ng2(:,i) = s_ng(:);
    
    g_knn = dist_gallery_ID(1:k2, g_knn);
    g_knn = g_knn(:);
    g_knns(:,i) = g_knn;    
end


offlineG.g_knn_w = g_knn_w;
offlineG.dist_gallery_new = tril(dist_gallery_new);
offlineG.s_ng1 = s_ng1;
offlineG.s_ng2 = s_ng2;
offlineG.g_knns = g_knns;













