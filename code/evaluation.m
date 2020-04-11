%% This function is from https://github.com/zhunzhong07/person-re-ranking/blob/master/evaluation/utils/evaluation.m
%% by Zhong et al.

function [CMC, map] = evaluation(dist, label_gallery, label_query, cam_gallery, cam_query)

junk0 = find(label_gallery == -1);
ap = zeros(size(dist, 2), 1);
CMC = [];

[~, indexs] = sort(dist);
for k = 1:size(dist, 2)
    q_label = label_query(k);
    q_cam = cam_query(k);
    pos = find(label_gallery == q_label);
    pos2 = cam_gallery(pos) ~= q_cam;
    good_image = pos(pos2);
    pos3 = cam_gallery(pos) == q_cam;
    junk = pos(pos3);
    junk_image = [junk0; junk];
    index = indexs(:,k);
    
    [ap(k), CMC(:, k)] = compute_AP(good_image, junk_image, index);
end


CMC = sum(CMC, 2)./size(dist, 2);
CMC = CMC';
ap(isnan(ap) == 1) = 0;
map = sum(ap)/length(ap);