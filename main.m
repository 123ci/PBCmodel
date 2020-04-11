clear
clc

addpath baseline
addpath code

% setting parameters 
par.dataset = 'CUHK03labeled';      % market, duke, MARS, CUHK03labeled, CUHK03detected
par.baseline = 'PCB';    % BTricks, IDE-C, IDE-C+KISSME, IDE-C+XQDA, MHN-6(PCB), PCB

% load baseline
load([par.dataset, '_', par.baseline, '.mat'])

% setting parameters 
par.setnum = length(infor);
[par.num_test, par.num_query] = size(infor(1).dist);
par.L = 200;
par.MethodOption = [1 1]; % [optimization with the second order context, optimization with the first order context]
par.flag = [1 1]; % [computing evaluation index of baseline method, running PBCmodel]
if strcmp(par.dataset, 'CUHK03labeled') || strcmp(par.dataset, 'CUHK03detected')
    par.k = [3 10];  % [k0,k]
else
    par.k = [2 10];  % [k0,k]
end

CMC_o = zeros(par.setnum, par.num_test);
map_o = zeros(par.setnum, 1);
CMC = CMC_o;
map = map_o;

for set = 1:par.setnum
    
    disp('---------------------------------------')
    disp(['set=' num2str(set)])
    disp('---------------------------------------')

    dist = infor(set).dist;
    dist_gallery = full(infor(set).dist_gallery);
    dist_gallery = dist_gallery+dist_gallery';
    queryID = infor(set).queryID;
    queryCam = infor(set).queryCam;
    testID = infor(set).testID;
    testCam = infor(set).testCam;
    
    %% compute evaluation index of baseline method
    if par.flag(1) == 1
       [CMC_o(set,:), map_o(set)] = evaluation(dist, testID, queryID, testCam, queryCam);
       disp('results of baseline method:')
       disp('    rank1     rank5     rank10    map')
       disp([CMC_o(set,[1 5 10]) map_o(set)])
       disp('---------------------------------------')
    end

    %% running PBCmodel
    if par.flag(2) == 1       
        if exist([par.dataset, '_', par.baseline, '_offline_gallery.mat'], 'file')~=0    
            load([par.dataset, '_', par.baseline, '_offline_gallery.mat'])
            offlineTime = 0;
        else
            temp = cputime;
            offlineG = Computer_offline_gallery(par, dist_gallery);
            offlineTime = cputime-temp;
            save(['baseline/', par.dataset, '_', par.baseline, '_offline_gallery.mat'], 'offlineG', '-v7.3');
        end
             
        [dist, onlineTime] = re_rank(par, offlineG, dist, dist_gallery);
        [CMC(set,:), map(set)] = evaluation(dist, testID, queryID, testCam, queryCam);
        disp('results of re-ranking method:')
        disp('    rank1     rank5     rank10    map')
        disp([CMC(set,[1 5 10]), map(set)])       
    end

end


