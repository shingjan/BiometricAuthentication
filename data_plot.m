clc;
clear close all;
%% Load data
scores = importdata('biomet_data.csv');
data = [];
for i = 1: numel(scores)
   dataTemp = textscan(scores{i},['"', repmat('%f ', [1, 1000]), '"']);
   data = [data; dataTemp];
end
data = cell2mat(data);
% save('data.mat', 'data');
trainSet = data(:, mod(0:999, 10)<5); % First 5 instances are for training. You may have better idea for masking.
testSet = data(:, mod(0:999, 10)>=5); % Last 5 instances are for testing
label = floor((5:504)/5); % Label is from 1 to 100

%% Matching scores generation
fprintf('Verification: Euclidean distance\n')
g_score = [];
i_score = [];
minDis = 100; % Set a large value
for k = 1:500
    temp = testSet(:, k);
    templ = label(k);
    for i = 1:100
        for j = 1:5
            anch = trainSet(:, (i-1)*5+j);
            Dis = norm(anch-temp)/144;
            if Dis < minDis
                minDis = Dis;
            end;
        end;
        
        anchl = label(i*5);
        if templ == anchl
            g_score = [g_score, minDis];
        else
            i_score = [i_score, minDis];
        end;
        minDis = 100;     
    end;
end;
eu_g = g_score;
eu_i = i_score;

fprintf('Verification: Cosine distance\n')
g_score = [];
i_score = [];
maxDis = 1;
g_score = [];
i_score = [];
maxDis = 0;
for k = 1:500
    temp = testSet(:, k);
    templ = label(k);
    for i = 1:100
        for j = 1:5
            anch = trainSet(:, (i-1)*5+j);
            Dis = dot(temp,anch)/(norm(temp,2)*norm(anch,2));
            if Dis > maxDis
                maxDis = Dis;
            end;
        end;
        
        anchl = label(i*5);
        if templ == anchl
            g_score = [g_score, maxDis];
        else
            i_score = [i_score, maxDis];
        end;
        maxDis = 0;     
    end;
end;
cos_g = g_score;
cos_i = i_score;
%% Plot Genuine Distribution
fprintf('Plot Genuine Distribution\n');
figure;
% Set 100 bins
histogram(eu_g,100,'facecolor',[0.8941, 0.1020, 0.1098],'facealpha',.5,'edgecolor','none','Normalization','probability')
title('Genuine Score Distribution (Euclidean) ');
figure;
histogram(cos_g,100,'facecolor',[0.8941, 0.1020, 0.1098],'facealpha',.5,'edgecolor','none','Normalization','probability')
title('Genuine Score Distribution (Cosine) ');
%% Plot Imposter Distribution
fprintf('Plot Imposter Distribution\n');
figure;
histogram(eu_i,100,'facecolor',[0.2157, 0.4941, 0.7216],'facealpha',.5,'edgecolor','none','Normalization','probability')
title('Imposter Score Distribution (Euclidean) ');
figure;
histogram(cos_i,100,'facecolor',[0.2157, 0.4941, 0.7216],'facealpha',.5,'edgecolor','none','Normalization','probability')
title('Imposter Score Distribution (Cosine) ');

%% Plot FAR & FRR
% Euclidean Distance Plot
resolu = 1000;
fprintf(' Plot FAR & FRR (Euclidean)\n')
true_scores = eu_g;
false_scores = eu_i;
cli_scor = numel(true_scores);
imp_scor = numel(false_scores);
dmax = true_scores;
dmin = false_scores;
dminx = min(true_scores);
dmaxx = max(false_scores);
delta = (dmaxx-dminx)/resolu;
% Calculate false reject rate
counter=1;
fre_eu = zeros(1,resolu);
for trash=dminx:delta:dmaxx
    num_ok = sum(dmax<trash);
    fre_eu(1,counter) = 1-(num_ok/cli_scor);
    counter = counter+1;
end
counter=1;
% Calculate false accept rate
fae_eu = zeros(1,resolu);
for trash=dminx:delta:dmaxx
    num_ok = sum(dmin<trash);
    fae_eu(1,counter) = (num_ok/imp_scor);
    counter = counter+1;
end
figure;
plot(fae_eu, 'r');
hold on
plot(fre_eu, 'k');
title('FAR & FRR (Euclidean)');
h = legend('gscore', 'iscore');
% Cosine distance Plot
resolu = 1000;
fprintf(' Plot FAR & FRR (Cosine)\n')
true_scores = 1-cos_g; % We always want genuine score is less than imposter
false_scores = 1-cos_i;
cli_scor = numel(true_scores);
imp_scor = numel(false_scores);
dmax = true_scores;
dmin = false_scores;
dminx = min(true_scores);
dmaxx = max(false_scores);
delta = (dmaxx-dminx)/resolu;
% Calculate false reject rate
counter=1;
fre_cos = zeros(1,resolu);
for trash=dminx:delta:dmaxx
    num_ok = sum(dmax<trash);
    fre_cos(1,counter) = 1-(num_ok/cli_scor);
    counter = counter+1;
end
counter=1;
% Calculate false accept rate
fae_cos = zeros(1,resolu);
for trash=dminx:delta:dmaxx
    num_ok = sum(dmin<trash);
    fae_cos(1,counter) = (num_ok/imp_scor);
    counter = counter+1;
end
figure;
plot(fae_cos, 'r');
hold on
plot(fre_cos, 'k');
title('FAR & FRR (Cosine)');
h = legend('gscore', 'iscore');

%% Plot ROC
fprintf(' Plot ROC (Euclidean)\n')
tar_eu = 1- fre_eu;
figure;
h=semilogx(fae_eu,tar_eu,'Color','r','Linewidth',2);
title('ROC (Euclidean)');
fprintf(' Plot ROC (Cosine)\n')
tar_cos = 1- fre_cos;
figure;
h=semilogx(fae_cos,tar_cos,'Color','r','Linewidth',2);
title('ROC (Cosine)');

%% Calculating DI index and EER
fprintf(' Calculating DI index & EER (Euclidean)\n')
DI_eu = abs(mean(eu_i)-mean(eu_g))/sqrt((var(eu_i)+var(eu_g))/2)
eer_eu = min(fae_eu+fre_eu)/2
fprintf(' Calculating DI index & EER (Cosine)\n')
DI_cos = abs(mean(cos_i)-mean(cos_g))/sqrt((var(cos_i)+var(cos_g))/2)
eer_cos = min(fae_cos+fre_cos)/2