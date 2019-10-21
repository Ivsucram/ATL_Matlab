% Marcus Vinicius Sousa Leite de Carvalho
% marcus.decarvalho@ntu.edu.sg
%
% NANYANG TECHNOLOGICAL UNIVERSITY - NTUITIVE PTE LTD Dual License Agreement
% Non-Commercial Use Only 
% This NTUITIVE License Agreement, including all exhibits ("NTUITIVE-LA") is a legal agreement between you and NTUITIVE (or “we”) located at 71 Nanyang Drive, NTU Innovation Centre, #01-109, Singapore 637722, a wholly owned subsidiary of Nanyang Technological University (“NTU”) for the software or data identified above, which may include source code, and any associated materials, text or speech files, associated media and "online" or electronic documentation and any updates we provide in our discretion (together, the "Software"). 
% 
% By installing, copying, or otherwise using this Software, found at https://github.com/Ivsucram/ATL_Matlab, you agree to be bound by the terms of this NTUITIVE-LA.  If you do not agree, do not install copy or use the Software. The Software is protected by copyright and other intellectual property laws and is licensed, not sold.   If you wish to obtain a commercial royalty bearing license to this software please contact us at marcus.decarvalho@ntu.edu.sg.
% 
% SCOPE OF RIGHTS:
% You may use, copy, reproduce, and distribute this Software for any non-commercial purpose, subject to the restrictions in this NTUITIVE-LA. Some purposes which can be non-commercial are teaching, academic research, public demonstrations and personal experimentation. You may also distribute this Software with books or other teaching materials, or publish the Software on websites, that are intended to teach the use of the Software for academic or other non-commercial purposes.
% You may not use or distribute this Software or any derivative works in any form for commercial purposes. Examples of commercial purposes would be running business operations, licensing, leasing, or selling the Software, distributing the Software for use with commercial products, using the Software in the creation or use of commercial products or any other activity which purpose is to procure a commercial gain to you or others.
% If the Software includes source code or data, you may create derivative works of such portions of the Software and distribute the modified Software for non-commercial purposes, as provided herein.  
% If you distribute the Software or any derivative works of the Software, you will distribute them under the same terms and conditions as in this license, and you will not grant other rights to the Software or derivative works that are different from those provided by this NTUITIVE-LA. 
% If you have created derivative works of the Software, and distribute such derivative works, you will cause the modified files to carry prominent notices so that recipients know that they are not receiving the original Software. Such notices must state: (i) that you have changed the Software; and (ii) the date of any changes.
% 
% You may not distribute this Software or any derivative works. 
% In return, we simply require that you agree: 
% 1.	That you will not remove any copyright or other notices from the Software.
% 2.	That if any of the Software is in binary format, you will not attempt to modify such portions of the Software, or to reverse engineer or decompile them, except and only to the extent authorized by applicable law. 
% 3.	That NTUITIVE is granted back, without any restrictions or limitations, a non-exclusive, perpetual, irrevocable, royalty-free, assignable and sub-licensable license, to reproduce, publicly perform or display, install, use, modify, post, distribute, make and have made, sell and transfer your modifications to and/or derivative works of the Software source code or data, for any purpose.  
% 4.	That any feedback about the Software provided by you to us is voluntarily given, and NTUITIVE shall be free to use the feedback as it sees fit without obligation or restriction of any kind, even if the feedback is designated by you as confidential. 
% 5.	THAT THE SOFTWARE COMES "AS IS", WITH NO WARRANTIES. THIS MEANS NO EXPRESS, IMPLIED OR STATUTORY WARRANTY, INCLUDING WITHOUT LIMITATION, WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, ANY WARRANTY AGAINST INTERFERENCE WITH YOUR ENJOYMENT OF THE SOFTWARE OR ANY WARRANTY OF TITLE OR NON-INFRINGEMENT. THERE IS NO WARRANTY THAT THIS SOFTWARE WILL FULFILL ANY OF YOUR PARTICULAR PURPOSES OR NEEDS. ALSO, YOU MUST PASS THIS DISCLAIMER ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE WORKS.
% 6.	THAT NEITHER NTUITIVE NOR NTU NOR ANY CONTRIBUTOR TO THE SOFTWARE WILL BE LIABLE FOR ANY DAMAGES RELATED TO THE SOFTWARE OR THIS NTUITIVE-LA, INCLUDING DIRECT, INDIRECT, SPECIAL, CONSEQUENTIAL OR INCIDENTAL DAMAGES, TO THE MAXIMUM EXTENT THE LAW PERMITS, NO MATTER WHAT LEGAL THEORY IT IS BASED ON. ALSO, YOU MUST PASS THIS LIMITATION OF LIABILITY ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE WORKS.
% 7.	That we have no duty of reasonable care or lack of negligence, and we are not obligated to (and will not) provide technical support for the Software.
% 8.	That if you breach this NTUITIVE-LA or if you sue anyone over patents that you think may apply to or read on the Software or anyone's use of the Software, this NTUITIVE-LA (and your license and rights obtained herein) terminate automatically.  Upon any such termination, you shall destroy all of your copies of the Software immediately.  Sections 3, 4, 5, 6, 7, 8, 11 and 12 of this NTUITIVE-LA shall survive any termination of this NTUITIVE-LA.
% 9.	That the patent rights, if any, granted to you in this NTUITIVE-LA only apply to the Software, not to any derivative works you make.
% 10.	That the Software may be subject to U.S. export jurisdiction at the time it is licensed to you, and it may be subject to additional export or import laws in other places.  You agree to comply with all such laws and regulations that may apply to the Software after delivery of the software to you.
% 11.	That all rights not expressly granted to you in this NTUITIVE-LA are reserved.
% 12.	That this NTUITIVE-LA shall be construed and controlled by the laws of the Republic of Singapore without regard to conflicts of law.  If any provision of this NTUITIVE-LA shall be deemed unenforceable or contrary to law, the rest of this NTUITIVE-LA shall remain in full effect and interpreted in an enforceable manner that most nearly captures the intent of the original language. 
% 
% Copyright (c) NTUITIVE. All rights reserved.

clc
clear

dm = DataManipulator('');
dm.loadCustomCSV();
dm.normalize();

dm.splitAsSourceTargetStreams(1000, 'none', 0.5);

crSource             = ones(dm.nMinibatches, 1);
crTarget             = ones(dm.nMinibatches, 1);
trainTime            = ones(dm.nMinibatches, 1);
testTime             = ones(dm.nMinibatches, 1);
KlLossEvolution      = ones(dm.nMinibatches, 1);
classificationLoss   = ones(dm.nMinibatches, 1);
nodeEvolution        = zeros(dm.nMinibatches, 1);
discriminativeLoss   = ones(dm.nMinibatches, 1);
generativeLossTarget = ones(dm.nMinibatches, 1);
agmmTargetGenSize    = [];
agmmSourceDiscSize   = [];

nodeEvolutionTarget = ones(dm.nMinibatches, 1);
nodeEvolutionSource = ones(dm.nMinibatches, 1);
gmmTargetBatch      = ones(dm.nMinibatches, 1);
gmmSourceBatch      = ones(dm.nMinibatches, 1);


nn = NeuralNetwork([dm.nFeatures 1 dm.nClasses]);
ae = DenoisingAutoEncoder([nn.layers(1) nn.layers(2) nn.layers(1)]);

% I am building the greedyLayerBias
x = dm.getXs(1);
ae.greddyLayerWiseTrain(x(1, :), 1, 0.1);
% I am building the greedyLayerBias

agmmSourceDisc = AGMM();
agmmTargetGen  = AGMM();

originalLearningRate = ae.learningRate;
epochs = 1;
for i = 1 : dm.nMinibatches
    Xs = dm.getXs(i);
    ys = dm.getYs(i);
    Xt = dm.getXt(i);
    yt = dm.getYt(i);
    
    %% Evaluation ~ Test Target
    tic
    nn.test(Xt, yt);
    crTarget(i) = nn.classificationRate;
    classificationLoss(i) = nn.lossValue;
    testTime(i) = toc;
    
    nn.test(Xs(max(Xs, [], 2) ~= 0, :), ys(max(Xs, [], 2) ~= 0, :));
    crSource(i) = nn.classificationRate;
    discriminativeLoss(i) = nn.lossValue;
    
    ae.test(Xt);
    generativeLossTarget(i) = ae.lossValue;
    
    if i > 1
        nodeEvolutionTarget(i) = nodeEvolutionTarget(i - 1);
        nodeEvolutionSource(i) = nodeEvolutionSource(i - 1);
    end
    
    tic
    for epoch = 1 : epochs
        %% Discriminative phase on Source
        nn.setAgmm(agmmSourceDisc);
        for j = 1 : size(Xs, 1)
            x = Xs(j, :);
            y = ys(j, :);
            if max(y) == 0
                continue
            end
            
            lastHiddenLayerNo = numel(nn.layers) - 1;
            
            nn.forwardpass(x);
            if epoch == 1
                agmmSourceDiscSize(end + 1) = nn.runAgmm(x, y).M();
                nn.widthAdaptationStepwise(lastHiddenLayerNo, y);
            else
                nn.nSamplesFeed = nn.nSamplesFeed - 1;
                nn.nSamplesLayer(lastHiddenLayerNo) = nn.nSamplesLayer(lastHiddenLayerNo) - 1;
                nn.widthAdaptationStepwise(lastHiddenLayerNo, y);
                nn.BIAS2{lastHiddenLayerNo}(end) = [];
                nn.VAR{lastHiddenLayerNo}(end) = [];
            end
            
            if nn.growable(lastHiddenLayerNo)
                nodeEvolutionSource(i) = nodeEvolutionSource(i) + nn.getAgmm().M();
                for numberOfGMMs = 1 : nn.getAgmm().M()
                    nn.grow(lastHiddenLayerNo);
                    ae.grow(lastHiddenLayerNo);
                end
            elseif nn.prunable{lastHiddenLayerNo}(1) ~= 0
                for k = size(nn.prunable{lastHiddenLayerNo}, 1) : -1 : 1
                    nodeToPrune = nn.prunable{lastHiddenLayerNo}(k);
                    ae.prune(lastHiddenLayerNo, nodeToPrune);
                    nn.prune(lastHiddenLayerNo, nodeToPrune);
                    nodeEvolutionSource(i) = nodeEvolutionSource(i) - 1;
                end
            end
            nn.train(x, y);
            
        end
        for j = 1 : numel(nn.layers)-2
            ae.weight{j} = nn.weight{j};
            ae.bias{j}   = nn.bias{j};
        end
        agmmSourceDisc = nn.getAgmm();
        %% Generative phase on Target
        ae.setAgmm(agmmTargetGen);
        for j = 1 : size(Xt, 1)
            x = Xt(j, :);
            y = x;
            lastHiddenLayerNo = numel(nn.layers) - 1;
            
            ae.forwardpass(x);
            if epoch == 1
                agmmTargetGenSize(end + 1) = ae.runAgmm(x, y).M();
                ae.widthAdaptationStepwise(lastHiddenLayerNo, y);
            else
                ae.nSamplesFeed = ae.nSamplesFeed - 1;
                ae.nSamplesLayer(lastHiddenLayerNo) = ae.nSamplesLayer(lastHiddenLayerNo) - 1;
                ae.widthAdaptationStepwise(lastHiddenLayerNo, y);
                ae.BIAS2{lastHiddenLayerNo}(end) = [];
                ae.VAR{lastHiddenLayerNo}(end) = [];
            end
            
            if ae.growable(lastHiddenLayerNo)
                nodeEvolutionTarget(i) = nodeEvolutionTarget(i) + ae.getAgmm().M();
                for numberOfGMMs = 1 : ae.getAgmm.M()
                    ae.grow(lastHiddenLayerNo);
                    nn.grow(lastHiddenLayerNo);
                end
            elseif ae.prunable{lastHiddenLayerNo}(1) ~= 0
                for k = size(ae.prunable{lastHiddenLayerNo}, 1) : -1 : 1
                    nodeToPrune = ae.prunable{lastHiddenLayerNo}(k);
                    ae.prune(lastHiddenLayerNo, nodeToPrune);
                    nn.prune(lastHiddenLayerNo, nodeToPrune);
                    nodeEvolutionTarget(i) = nodeEvolutionTarget(i) - 1;
                end
            end
            ae.greddyLayerWiseTrain(x, 1, 0.1);
        end
        for j = 1 : numel(ae.layers)-2
            nn.weight{j} = ae.weight{j};
            nn.bias{j}   = ae.bias{j};
        end
        agmmTargetGen = ae.getAgmm();
        
%         Kullback-Leibler Divergence
        try
            KlLossEvolution(i) = ae.updateWeightsByKullbackLeibler(Xs, Xt);
        end
        
        for j = 1 : numel(ae.layers)-2
            nn.weight{j} = ae.weight{j};
            nn.bias{j}   = ae.bias{j};
        end
    end
    if agmmSourceDisc.M() > 1
        agmmSourceDisc.deleteCluster();
    end
    if agmmTargetGen.M() > 1
        agmmTargetGen.deleteCluster();
    end
    trainTime(i) = toc;
    gmmTargetBatch(i) = agmmTargetGen.M();
    gmmSourceBatch(i) = agmmSourceDisc.M();
    
    %% Print metrics
    nodeEvolution(i, :) = nn.layers(2 : end - 1);
    
    fprintf('Minibatch: %d/%d\n', i, dm.nMinibatches);
    fprintf('Total of samples: %d Source | %d Target\n', size(Xs,1), size(Xt,1));
    fprintf('Max Mean Min Now Accu Training time: %f %f %f %f %f\n', max(trainTime(1:i)), mean(trainTime(1:i)), min(trainTime(1:i)), trainTime(i), sum(trainTime(1:i)));
    fprintf('Max Mean Min Now Accu Testing time: %f %f %f %f %f\n', max(testTime(1:i)), mean(testTime(1:i)), min(testTime(1:i)), testTime(i), sum(testTime(1:i)));
    fprintf('Max Mean Min Now AGMM Source: %d %f %d %d\n', max(agmmSourceDiscSize), mean(agmmSourceDiscSize), min(agmmSourceDiscSize), agmmSourceDiscSize(end));
    fprintf('Max Mean Min Now AGMM Target: %d %f %d %d\n', max(agmmTargetGenSize), mean(agmmTargetGenSize), min(agmmTargetGenSize), agmmTargetGenSize(end));
    fprintf('Max Mean Min Now CR: %f%% %f%% %f%% %f%%\n', max(crTarget(2:i)) * 100., mean(crTarget(2:i)) * 100., min(crTarget(2:i)) * 100., crTarget(i) * 100.);
    fprintf('Max Mean Min Now Classification Loss: %f %f %f %f\n', max(classificationLoss(2:i)), mean(classificationLoss(2:i)), min(classificationLoss(2:i)), classificationLoss(i));
    fprintf('Max Mean Min Now KL: %f %f %f %f\n', max(KlLossEvolution(2:i)), mean(KlLossEvolution(2:i)), min(KlLossEvolution(2:i)), KlLossEvolution(i));
    fprintf('Max Mean Min Now Nodes: %d %f %d %d\n', max(nodeEvolution(2:i)), mean(nodeEvolution(2:i)), min(nodeEvolution(2:i)), nodeEvolution(i));
    fprintf('Network structure: %s (Discriminative) | %s (Generative)\n', num2str(nn.layers(:).'), num2str(ae.layers(:).'));
    fprintf('\n');
end

%% Plots
plotTime(trainTime, testTime);
plotAGMM(agmmSourceDiscSize, agmmTargetGenSize);
plotNodeEvolution(nodeEvolution);
plotLosses(classificationLoss, discriminativeLoss, generativeLossTarget, KlLossEvolution);
plotClassificationRate(crSource, crTarget, dm.nMinibatches);
plotBIAS2andVAR(nn.BIAS2, nn.VAR);
plotBIAS2andVARGen(ae.BIAS2, ae.VAR);

%% ---------------------------- Plotters ----------------------------------
function plotTime(trainTime, testTime)
    figure('Name', 'Processing Time', 'NumberTitle', 'off');
    hold on
    
    ylim([0 max(max(trainTime), max(testTime)) * 1.1]);
    xlim([1 size(trainTime, 1)]);
    
    pTrain = plot(trainTime);
    pTest  = plot(testTime);
    
    if max(trainTime) > 1
        text(find(trainTime == max(trainTime(trainTime > 1)), 1), max(trainTime(trainTime > 1)),...
            strcat('\leftarrow Max Train Time:', {' '}, string(max(trainTime(trainTime > 1)))),...
            'FontSize', 8,...
            'Color', 'black');
        text(find(trainTime == min(trainTime(trainTime > 1)), 1), min(trainTime(trainTime > 1)),...
            strcat('\leftarrow Min Train Time:', {' '}, string(min(trainTime(trainTime > 1)))),...
            'FontSize', 8,...
            'Color', 'black');
    end
    if max(testTime) > 1
        text(find(testTime == max(testTime(testTime > 1)), 1), max(testTime(testTime > 1)),...
            strcat('\leftarrow Max Test Time:', {' '}, string(max(testTime(testTime > 1)))),...
            'FontSize', 8,...
            'Color', 'black');
        text(find(testTime == min(testTime(testTime > 1)), 1), min(testTime(testTime > 1)),...
            strcat('\leftarrow Min Test Time:', {' '}, string(min(testTime(testTime > 1)))),...
            'FontSize', 8,...
            'Color', 'black');
    end
    
    legend([pTrain,...
        pTest], [strcat('Train Time Mean | Accumulative:', {' '}, string(mean(trainTime)), {' | '}, string(sum(trainTime))),...
        strcat('Test Time Mean | Accumulative:',  {' '},  string(mean(testTime)), {' | '}, string(sum(testTime)))]);
    
    
    ylabel('Time in seconds');
    xlabel('Minibatches');
    
    hold off
end

function plotNodeEvolution(nodeEvolution)
    figure('Name', 'Node Evolution', 'NumberTitle', 'off');
    hold on
    
    ylim([0 max(nodeEvolution, [], 'all') * 1.1]);
    xlim([1 size(nodeEvolution, 1)]);
    
    plotArray   = [];
    legendArray = [];
    for i = 1 : size(nodeEvolution, 2)
        p = plot(nodeEvolution(:, i));
        plotArray   = [plotArray, p];
        legendArray = [legendArray, strcat('HL', {' '}, string(i), {' '}, 'mean:', {' '}, string(mean(nodeEvolution(nodeEvolution(:, i) > 0, i))))];
        
        text(find(nodeEvolution(:, i) == max(nodeEvolution(:, i)), 1), max(nodeEvolution(:, i)),...
            strcat('\leftarrow Max nodes HL ', {' '}, string(i), ':', {' '}, string(max(nodeEvolution(:, i)))),...
            'FontSize', 8,...
            'Color', 'black');
        text(find(nodeEvolution(:, i) == min(nodeEvolution(nodeEvolution(:, i) > 0, i)), 1), min(nodeEvolution(nodeEvolution(:, i) > 0, i)),...
            strcat('\leftarrow Min nodes HL ', {' '}, string(i), ':', {' '}, string(min(nodeEvolution(nodeEvolution(:, i) > 0, i)))),...
            'FontSize', 8,...
            'Color', 'black');
    end
    
    ylabel('Number of nodes');
    xlabel('Minibatches');
    
    legend(plotArray, legendArray);
    
    hold off
end

function plotAGMM(agmmSource, agmmTarget)
    figure('Name', 'Number of GMMs on AGMMs', 'NumberTitle', 'off');
    hold on
    
    ylim([0 max(max(agmmTarget), max(agmmSource)) * 1.1]);
    xlim([1 size(agmmSource, 2)]);
    
    pAgmmSource    = plot(agmmSource);
    pAgmmTarget    = plot(agmmTarget);
    
    if max(agmmSource) > 1
        text(find(agmmSource == max(agmmSource(agmmSource > 1)), 1), max(agmmSource(agmmSource > 1)),...
            strcat('\leftarrow Max GMMs Source Discriminative:', {' '}, string(max(agmmSource(agmmSource > 1)))),...
            'FontSize', 8,...
            'Color', 'black');
        text(find(agmmSource == min(agmmSource(agmmSource > 1)), 1), min(agmmSource(agmmSource > 1)),...
            strcat('\leftarrow Min GMMs Source Discriminative:', {' '}, string(min(agmmSource(agmmSource > 1)))),...
            'FontSize', 8,...
            'Color', 'black');
    end
    if max(agmmTarget) > 1
        text(find(agmmTarget == max(agmmTarget(agmmTarget > 1)), 1), max(agmmTarget(agmmTarget > 1)),...
            strcat('\leftarrow Max GMMs Target Generative:', {' '}, string(max(agmmTarget(agmmTarget > 1)))),...
            'FontSize', 8,...
            'Color', 'black');
        text(find(agmmTarget == min(agmmTarget(agmmTarget > 1)), 1), min(agmmTarget(agmmTarget > 1)),...
            strcat('\leftarrow Min GMMs Target Generative:', {' '}, string(min(agmmTarget(agmmTarget > 1)))),...
            'FontSize', 8,...
            'Color', 'black');
    end
    
    legend([pAgmmSource,...
        pAgmmTarget], [strcat('AGMM Source Discriminative Mean:', {' '}, string(mean(agmmSource))),...
        strcat('AGMM Target Generative Mean:', {' '},     string(mean(agmmTarget)))]);
    
    
    ylabel('Number of GMMs');
    xlabel('Samples');
    
    hold off
end

function plotLosses(classificationLoss, discriminativeLoss, generativeTargetLoss, kullbackLeiblerLoss)
    figure('Name', 'Losses', 'NumberTitle', 'off');
    hold on
    
    ylim([0 max(max(kullbackLeiblerLoss), max(max(max(classificationLoss), max(discriminativeLoss)), max(generativeTargetLoss))) * 1.1]);
    xlim([1 size(classificationLoss, 1)]);
    
    pClassificationLoss   = plot(classificationLoss);
    pDiscriminativeLoss   = plot(discriminativeLoss);
    pGenerativeTargetLoss = plot(generativeTargetLoss);
    pKullbackLeiblerLoss  = plot(kullbackLeiblerLoss);
    
    text(find(classificationLoss == max(classificationLoss), 1), max(classificationLoss),...
        strcat('\leftarrow Max Classification Loss:', {' '}, string(max(classificationLoss))),...
        'FontSize', 8,...
        'Color', 'black');
    text(find(classificationLoss == min(classificationLoss), 1), min(classificationLoss),...
        strcat('\leftarrow Min Classification Loss:', {' '}, string(min(classificationLoss))),...
        'FontSize', 8,...
        'Color', 'black');
    
    text(find(discriminativeLoss == max(discriminativeLoss), 1), max(discriminativeLoss),...
        strcat('\leftarrow Max Discriminative Loss:', {' '}, string(max(discriminativeLoss))),...
        'FontSize', 8,...
        'Color', 'black');
    text(find(discriminativeLoss == min(discriminativeLoss), 1), min(discriminativeLoss),...
        strcat('\leftarrow Min Discriminative Loss:', {' '}, string(min(discriminativeLoss))),...
        'FontSize', 8,...
        'Color', 'black');
    
    text(find(generativeTargetLoss == max(generativeTargetLoss), 1), max(generativeTargetLoss),...
        strcat('\leftarrow Max Generative Target Loss:', {' '}, string(max(generativeTargetLoss))),...
        'FontSize', 8,...
        'Color', 'black');
    text(find(generativeTargetLoss == min(generativeTargetLoss), 1), min(generativeTargetLoss),...
        strcat('\leftarrow Min Generative Target Loss:', {' '}, string(min(generativeTargetLoss))),...
        'FontSize', 8,...
        'Color', 'black');
    
    text(find(kullbackLeiblerLoss == max(kullbackLeiblerLoss), 1), max(kullbackLeiblerLoss),...
        strcat('\leftarrow Max KL Div Loss:', {' '}, string(max(kullbackLeiblerLoss))),...
        'FontSize', 8,...
        'Color', 'black');
    text(find(kullbackLeiblerLoss == min(kullbackLeiblerLoss), 1), min(kullbackLeiblerLoss),...
        strcat('\leftarrow Min KL Div Loss:', {' '}, string(min(kullbackLeiblerLoss))),...
        'FontSize', 8,...
        'Color', 'black');
    
    ylabel('Loss Value');
    xlabel('Minibatches');
    
    legend([pClassificationLoss,...
        pDiscriminativeLoss,...
        pGenerativeTargetLoss,...
        pKullbackLeiblerLoss], [strcat('Classification Loss Mean:', {' '}, string(mean(classificationLoss(2:end)))),...
        strcat('Discriminative Loss Mean:', {' '}, string(mean(discriminativeLoss))),...
        strcat('Generative Target Loss Mean:', {' '}, string(mean(generativeTargetLoss))),...
        strcat('Kullback Leibler Divergence Loss Mean:', {' '}, string(mean(kullbackLeiblerLoss)))]);
    
    hold off
end

function plotClassificationRate(source, target, nMinibatches)
    figure('Name', 'Source and Target Classification Rates', 'NumberTitle', 'off');
    hold on
    ylim([0 max(max(source), max(target)) * 1.1]);
    xlim([1 nMinibatches]);
    
    niceBlue    = [0      0.4470 0.7410];
    niceYellow  = [0.8500 0.3250 0.0980];
    
    pSource = plot(source, 'Color', niceYellow, 'LineStyle', ':');
    pTarget = plot(target, 'Color', niceBlue);
    
    text(find(source == max(source), 1), max(source),...
        strcat('\leftarrow Max Source:', {' '}, string(max(source))),...
        'FontSize', 8,...
        'Color', 'black');
    text(find(source == min(source), 1), min(source),...
        strcat('\leftarrow Min Source:', {' '}, string(min(source))),...
        'FontSize', 8,...
        'Color', 'black');
    text(find(target == max(target), 1), max(target),...
        strcat('\leftarrow Max Target:', {' '}, string(max(target))),...
        'FontSize', 8,...
        'Color', 'black');
    text(find(target == min(target), 1), min(target),...
        strcat('\leftarrow Min Target:', {' '}, string(min(target))),...
        'FontSize', 8,...
        'Color', 'black');
    
    ylabel('Classification Rate');
    xlabel('Minibatches');
    
    legend([pSource, pTarget], [strcat('Source Mean:', {' '}, string(mean(source(2:end)))),...
        strcat('Target Mean:', {' '}, string(mean(target(2:end))))]);
    
    hold off
    
end

function plotBIAS2andVAR(BIAS2, VAR)
    sampleLayerCount = zeros(1, size(BIAS2, 2));
    yAxisLim  = 0;
    bias2 = [];
    var   = [];
    for i = 2 : size(BIAS2, 2)
        sampleLayerCount(i) = sampleLayerCount(i - 1) + size(BIAS2{i}, 2);
        for j = 1 : size(BIAS2{i}, 2)
            bias2     = [bias2, BIAS2{i}(j)];
            var       = [var,   VAR{i}(j)];
            yAxisLim  = max(yAxisLim, bias2(end) + var(end));
        end
    end
    clear BIAS2 VAR
    
    figure('Name', 'BIAS2, VAR, and NS', 'NumberTitle', 'off');
    hold on
    
    ylim([0 max(max(bias2), max(var)) * 1.1]);
    xlim([1 size(bias2, 2)]);
    
    p1 = plot(bias2);
    p2 = plot(var);
    p3 = plot(bias2 + var);
    for j = 1: ceil(size(bias2, 2)/4) : size(bias2, 2)
        if ~isnan(bias2(j))
            text(j, bias2(j),...
                strcat('\leftarrow', {' '}, 'BIAS2 =', {' '}, string(bias2(j))),...
                'FontSize', 8);
        end
    end
    text(size(bias2, 2), bias2(end), string(bias2(end)));
    
    for j = 1: ceil(size(var, 2)/4) : size(var, 2)
        if ~isnan(var(j))
            text(j, var(j),...
                strcat('\leftarrow', {' '}, 'VAR =', {' '}, string(var(j))),...
                'FontSize', 8);
        end
    end
    text(size(var, 2), var(end), string(var(end)));
    
    for j = 1: ceil(size(var + bias2, 2)/4) : size(var + bias2, 2)
        if ~isnan(var(j)) && ~isnan(bias2(j)) && ~isnan(var(j) + bias2(j))
            text(j, var(j) + bias2(j),...
                strcat('\leftarrow', {' '}, 'NS =', {' '}, string(var(j) + bias2(j))),...
                'FontSize', 8);
        end
    end
    text(size(var + bias2, 2), var(end) + bias2(end), string(var(end) + bias2(end)));
    
    for i = 2 : size(sampleLayerCount, 2) - 1
        line([sampleLayerCount(i), sampleLayerCount(i)], [-yAxisLim * 2 yAxisLim * 2],...
            'LineStyle', ':',...
            'Color', 'magenta');
    end
    
    ylabel('Value');
    xlabel('Sample');
    
    legend([p1, p2, p3], [strcat('BIAS2 Mean:', {' '}, string(mean(bias2(2:end)))),...
        strcat('VAR Mean:', {' '}, string(mean(var(2:end)))),...
        strcat('NS Mean:', {' '}, string(mean(var(2:end) + bias2(2:end))))]);
    hold off
end

function plotBIAS2andVARGen(BIAS2, VAR)
    sampleLayerCount = zeros(1, size(BIAS2, 2));
    yAxisLim  = 0;
    bias2 = [];
    var   = [];
    for i = 2 : size(BIAS2, 2)
        sampleLayerCount(i) = sampleLayerCount(i - 1) + size(BIAS2{i}, 2);
        for j = 1 : size(BIAS2{i}, 2)
            bias2     = [bias2, BIAS2{i}(j)];
            var       = [var,   VAR{i}(j)];
            yAxisLim  = max(yAxisLim, bias2(end) + var(end));
        end
    end
    clear BIAS2 VAR
    
    figure('Name', 'BIAS2, VAR, and NS Generative', 'NumberTitle', 'off');
    hold on
    
    ylim([0 max(max(bias2), max(var)) * 1.1]);
    xlim([1 size(bias2, 2)]);
    
    p1 = plot(bias2);
    p2 = plot(var);
    p3 = plot(bias2 + var);
    for j = 1: ceil(size(bias2, 2)/4) : size(bias2, 2)
        if ~isnan(bias2(j))
            text(j, bias2(j),...
                strcat('\leftarrow', {' '}, 'BIAS2 =', {' '}, string(bias2(j))),...
                'FontSize', 8);
        end
    end
    text(size(bias2, 2), bias2(end), string(bias2(end)));
    
    for j = 1: ceil(size(var, 2)/4) : size(var, 2)
        if ~isnan(var(j))
            text(j, var(j),...
                strcat('\leftarrow', {' '}, 'VAR =', {' '}, string(var(j))),...
                'FontSize', 8);
        end
    end
    text(size(var, 2), var(end), string(var(end)));
    
    for j = 1: ceil(size(var + bias2, 2)/4) : size(var + bias2, 2)
        if ~isnan(var(j)) && ~isnan(bias2(j)) && ~isnan(var(j) + bias2(j))
            text(j, var(j) + bias2(j),...
                strcat('\leftarrow', {' '}, 'NS =', {' '}, string(var(j) + bias2(j))),...
                'FontSize', 8);
        end
    end
    text(size(var + bias2, 2), var(end) + bias2(end), string(var(end) + bias2(end)));
    
    for i = 2 : size(sampleLayerCount, 2) - 1
        line([sampleLayerCount(i), sampleLayerCount(i)], [-yAxisLim * 2 yAxisLim * 2],...
            'LineStyle', ':',...
            'Color', 'magenta');
    end
    
    ylabel('Value');
    xlabel('Sample');
    
    legend([p1, p2, p3], [strcat('BIAS2 Mean:', {' '}, string(mean(bias2))),...
        strcat('VAR Mean:', {' '}, string(mean(var))),...
        strcat('NS Mean:', {' '}, string(mean(var + bias2)))]);
    hold off
end