% Marcus Vinicius Sousa Leite de Carvalho
% marcus.decarvalho@ntu.edu.sg
%
% NANYANG TECHNOLOGICAL UNIVERSITY - NTUITIVE PTE LTD Dual License Agreement
% Non-Commercial Use Only 
% This NTUITIVE License Agreement, including all exhibits ("NTUITIVE-LA") is a legal agreement between you and NTUITIVE (or �we�) located at 71 Nanyang Drive, NTU Innovation Centre, #01-109, Singapore 637722, a wholly owned subsidiary of Nanyang Technological University (�NTU�) for the software or data identified above, which may include source code, and any associated materials, text or speech files, associated media and "online" or electronic documentation and any updates we provide in our discretion (together, the "Software"). 
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

classdef ElasticNodes < handle
    %ELASTICNODES It encapsulate global variables necessary for width
    %adaptation
    %
    %   This class enabless elastic network width. Network width adaptation
    %   supports automatic generation of new hidden nodes and prunning of 
    %   inconsequential nodes. This mechanism is controlled by the NS 
    %   (Network Significance) method which estimates the network 
    %   generalization power in terms of bias and variance
    properties (Access = public)
        
        growable; % See full comment below
        % Hold an array of boolean elements indicating if that layer can 
        % receive grow or not during width adaptation procedure
        
        prunable; % See full comment below
        % Hold an array of integer elements indicating if that layer can 
        % receive prune or not during width adaptation procedure.
        % 0 indicates that no node should be pruned. Anything different
        % than zero indicantes which node should be pruned in that layer. 
    end
    properties (Access = public)
        dataMean = 0;
        dataStd = 0;
        dataVar = 0;
        
        nSamplesFeed = 0;
        nSamplesLayer;
        
        % NS = Network Significance
        %BIAS VARIABLES
        meanBIAS;
        varBIAS;
        stdBIAS;
        minMeanBIAS;
        minStdBIAS;
        BIAS2;
        %VAR VARIABLES
        meanVAR;
        varVAR;
        stdVAR;
        minMeanVAR;
        minStdVAR;
        VAR;
        
        % metrics
        nodeEvolution = {}; % TODO: Need to include at the grow/prune part
    end
    %% Evolving layers properties
    properties (Access = public)
        alpha = 0.005;
        gradientBias = [];
        meanNetBias2;
        meanNetVar;
    end
    
    methods (Access = protected)
        function self = ElasticNodes(nHiddenLayers)
            nhl = nHiddenLayers; % readability
            
            self.nSamplesLayer = zeros(1,nhl);
            self.meanBIAS = zeros(1,nhl);
            self.varBIAS = zeros(1,nhl);
            self.stdBIAS = zeros(1,nhl);
            self.minMeanBIAS = ones(1,nhl) * inf;
            self.minStdBIAS = ones(1,nhl) * inf;
            self.BIAS2 = num2cell(zeros(1,nhl));
            
            self.meanVAR = zeros(1,nhl);
            self.varVAR = zeros(1,nhl);
            self.stdVAR = zeros(1,nhl);
            self.minMeanVAR = ones(1,nhl) * inf;
            self.minStdVAR = ones(1,nhl) * inf;
            self.VAR = num2cell(zeros(1,nhl));
        
            self.growable = zeros(1,nhl);
%             self.prunable = zeros(1,nhl);
            self.prunable = cell(1,nhl);
            for i = 1 : nhl
               self.prunable{i} = 0;
            end
        end
        
        function growLayerEvolutiveParameter(self, numberHiddenLayers)
            nhl = numberHiddenLayers; %readability
            
            self.nSamplesLayer = [self.nSamplesLayer, 0];
            self.meanBIAS      = [self.meanBIAS, 0];
            self.varBIAS       = [self.varBIAS, 0];
            self.stdBIAS       = [self.stdBIAS, 0];
            self.minMeanBIAS   = [self.minMeanBIAS, 0];
            self.minStdBIAS    = [self.minStdBIAS, 0];
            self.BIAS2         = [self.BIAS2, 0];
            
            self.meanVAR    = [self.meanVAR, 0];
            self.varVAR     = [self.varVAR, 0];
            self.stdVAR     = [self.stdVAR, 0];
            self.minMeanVAR = [self.minMeanVAR, 0];
            self.minStdVAR  = [self.minStdVAR, 0];
            self.VAR        = [self.VAR, 0];
            
            self.growable = zeros(1, nhl + 1);
            self.prunable = cell(1, nhl + 1);
            for i = 1 : nhl + 1
               self.prunable{i} = 0;
            end
        end
    end
end

