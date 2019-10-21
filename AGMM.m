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


classdef AGMM < handle
    properties (Access = public)
        gmmArray = [];
        nSamplesFeed = 0;
        rho = 0.1;
        nFeatures;
    end
    
    methods (Access = public)
        function run(self, x, bias2)
            self.nSamplesFeed = self.nSamplesFeed + 1;
            if size(self.gmmArray, 1) == 0
                self.gmmArray = [self.gmmArray; GMM(x)];
                
                self.nFeatures = size(x, 2);
            else
                self.computeInference(x);
                
                [~, gmmWinnerIdx] = max(self.updateWeights());
                if self.M() > 1
                    self.computeOverlapsDegree(gmmWinnerIdx, 3, 3);
                end
                
                denominator = 1.25 * exp(-bias2) + 0.75 * self.nFeatures;
                numerator   = 4 - 2 * exp( -self.nFeatures / 2);
                threshold = exp(- denominator / numerator);
                if self.gmmArray(gmmWinnerIdx).inference < threshold ...
                        && self.gmmArray(gmmWinnerIdx).hyperVolume > self.rho * (self.computeSumHyperVolume() - self.gmmArray(gmmWinnerIdx).hyperVolume)...
                        && self.nSamplesFeed > 10
                    
                    % Create a new cluster
                    self.createCluster(x);
                    self.gmmArray(end).var = (x - self.gmmArray(gmmWinnerIdx).center) .^ 2;
                else
                    % Update the winning cluster
                    self.updateCluster(x, self.gmmArray(gmmWinnerIdx));
                end
            end
        end
        
        function createCluster(self, x)
            self.gmmArray = [self.gmmArray; GMM(x)];
            
            weightSum = 0;
            for i = 1 : size(self.gmmArray, 1)
                weightSum = weightSum + self.gmmArray(i).weight;
            end
            for i = 1 : size(self.gmmArray, 1)
                self.gmmArray(i).weight = self.gmmArray(i).weight/weightSum;
            end
        end
        
        function updateCluster(~, x, gmm)
            gmm.winCounter = gmm.winCounter + 1;
            gmm.center     = gmm.center +  (x - gmm.center) / gmm.winCounter;
            gmm.var        = gmm.var    + ((x - gmm.center) .^ 2 - gmm.var) / gmm.winCounter;
        end
        
        function deleteCluster(self)
            accu_e = zeros(1, size(self.gmmArray, 1));
            for i = 1 : size(self.gmmArray, 1)
                accu_e(i) = self.gmmArray(i).inferenceSum / self.gmmArray(i).surviveCounter;
            end
            accu_e(isnan(accu_e)) = [];
            deleteList = find(accu_e <= mean(accu_e) - 0.5 * std(accu_e));
            
            if ~isempty(deleteList)
                self.gmmArray(deleteList) = [];
                accu_e(deleteList) = [];
            end
            
            sumWeight = 0;
            for i = 1 : size(self.gmmArray, 1)
                sumWeight = sumWeight + self.gmmArray(i).weight;
            end
            if sumWeight == 0
                [~, maxIdx] = max(accu_e);
                self.gmmArray(maxIdx).weight = self.gmmArray(i).weight + 1;
            end
            
            sumWeight = 0;
            for i = 1 : size(self.gmmArray, 1)
                sumWeight = sumWeight + self.gmmArray(i).weight;
            end
            for i = 1 : size(self.gmmArray, 1)
                self.gmmArray(i).weight = self.gmmArray(i).weight / sumWeight;
            end
        end
        
        function hyperVolume = computeSumHyperVolume(self)
            hyperVolume = 0;
            for i = 1 : size(self.gmmArray, 1)
                hyperVolume = hyperVolume + self.gmmArray(i).hyperVolume;
            end
        end
        
        function computeInference(self, x, y)
            for i = 1 : size(self.gmmArray, 1)
                gmm = self.gmmArray(i);
                
                if nargin == 3
                    gmm.computeInference(x, y);
                else
                    gmm.computeInference(x);
                end
            end
        end
        
        function weights = updateWeights(self)
            denumerator  = zeros(1, size(self.gmmArray, 1));
            probX_J      = zeros(1, size(self.gmmArray, 1));
            probJ        = zeros(1, size(self.gmmArray, 1));
            probX_JprobJ = zeros(1, size(self.gmmArray, 1));
            weights      = zeros(1, size(self.gmmArray, 1));
            
            sumWinCounter = 0;
            maxInference = 0;
            maxInferenceIdx = 1;
            for i = 1 : size(self.gmmArray, 1)
                sumWinCounter = sumWinCounter + self.gmmArray(i).winCounter;
                if self.gmmArray(i).inference > maxInference
                    maxInference = self.gmmArray(i).inference;
                    maxInferenceIdx = i;
                end
            end
            
            for i = 1 : size(self.gmmArray, 1)
                self.gmmArray(i).inferenceSum   = self.gmmArray(i).inferenceSum   + self.gmmArray(i).inference;
                self.gmmArray(i).surviveCounter = self.gmmArray(i).surviveCounter + 1;

                denumerator(i) = sqrt(2 * pi * self.gmmArray(i).hyperVolume);
                probX_J(i) = denumerator(i) .* self.gmmArray(i).inference;
                probJ(i)   = self.gmmArray(i).winCounter / sumWinCounter;
                probX_JprobJ(i) = probX_J(i) * probJ(i);
            end
            
            if sum(probX_JprobJ) == 0
                probX_JprobJ(maxInferenceIdx) = probX_JprobJ(maxInferenceIdx) + 1;
            end
            
            for i = 1 : size(self.gmmArray, 1)
                self.gmmArray(i).weight = probX_JprobJ(i) / sum(probX_JprobJ);
                weights(i) = self.gmmArray(i).weight;
            end
        end
        
        function computeOverlapsDegree(self, gmmWinnerIdx, maximumLimit, minimumLimit)
            if nargin == 2
                maximumLimit = 3;
                minimumLimit = maximumLimit;
            elseif nargin == 3
                minimumLimit = maximumLimit;
            end
            maximumLimit = abs(maximumLimit);
            minimumLimit = abs(minimumLimit);
            
            nGMM = size(self.gmmArray, 1);
            overlap_coefficient = 1/(nGMM-1);
            
            sigmaMaximumWinner = maximumLimit * sqrt(self.gmmArray(gmmWinnerIdx).var);
            sigmaMinimumWinner = minimumLimit * sqrt(self.gmmArray(gmmWinnerIdx).var);
            
            if maximumLimit == minimumLimit
                miu_plus_sigma_winner = self.gmmArray(gmmWinnerIdx).center + sigmaMaximumWinner;
                miu_mins_sigma_winner = self.gmmArray(gmmWinnerIdx).center - sigmaMinimumWinner;
            else
                miu_plus_sigma_winner =   sigmaMinimumWinner + sigmaMaximumWinner;
                miu_mins_sigma_winner = -sigmaMinimumWinner -sigmaMaximumWinner;
            end
            
            miu_plus_sigma    = zeros(nGMM, self.nFeatures);
            miu_mins_sigma    = zeros(nGMM, self.nFeatures);
            overlap_mins_mins = zeros(1, nGMM);
            overlap_mins_plus = zeros(1, nGMM);
            overlap_plus_mins = zeros(1, nGMM);
            overlap_plus_plus = zeros(1, nGMM);
            overlap_score     = zeros(1, nGMM);
            
            for i = 1 : nGMM
                sigmaMaximum = maximumLimit * sqrt(self.gmmArray(i).var);
                sigmaMinimum = minimumLimit * sqrt(self.gmmArray(i).var);
                
                if maximumLimit == minimumLimit
                    miu_plus_sigma(i, :) = self.gmmArray(i).center + sigmaMaximum;
                    miu_mins_sigma(i, :) = self.gmmArray(i).center - sigmaMaximum;
                else
                    miu_plus_sigma(i, :) = sigmaMinimum  + sigmaMaximum;
                    miu_mins_sigma(i, :) = -sigmaMinimum - sigmaMaximum;
                end
                
                overlap_mins_mins(i) = mean(miu_mins_sigma(i,:) - miu_mins_sigma_winner);
                overlap_mins_plus(i) = mean(miu_plus_sigma(i,:) - miu_mins_sigma_winner);
                overlap_plus_mins(i) = mean(miu_mins_sigma(i,:) - miu_plus_sigma_winner);
                overlap_plus_plus(i) = mean(miu_plus_sigma(i,:) - miu_plus_sigma_winner);
                
                condition1 = overlap_mins_mins(i) >= 0 ...
                    && overlap_mins_plus(i) >= 0 ...
                    && overlap_plus_mins(i) <= 0 ...
                    && overlap_plus_plus(i) <= 0;
                condition2 = overlap_mins_mins(i) <= 0 ...
                    && overlap_mins_plus(i) >= 0 ...
                    && overlap_plus_mins(i) <= 0 ...
                    && overlap_plus_plus(i) >= 0;
                condition3 = overlap_mins_mins(i) > 0 ...
                    && overlap_mins_plus(i) > 0 ...
                    && overlap_plus_mins(i) < 0 ...
                    && overlap_plus_plus(i) > 0;
                condition4 = overlap_mins_mins(i) < 0 ...
                    && overlap_mins_plus(i) > 0 ...
                    && overlap_plus_mins(i) < 0 ...
                    && overlap_plus_plus(i) < 0;
                
                if condition1 || condition2
                    % full overlap, the cluster is inside the winning cluster
                    % the score is full score 1/(nGMM-1)
                    overlap_score(i) = overlap_coefficient;
                elseif condition3 || condition4
                    % partial overlap, the score is the full score multiplied
                    % by the overlap degree
                    reward = norm(self.gmmArray(i).center    - self.gmmArray(gmmWinnerIdx).center)...
                                   / norm(self.gmmArray(i).center    + self.gmmArray(gmmWinnerIdx).center)...
                                   + norm(sqrt(self.gmmArray(i).var) - sqrt(self.gmmArray(gmmWinnerIdx).var))...
                                   / norm(sqrt(self.gmmArray(i).var) + sqrt(self.gmmArray(gmmWinnerIdx).var));
                    overlap_score(i) = overlap_coefficient * reward;
                end
            end
            overlap_score(gmmWinnerIdx) = []; % take out the winner score from the array
            self.rho = sum(overlap_score);
            self.rho = min(self.rho, 1);
            self.rho = max(self.rho, 0.1); % Do not let rho = zero
        end
        
        function M = computeNumberOfGmms(self)
            M = size(self.gmmArray, 1);
        end
        
        function M = M(self)
            M = self.computeNumberOfGmms();
        end
    end
end