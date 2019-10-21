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

classdef AutoEncoder < NeuralNetwork
    %AutoEncoder
    %   This object mimics the behavior of a Auto Encoder network, which is
    %   a Neural Network that has the output equal to input.
    %   This object has elastic habilities, being able to grow and prune
    %   nodes automatically.
    %   TODO: Provide the paper or study material for the Auto Encoder
     
    properties (Access = protected)
        greedyLayerBias       = [];
        greedyLayerOutputBias;
    end
    methods (Access = public)
        function self = AutoEncoder(layers)
            %   AutoEncoder
            %   layers (array)
            %       This array describes a FeedForward Network structure by
            %       the number of layers on it.
            %       An FFNN with an input layer of 8 nodes, a hidden layer
            %       of 10 nodes and an output layer of 3 nodes would be
            %       described by [8 10 3].
            %       An FFNN with an input layer of 784 nodes, a hidden
            %       layer 1 of 800 nodes, a hidden layer 2 of 400 nodes and
            %       an output layer of 10 nodes would be described as [784 800 400 10]
            
            self@NeuralNetwork(layers);
            self.outputActivationFunctionLossFunction = self.ACTIVATION_LOSS_FUNCTION_SIGMOID_MSE();
        end        
        
        function test(self, X)
            % test
            %   See test@NeuralNetwork
            %   X (matrix)
            %       Input and output data
            test@NeuralNetwork(self, X, X)
        end
        
        function grow(self, layerNo)
            grow@NeuralNetwork(self, layerNo);
            self.growGreedyLayerBias(layerNo);
        end
        
        function prune(self, layerNo, nodeNo)
            prune@NeuralNetwork(self, layerNo, nodeNo);
            self.pruneGreedyLayerBias(layerNo, nodeNo);
        end
        
        function growGreedyLayerBias(self, layerNo)
            b = layerNo; %readability
            if b == (numel(self.layers) - 1)
                self.greedyLayerOutputBias = [self.greedyLayerOutputBias normrnd(0, sqrt(2 / (self.layers(end-1) + 1)))];
            else
                self.greedyLayerBias{b} = [self.greedyLayerBias{b} normrnd(0, sqrt(2 / (self.layers(b) + 1)))];
            end
            
        end
        
        function growLayer(self, option, numberOfNodes)
            if option == self.CREATE_MIRRORED_LAYER()
                nhl = self.nHiddenLayers + 1;
                growLayer@NeuralNetwork(self, self.CREATE_LAYER_BY_ARGUMENT(), numberOfNodes);
                growLayer@NeuralNetwork(self, self.CREATE_LAYER_BY_ARGUMENT(), self.layers(nhl));
            else
                growLayer@NeuralNetwork(self, option, numberOfNodes);
                self.greedyLayerBias{size(self.greedyLayerBias, 2) + 1} = self.greedyLayerOutputBias;
                self.greedyLayerOutputBias = normrnd(0, sqrt(2 / (self.layers(end-1) + 1)));
            end
        end
        
        function pruneGreedyLayerBias(self, layerNo, nodeNo)
            b = layerNo; % readability
            n = nodeNo;   %readability
            if b == (numel(self.layers) - 1)
                self.greedyLayerOutputBias(n) = [];
            else
                self.greedyLayerBias{b}(n) = [];
            end
        end
        
        function greddyLayerWiseTrain(self, X, nEpochs, noiseRatio)
            %greddyLayerWiseTrain
            %   Performs Greedy Layer Wise train
            %   X (matrix)
            %       Input and output data
            %   nEpochs (integer)
            %       The number of epochs which the greedy layer wise train
            %       will occurs. If you are running a single pass model,
            %       you want this to be equal one.
            if nargin == 3
                noiseRatio = 0;
            end
%             disp(self.layers)
            for i = 1 : numel(self.layers) - 1
                self.forwardpass(X);
                trainingX = self.layerValue{i};
                Xnoise = (rand(size(trainingX)) >= noiseRatio) .* trainingX;
                
                if i > self.nHiddenLayers
                    nn = NeuralNetwork([self.layers(i) self.layers(end) self.layers(i)]);
                else
                    nn = NeuralNetwork([self.layers(i) self.layers(i+1) self.layers(i)]);
                end
                nn.outputActivationFunctionLossFunction = self.ACTIVATION_LOSS_FUNCTION_SIGMOID_MSE();
                
                if i > self.nHiddenLayers
                    nn.weight{1}    = self.outputWeight;
                    nn.bias{1}      = self.outputBias;
                    nn.outputWeight = self.outputWeight';
                    if isempty(self.greedyLayerOutputBias)
                        self.greedyLayerOutputBias = normrnd(0, sqrt(2 / (size(self.outputWeight', 2) + 1)),...
                                                             1, size(self.outputWeight', 1));
                        nn.outputBias   = self.greedyLayerOutputBias;
                    else
                        nn.outputBias   = self.greedyLayerOutputBias;
                    end
                else
                    nn.weight{1}    = self.weight{i};
                    nn.bias{1}      = self.bias{i};
                    nn.outputWeight = self.weight{i}';
                    try
                        nn.outputBias   = self.greedyLayerBias{i};
                    catch
                        self.greedyLayerBias{i} = normrnd(0, sqrt(2 / (size(self.weight{i}', 2) + 1)),...
                                                          1, size(self.weight{i}', 1));
                        nn.outputBias   = self.greedyLayerBias{i};
                    end
                end
                
                for j = 1 : nEpochs
                    nn.train(Xnoise, trainingX);
                end
                
                if i > self.nHiddenLayers
                    self.outputWeight = nn.weight{1};
                    self.outputBias   = nn.bias{1};
                else
                    self.weight{i} = nn.weight{1};
                    self.bias{i}   = nn.bias{1};
                end
            end
        end
        
        function loss = updateWeightsByKullbackLeibler(self, Xs, Xt, GAMMA)
            if nargin == 3
                GAMMA = 0.0001;
            end
            loss = updateWeightsByKullbackLeibler@NeuralNetwork(self, Xs, Xs, Xt, Xt, GAMMA);
        end
    end
    methods (Access = protected)
        function BIAS2 = computeBIAS2(~, Ez, y)
            %getBIAS2
            %   The way AutoEncoders calculata its BIAS2 value per layer is
            %   different than normal neural networks. Because we use
            %   sigmoid as our output activation function, and because the
            %   error is too high, we prefer use mean as a way to squish
            %   the bias2
            %   Ez (double, vector or matrix)
            %       Expected outbound value of that layer
            %   y (double, vector or matrix)
            %       A target class
            %
            %   return BIAS2 = The network squared BIAS
            BIAS2 = mean((Ez - y') .^ 2);
        end
        function var = computeVAR(~, Ez, Ez2)
            %getVAR
            %   The way AutoEncoders calculata its VAR value per layer is
            %   different than normal neural networks. Because we use
            %   sigmoid as our output activation function, and because the
            %   error is too high, we prefer use mean as a way to squish
            %   the bias2
            %   Ez (double, vector or matrix)
            %       Expected outbound value of that layer
            %   Ez2 (double, vector or matrix)
            %       Expected outbound squared value of that layer
            %
            %   return VAR = The network VAR (variance)
            var = mean(Ez2 - Ez .^ 2);
        end
    end
end

