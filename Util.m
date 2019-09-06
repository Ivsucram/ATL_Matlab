% MIT License
%
% Copyright (c) 2019
% Marcus Vinicius Sousa Leite de Carvalho
% marcus.decarvalho@ntu.edu.sg
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

classdef Util
    methods
        function [mean, var, std] = recursiveMeanStd(~, x, oldMean, oldVar, n)
            %TODO DOC
            %http://www.scalaformachinelearning.com/2015/10/recursive-mean-and-standard-deviation.html
            %mean  = (1 - 1/n)*oldMean + (oldMean./n);
            mean  = oldMean + (x - oldMean)./n;
            var   = oldVar + (x - oldMean) .*(x - mean);
            std   = sqrt(var/n);
        end
        
        function p = probit(~, mean, standardDeviation)
            %probit TODO documentation
            % calculate probit function which has xi^2 = pi/8
            p = (1 + pi .* (standardDeviation .^ 2) ./ 8);
            p = mean ./ sqrt(p);
        end
        
        function dist=KLDiv(~, P,Q)
            %  dist = KLDiv(P,Q) Kullback-Leibler divergence of two discrete probability
            %  distributions
            %  P and Q  are automatically normalised to have the sum of one on rows
            % have the length of one at each
            % P =  n x nbins
            % Q =  1 x nbins or n x nbins(one to one)
            % dist = n x 1
            if size(P,2)~=size(Q,2)
                error('the number of columns in P and Q should be the same');
            end
            if sum(~isfinite(P(:))) + sum(~isfinite(Q(:)))
                error('the inputs contain non-finite values!')
            end
            % normalizing the P and Q
            if size(Q,1)==1
                Q = Q ./sum(Q);
                P = P ./repmat(sum(P,2),[1 size(P,2)]);
                temp =  P.*log(P./repmat(Q,[size(P,1) 1]));
                temp(isnan(temp))=0;% resolving the case when P(i)==0
                dist = sum(temp,2);
                
                
            elseif size(Q,1)==size(P,1)
                
                Q = Q ./repmat(sum(Q,2),[1 size(Q,2)]);
                P = P ./repmat(sum(P,2),[1 size(P,2)]);
                temp =  P.*log(P./Q);
                temp(isnan(temp))=0; % resolving the case when P(i)==0
                dist = sum(temp,2);
            end
        end
    end
end

