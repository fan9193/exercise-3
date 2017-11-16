function [out1,out2,out3,out4] = iteration()
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% input: priors
% prior distribution 
  % higher layer G: G0,A0
  % higher layer Sigma: rho0, R0
  % beta: beta0,B0
  % sigma: v0,delta0
 global k x tx T X z G0 A0 rho0 R0 beta0 B0 v0 delta0;  

 n = size(X,1); 
 p = size(X,2); %5
 %m = 3; %number of parameters in theta
 d = z(:,2);
 
%% functions for mu and lambda
    %likelihood of mu and lambda, given data x,tx,T: L(.|x,tx,T)
    %function out = likelihood1(mu,lambda)
    %    out = lambda.^x.*mu./(lambda + mu).*exp(-(lambda + mu).*tx) + lambda.^(x+1)./(lambda + mu).*exp(-(lambda + mu).*T);%n*1
    %end
        %problem: exp(-tx) can be very small, so multiply by exp(-tx) to calculate likelihood ratio of new to old likehood
        
    function out = likelihoodratio(munew,lambdanew,mu,lambda)
        out = (exp(lambdanew).^x.*exp(munew)./(exp(lambdanew) + exp(munew)).*exp((exp(lambda)+exp(mu)-exp(lambdanew)-exp(munew)).*tx) + exp(lambdanew).^(x+1)./(exp(lambdanew) + exp(munew)).*exp((exp(lambda)+exp(mu)).*tx-(exp(lambdanew) + exp(munew)).*T))./(exp(lambda).^x.*exp(mu)./(exp(lambda) + exp(mu)) + exp(lambda).^(x+1)./(exp(lambda) + exp(mu)).*exp((exp(lambda) + exp(mu)).*(tx-T)));%n*1
    end   

    % conditional density function of mu and lambda, given b, G, Sigma
    function out = condpdf1(mu,lambda,b,G,Sigma)
        Sigma11_temp = Sigma(1:2,1:2);
        Sigma12_temp = Sigma(1:2,3);%2*1
        Sigma21_temp = Sigma(3,1:2);% 1*2
        Sigma22_temp = Sigma(3,3);
        thetamean_temp = X*G; %G: 5*3 -> mean: n*3
        mumean_temp = thetamean_temp(:,1);
        lambdamean_temp = thetamean_temp(:,2);
        bmean_temp = thetamean_temp(:,3);
        out = mvnpdf([mu,lambda],[mumean_temp,lambdamean_temp]+(b-bmean_temp)*Sigma12_temp'/Sigma22_temp,Sigma11_temp-Sigma12_temp*Sigma21_temp/Sigma22_temp); %n*2
    end
    
    % acceptance rate
    function out = accept(munew,lambdanew,mu,lambda,b,G,Sigma) % scalar
        out = min(ones(n,1),likelihoodratio(munew,lambdanew,mu,lambda).*condpdf1(munew,lambdanew,b,G,Sigma)./condpdf1(mu,lambda,b,G,Sigma));
    end

%% initial values:
   mu = ones(n,1);
   lambda = ones(n,1);
   b = ones(n,1);
   
   Sigma = iwishrnd(rho0,R0); %m*m
   vec = mvnrnd(G0(:),kron(Sigma,A0));%A0: p*p
   G = vec2mat(vec,p)'; %p*m
   
   beta = normrnd(beta0,B0);
   temp_sigma = gamrnd(v0/2,2/delta0);
   sigmaepi = 1/temp_sigma;
%% store estimates
   mu_store = zeros(k,n);
   lambda_store = zeros(k,n);
   b_store = zeros(k,n);
   

%% iteration 
   for t = 1:k   
    %% use random-walk Metropolis to draw mu and lambda, given b,G,Sigma (update mu,lambda,given b,G,Sigma fixed) 
       mu_new = mu + normrnd(0,0.005,n,1);
       lambda_new = lambda + normrnd(0,0.005,n,1);
       
       % the accpetance rate of [mu,lambda,b] given b...
       a = accept(mu_new,lambda_new,mu,lambda,b,G,Sigma);%n*1
       r = rand(n,1);
       
       mu = mu_new .* double(r<a) + mu .* double(r>=a);
       lambda = lambda_new .* double(r<a) + lambda_new .*double(r>=a);

       mu_store(t,:) = mu'; %mu_store: k*n(k>n)
       lambda_store(t,:) = lambda';
           

    %% use Gibbs to draw b|z,mu,lambda,beta,sigmaepi,G,Sigma
       %posterial of b: norm(bnorm,sigmab)
       thetamean = X*G;%G: p*m -> thetamean: n*m
       mumean = thetamean(:,1);
       lambdamean = thetamean(:,2);
       bmean = thetamean(:,3);
       
       Sigma11 = Sigma(1:2,1:2);
       Sigma12 = Sigma(1:2,3);%2*1
       Sigma21 = Sigma(3,1:2);% 1*2
       Sigma22 = Sigma(3,3);
       sigmahat = Sigma22-Sigma21/Sigma11*Sigma12; %scalar
       bhati = bmean + ([mu,lambda] - [mumean,lambdamean])*Sigma21'/Sigma22;%n*1
       
       sigmabi = 1./(1/sigmahat + (x+1)./sigmaepi); %n*1
       
       z(:,4) = log(z(:,3))-beta * log(d);
       ID_min = min(z(:,1));
       ID_max = max(z(:,1));
       for j=ID_min:ID_max 
           idx_tmp =find(z(:,1)==j);  
           sum_overall(j)=sum(z(idx_tmp,4)); 
       end
       
       bi = sigmabi.*(bhati./sigmahat + 1/sigmaepi.*sum_overall'); %n*1
       b = mvnrnd(bi,diag(sigmabi))'; %updated b,1*n
       b_store(t,:) = b';

    %% draw beta,sigmaepi|z,b
      % posterial distribution of beta ~ N(betahat,Bhat)
       Bhat = 1/(1/B0+log(d)'*log(d)/sigmaepi);
       
       for j=ID_min:ID_max 
           idx =find(z(:,1) == j);  
           z(idx,5)=b(j); 
       end
       
       betahat = Bhat * (beta0/B0 + log(d)'*(log(z(:,3))-z(:,5))/sigmaepi);
       beta = normrnd(betahat,Bhat);
       temp_sig = gamrnd((v0+n)/2,2/(delta0 + sum((log(z(:,3))-z(:,5)-log(d)*beta).^2)));
       sigmaepi = 1/temp_sig;
       
   %% draw G,Sigma|theta(n*3),X(n*5); 
      Y = [mu,lambda,b]; %n*3
      Bhat = kron(Sigma, inv(X'*X + A0)); %X:n*5, A0:5*5; G0:5*3
      betahat_ols = (X'*X)^(-1)*X'*Y; % 5*3
      betahat = (X'*X + A0)^(-1)*(X'*X*betahat_ols + A0 * G0);
      G_hat = mvnrnd(betahat(:),Bhat); %mp*1 
      G = vec2mat(G_hat,p)'; %p*m
   
      S0 = (Y-X*betahat)'*(Y-X*betahat) + (betahat-G0)'*A0*(betahat-G0); %m*m
      Sigma = iwishrnd(rho0+S0,R0+n);
   end 
   out1 = G;
   out2 = beta;
   out3 = sigmaepi;
   out4 = Sigma; 
end

