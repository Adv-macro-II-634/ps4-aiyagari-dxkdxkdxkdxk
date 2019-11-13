beta = .99; 
alpha=1/3; 
sigma = 2; 
delta=.025; 
rho_epsi=0.5; 
sigma_epsi=0.2; 

%exogenous variable grid
num_z=5;
[z_grid, PI]=TAUCHEN(num_z,rho_epsi,sigma_epsi,3);
z_grid=exp(z_grid');
% invariant distribution is 
PI_inv=PI^1000;
PI_inv=PI_inv(1,:)';
N_s=z_grid*PI_inv; 
a_lo = 0; 
a_hi = 100;
num_a = 50;
n_a=500;

a = linspace(a_lo, a_hi, num_a); 
a_inter = linspace(a_lo, a_hi, n_a);

K_min=20;
K_max=50;
K_tol=1;
while abs(K_tol)>.01
    if K_max-K_min<0.00001
       break
   end
    
    K_guess=(K_min+K_max)/2;
    interest= alpha*K_guess^(alpha-1)*N_s^(1-alpha)+(1-delta);
    wage=(1-alpha)*K_guess^alpha*N_s^(-alpha);
    
   	    cons = bsxfun(@minus, interest* a', a);
    cons = bsxfun(@plus, cons, permute(z_grid, [1 3 2])*wage);
    ret = (cons .^ (1-sigma)) ./ (1 - sigma); % current period utility
    ret(cons<0)=-Inf;
    
    v_guess = zeros(num_z, num_a);
    
    
    v_tol = 1;
    while v_tol >.000001;
        
	        value_mat=ret+beta*repmat(permute((PI*v_guess),[3 2 1]), [num_a 1 1]);
       [vfn, pol_indx] = max(value_mat, [], 2); %max for each row
       vfn=permute(vfn, [3 1 2]);
       v_tol = max(abs(vfn-v_guess));
       v_tol = max(v_tol(:));
       
       v_guess = vfn;
    end;
    
    pol_indx=permute(pol_indx, [3 1 2]);
    pol_fn = a(pol_indx);
    
    
    V=zeros(num_z,n_a);
    for i=1:num_z
        V(i,:)=interp1(a,v_guess(i,:),a_inter);
    end
   
    cons = bsxfun(@minus, interest* a_inter', a_inter);
    cons = bsxfun(@plus, cons, permute(z_grid, [1 3 2])*wage);
    ret = (cons .^ (1-sigma)) ./ (1 - sigma); % current period utility
    ret(cons<0)=-Inf;
    value_mat=ret+beta*repmat(permute((PI*V),[3 2 1]), [n_a 1 1]);
    [vfn, pol_indx] = max(value_mat, [], 2);
    vfn=permute(vfn, [3 1 2]);
    pol_indx=permute(pol_indx, [3 1 2]);
    pol_fn = a_inter(pol_indx);
    
    MU=zeros(num_z,n_a);
    MU(:)=1/(num_z*n_a);
    
    dis=1;
  while dis>0.0000001 
      % ITERATE OVER DISTRIBUTIONS
      MuNew = zeros(size(MU));
     [z_ind, a_ind, mass] = find(MU); % find non-zero indices
    
    for ii = 1:length(z_ind)
        apr_ind = pol_indx(z_ind(ii),a_ind(ii)); % which a prime does the policy fn prescribe?
        
        MuNew(:, apr_ind) = MuNew(:, apr_ind) + ... % which mass of households goes to which exogenous state?
            (PI(z_ind(ii), :)*mass(ii))';
        
    end
    dis = max(max(abs(MU-MuNew)));
    MU=MuNew;
  end
   
   K=sum(sum(MU.*pol_fn));
   K_tol=K-K_guess;
   if K_tol>0;
       K_min=K_guess;
   else K_max=K_guess;
   end
end

% policy function
figure(1)
plot(a_inter,pol_fn)
z_name=cellstr(num2str(z_grid'));
legend(z_name,'location','southeast')
title(['Policy Function for Productivity z'])

% gini coefficient and lorenz curve
pop=reshape(MU',[num_z*n_a,1]);
wealth=reshape(repmat(a_inter,num_z,1)',[num_z*n_a,1]);

mu=sum(MU);
figure(2)
bar(a_inter,mu)
title('Distribution of Assets')
%%%%%%% Gini coefficient and lorenz curve%%%%
WEALTH=sortrows([wealth,pop,pop.*wealth]);
WEALTH=cumsum(WEALTH);
pw=WEALTH(:,2);
pw=pw(end);
WEALTH(:,2)=WEALTH(:,2)/pw;
w=WEALTH(:,3);
w=w(end);
WEALTH(:,3)=WEALTH(:,3)/w;
gini_wealth2 = 1 - sum((WEALTH(1:end-1,3)+WEALTH(2:end,3)) .* diff(WEALTH(:,2)));

figure(3)
suptitle('Lorenz Curve' )
area(WEALTH(:,2),WEALTH(:,3),'FaceColor',[0.5,0.5,1.0])
hold on
plot([0,1],[0,1],'--k')
axis square
title(['Wealth, Gini=',num2str(gini_wealth2)])
hold off

y=z_grid*wage
Y=repmat(y',[1,n_a]);
A=repmat(a,[num_z,1])
c=Y+interest*A-pol_fn;
cf=c(:,pol_indx');
cf1=reshape(cf,[num_z n_a num_z]);
i=1;
while i < num_z+1
c1(i,:)=PI(i,:)*cf1(:,:,i);
i=i+1;
end
Eulererror=sum(sum(abs(c.^(-sigma)-beta*c1.^(-sigma)*interest).*MU))