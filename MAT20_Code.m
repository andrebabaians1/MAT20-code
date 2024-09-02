close all; 

% Variable initialisation
admissionsFileName = 'modified_admissions.csv'; % File containing data (dates and values).
startDate = datetime('2020-08-15'); % Start point of data.
endDate = datetime('2020-12-30'); % End point of data.
changePoints = [datetime('2020-10-27') datetime('2020-11-11') datetime('2020-11-27')]; % Points where the transmission rate changes.
alpha = 0.12; % Recovery rate for infected individuals.
gamma = 0.07; % Recovery rate for hospitalized individuals.
rho = 0.44; % Probability of dying in hospital.
p = 0.08; % Probability of being hospitalized.
N = 56000000; % Total population size.
backdate = 30; % How long before the data the model starts.
params0 = [1000; 0.15; 0.15; 0.1;0.15]; % Initial guesses (ignore if adapt = 1).
var = 0.0000001 * [1000000 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0; 0 0 0 1 0;0 0 0 0 1]; % Initial covariance matrix (ignore if adapt = 1).
runNum = 50000; % Amount each chain runs for.
adapt = 1; % Flag to show if adaptive MCMC is enabled.
chainFileName = 'chain.csv'; % File name to store the previous chain.
varStep = 1; % Covariance matrix scale (leave at 1 for default).

% Function that's passed into the ode45 function. Calculates the derivatives
% and returns them to be integrated.
function derivatives = SihrdModel(t, y, betaVector, alpha, gamma, p, rho, changePoints)
   
    % Determines beta depending on what point in time we are at.
    index = find(changePoints > t, 1);
    
    if isempty(index)
        index = length(changePoints) + 1;
    end

    beta = betaVector(index);
  
    % ODE model.
    N = sum(y);
    inc = beta * y(1) * y(2) / N;
    dSdt = - inc;
    dIdt = inc - alpha * y(2);
    dHdt = alpha * p * y(2) - gamma * y(3);
    dRdt = gamma * (1 - rho) * y(3) + alpha * (1 - p) * y(2);

    % Returns derivatives.
    derivatives = [dSdt; dIdt; dHdt; dRdt];

end

% Takes model paramters, proposed parameters, data, backdate and change
% points. It calculates the value of a fitting function proportional to the log
% likelihood using the parameter estimates and returns it.
function fit = FittingFunction(params, tspan, alpha, gamma, p, rho, N, backdate, dataValue, changePoints)
    
    % Initial conditions vector (S I H R).
    y0 = [N - params(1); params(1); 0; 0]; 

    % Solves ODE for the specific set of params.
    [t, y] = ode45(@(t, y) SihrdModel(t, y, params(2:end), alpha, gamma, p, rho, changePoints), tspan, y0);  

    % Find the expected admissions as per the ODE model.
    eta = y(backdate + 2:end,2) .* p .* alpha;

    % Calculates and returns value of fitting function.
    expression = -eta + log(eta).*dataValue;
    fit = sum(expression);
end

% Takes data, model parameters, intial guesses, change points, variance, backdate, and run number and carries out Metropolis Hastings algorithm.
% It will output a vector containing the chain of accepted values.
function paraSet = McmcProcess(data, alpha, gamma, rho, p, N, tspan, params0, changePoints, var, backdate, runNum)
   
    % Converts columns of data into arrays.
    data_value = table2array(data(:,2));
    
    % Finds value of fitting function using initial guesses.
    ll_init = FittingFunction(params0, tspan, alpha, gamma, p, rho, N, backdate, data_value, changePoints);

    % Array that will store all parameters that have been accepted by MCMC.
    paraSet = [];
    paraSet = [paraSet params0];

    % Array that will store all log likelihoods of the parameters that have been accepted.
    likelihoodSet = [];
    likelihoodSet = [likelihoodSet ll_init];
    
    % Array that will store all parameter guesses made, accepted or not. 
    allGuesses = [];
    allGuesses = [allGuesses; params0];
    
    accepts = 0;

    % Metropolis Hastings algorithm.
    for i = 2:runNum
        
        % Outputs progress of the loop.
        if rem(i,10000)==0                
            i/runNum
        end

        % Pulls from proposal distribution and calculates value of fitting
        % function using the proposed parameters.
        paraTest = mvnrnd(paraSet(:,end), var);
        ll_test = FittingFunction(paraTest, tspan, alpha, gamma, p, rho, N,backdate, data_value, changePoints);
        
        % Finds the ratio between the proposed fitting function and the
        % currently accepted one.
        ratio =  exp(ll_test - likelihoodSet(end));

        % Updates 'loglikelihoodset' and 'paraset' if 'll_test' is greater than the 
        % current proposed loglikelihood. If not, it updates them with some
        % probability equal to the ratio. 'allGuesses' gets updated no matter what. 
        if (ratio >= 1) || (random('Uniform',0,1) < ratio)
            accepts = accepts + 1;
            likelihoodSet = [likelihoodSet ll_test];
            paraSet = [paraSet paraTest'];           
            allGuesses = [allGuesses paraTest']; 
        else
            likelihoodSet = [likelihoodSet likelihoodSet(end)];
            paraSet = [paraSet paraSet(:,end)];
            allGuesses = [allGuesses paraTest'];
        end
       
    end
    
    % Outputs acceptance rate.
    acceptanceRate = accepts/runNum
         

end

% Takes the model prarmeters, proposed parameters, time span, backdate and
% change points. Plots the expected admissions using the data.
function PlotGraph(params, tspan, alpha, gamma, p, rho, N, backdate, changePoints, label)
   
    % Initial conditions vector (S I H R).
    y0 = [N - params(1); params(1); 0; 0]; 

    % Solves ODE for the specific set of params.
    [t, y] = ode45(@(t, y) SihrdModel(t, y, params(2:end), alpha, gamma, p, rho, changePoints), tspan, y0);
    
    % Find the expected admissions as per the ODE model.
    eta = y((backdate + 2):end,2) .* p .* alpha;
    
    % Plots expected admissions.
    hold on;
    plot(t((backdate + 2):end), eta, '-o', 'DisplayName', label)
end


% Uses the previous chain to set the mean and variance of the proposal
% distribution to that of the previous chain if adaptive MCMC is enabled.
if (adapt == 1)
   prevChain = table2array(readtable(chainFileName));
   params0 = mean(prevChain)';
   var = varStep * cov(prevChain);
end

% Read file as array.
admissions = readtable(admissionsFileName);

% Limit data between the start and end points.
admissions = admissions(admissions.date < endDate, :);
admissions = admissions(startDate <= admissions.date, :);

% Convert the change points to number of days after the start date.
changePoints = [days(changePoints - startDate)];

% Set the timespan we will be looking at. Backdate is applied here.
tspan = [-backdate:days(endDate - startDate)];


% Carry out MCMC and get values for our chain.
paraSet = McmcProcess(admissions, alpha, gamma, rho, p, N, tspan, params0, changePoints, var, backdate, runNum);

% Save second half of chain to csv file.
T = array2table(paraSet(:,fix(runNum/2) + 1:end)');
writetable(T,chainFileName);
writetable(T, 'mytable.csv', 'WriteRowNames', true);


% Plot posterior density of the second half of the chain for each
% parameter.
figure;
subplot(2,3,1)
histogram(paraSet(1, fix(runNum/2) + 1:end))
xlabel('Parameter value','FontSize',15);
ylabel('Density','FontSize',15);
title('I0','FontSize',15);
set(gca, 'FontSize',15)
subplot(2,3,2)
histogram(paraSet(2,fix(runNum/2) + 1:end))
xlabel('Parameter value','FontSize',15);
ylabel('Density','FontSize',15);
title('Beta1','FontSize',15);
set(gca, 'FontSize',15)
subplot(2,3,3)
histogram(paraSet(3,fix(runNum/2) + 1:end))
xlabel('Parameter value','FontSize',15);
ylabel('Density','FontSize',15);
title('Beta2','FontSize',15);
set(gca, 'FontSize',15)
subplot(2,3,4)
histogram(paraSet(4,fix(runNum/2) + 1:end))
xlabel('Parameter value','FontSize',15);
ylabel('Density','FontSize',15);
title('Beta3','FontSize',15);
set(gca, 'FontSize',15)
subplot(2,3,5)
histogram(paraSet(5,fix(runNum/2) + 1:end))
xlabel('Parameter value','FontSize',15);
ylabel('Density','FontSize',15);
title('Beta4','FontSize',15);
set(gca, 'FontSize',15)

% Plot observed admissions.
figure;
plot(table2array(admissions(:,1)), table2array(admissions(:,2)), '-o', 'DisplayName','Observed Admissions');
hold on;

% Plot median and 5% percentile of the posterior distribution against the
% observed values.
PlotGraph(prctile(paraSet',2.5), tspan, alpha, gamma, p , rho, N, backdate, changePoints, 'Lower percentile');
PlotGraph(prctile(paraSet',50), tspan, alpha, gamma, p , rho, N, backdate, changePoints,'Median' );
PlotGraph(prctile(paraSet',97.5), tspan, alpha, gamma, p , rho, N, backdate, changePoints, 'Upper percentile');
legend('show');
xlabel('Time', 'FontSize',20);
ylabel('Hospital Admissions', 'FontSize',20);
set(gca, 'FontSize',20)

% Displays the trace plots of our parameters.
figure;
subplot(2,3,1) 
plot(1:runNum, paraSet(1,:));
xlabel('Iteration','FontSize',15);
ylabel('Parameter Value','FontSize',15);
title('I0','FontSize',15);
set(gca, 'FontSize',15)
subplot(2,3,2)
plot(1:runNum, paraSet(2,:));
xlabel('Iteration','FontSize',15);
ylabel('Parameter Value','FontSize',15);
title('Beta1','FontSize',15);
set(gca, 'FontSize',15)
subplot(2,3,3)
plot(1:runNum, paraSet(3,:));
xlabel('Iteration','FontSize',15);
ylabel('Parameter Value','FontSize',15);
title('Beta2','FontSize',15);
set(gca, 'FontSize',15)
subplot(2,3,4)
plot(1:runNum, paraSet(4,:));
xlabel('Iteration','FontSize',15);
ylabel('Parameter Value','FontSize',15);
title('Beta3','FontSize',15);
set(gca, 'FontSize',15)
subplot(2,3,5)
plot(1:runNum, paraSet(5,:));
xlabel('Iteration','FontSize',15);
ylabel('Parameter Value','FontSize',15);
title('Beta4','FontSize',15);
set(gca, 'FontSize',15)
grid on;