%% Genetic Algorithm in Global Optimizaition Toolbox
% define the function
% use @ to declare the arguments
func = @(x) 1/x + x;

% define variable for the optimization
x = optimvar("x", "LowerBound", 0, "UpperBound", 200);
% create the problem with the function
prob = optimproblem ("Objective", func(x));

% create options. here the option is to draw a diagram
options = optimoptions("ga", "PlotFcn","gaplotdistance");

% use the solve function to solve the problem
rng default
[sol, fval] = solve(prob, "Solver", "ga", "Options" , options)