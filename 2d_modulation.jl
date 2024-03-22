using LinearAlgebra


include("functions.jl")

l1 = 1
l2 = -1
V = [-1 -1; -1 1]
D = [l1 0; 0 l2]
#A = V * D * inv(V)
A = [0 -1; -1 0]
M = zeros(2, 2, 2)

n = 1
ylims = (-1., 1.)
xlims = (0.5, 1.5)

pars = (A = A, M = M, b = zeros(2), U0 = 0.5, s = 0., n = n, bbar = 0.)
x0 = [0.0, 0.0]

recordFromSolution(x, p) = (x1 = x[1], x2 = x[2])
prob1 = BifurcationProblem(F, x0, pars, (@lens _.U0),
                          record_from_solution = recordFromSolution)

opts_br1 = ContinuationPar(p_min = 0.5, p_max = 1.5, n_inversion = 4)
br1 = continuation(prob1, PALC(), opts_br1; normC = norminf, bothside = true)
#@show br

diagram1 = bifurcationdiagram(prob1, PALC(), 2, (args...) -> opts_br1, )
@show diagram
plt = plot(layout = (2, 1), size = (800, 800), grid = false)
plot!(plt[1], diagram1, vars = (:param, :x1), legend = false,
      linewidthstable = 5, linewidthunstable = 3, ylims = ylims,
      xlims = xlims)



M[2, 1, 1] = 3
pars2 = (A = A, M = M, b = zeros(2), U0 = 0.5, s = 0., n = n, bbar = 1.)

prob2 = BifurcationProblem(F, x0, pars2, (@lens _.U0),
                          record_from_solution = recordFromSolution)

opts_br2 = ContinuationPar(p_min = 0.5, p_max = 1.5, n_inversion = 4)
br2 = continuation(prob2, PALC(), opts_br2; normC = norminf, bothside = true)
#@show br2

diagram2 = bifurcationdiagram(prob2, PALC(), 2, (args...) -> opts_br2, )
@show diagram2
plot!(plt[2], diagram2, vars = (:param, :x1), legend = false,
      linewidthstable = 5, linewidthunstable = 3, ylims = ylims,
      xlims = xlims)
display(plt)

