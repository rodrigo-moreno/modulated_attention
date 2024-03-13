using LinearAlgebra
using NLsolve
include("functions.jl")

A = zeros(4, 4)
A[1, 2] = -10
A[2, 1] = -10
A[3, 1] = 1.1
A[1, 3] = 1.1
A[4, 2] = 1.1
A[2, 4] = 1.1
A[3, 4] = 3.0
A[4, 3] = 3.0

M = zeros(4, 4, 4)
M[4, 3, 1] = 3
M[3, 4, 1] = 3
#M[3, 1, 3] = -50
#M[1, 2, 1] = 10
#M[2, 1, 1] = 10
#M = M ./ 5

spectrum, V = eigen(A)
W = inv(V)
pos = argmax(spectrum)
lead = spectrum[pos]
A = A ./ lead
spectrum, V = eigen(A)
vmax = V[:, pos]
wmax = W[pos, :]

pars = (A = A, M = M, bbar = 0., b = wmax*0.001, U0 = 0., s = 0.0, n = 3)
x0 = [0., 0., 0., 0.]

rfs(x, p) = (d12 = x[1]-x[2], s34 = x[3]+x[4],
             x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4],
             p = dot(x, vmax))
prob = BifurcationProblem(F, x0, pars, (@lens _.U0),
                          record_from_solution = rfs)

pmin = 0.
pmax = 2.5
opts = ContinuationPar(p_min = pmin, p_max = pmax, n_inversion = 4)
br = continuation(prob, PALC(), opts; normC = norminf, bothside = true)
#@show br

diagram = bifurcationdiagram(prob, PALC(), 3, (args...) -> opts, nev = 4,
                             bothside = true)
@show diagram
#plt = plot(layout = (3, 1), size = (800, 800))
#plot!(plt[1], diagram, vars = (:param, :p))
#plot!(plt[2], diagram, vars = (:param, :d12))
#plot!(plt[3], diagram, vars = (:param, :s34))#, ylims = (-0.01, 0.01))

plt = plot(layout = (2, 2), size = (800, 500), grid = false, legend = false,
           xlims = (0.75, 2.0))
plot!(plt[1], diagram, vars = (:param, :x1), ylims = (-1., 1.),
      linewidthstable = 5, linewidthunstable = 3)
plot!(plt[2], diagram, vars = (:param, :x2), ylims = (-1., 1.),
      linewidthstable = 5, linewidthunstable = 3)
plot!(plt[3], diagram, vars = (:param, :x3), ylims = (-1., 1.),
      linewidthstable = 5, linewidthunstable = 3)
plot!(plt[4], diagram, vars = (:param, :x4), ylims = (-1., 1.),
      linewidthstable = 5, linewidthunstable = 3)


pars = (A = A, M = M, bbar = 0., b = wmax*0.001, U0 = 2.5, s = 0.0, n = 3)
x0 = [1., -1., 0.1, -0.1]

rfs(x, p) = (d12 = x[1]-x[2], s34 = x[3]+x[4],
             x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4],
             p = dot(x, vmax))
prob = BifurcationProblem(F, x0, pars, (@lens _.U0),
                          record_from_solution = rfs)

pmin = 0.
pmax = 2.5
opts = ContinuationPar(p_min = pmin, p_max = pmax, n_inversion = 4)
br = continuation(prob, PALC(), opts; normC = norminf, bothside = true)

diagram = bifurcationdiagram(prob, PALC(), 2, (args...) -> opts, nev = 4,
                             bothside = true)

#plot!(plt[1], diagram, vars = (:param, :p))
#plot!(plt[2], diagram, vars = (:param, :d12))
#plot!(plt[3], diagram, vars = (:param, :s34))#, ylims = (-0.01, 0.01))

#plot!(plt[1], diagram, vars = (:param, :x1), ylims = (-1., 1.),
      #linewidthstable = 5, linewidthunstable = 3)
#plot!(plt[2], diagram, vars = (:param, :x2), ylims = (-1., 1.),
      #linewidthstable = 5, linewidthunstable = 3)
#plot!(plt[3], diagram, vars = (:param, :x3), ylims = (-1., 1.),
      #linewidthstable = 5, linewidthunstable = 3)
#plot!(plt[4], diagram, vars = (:param, :x4), ylims = (-1., 1.),
      #linewidthstable = 5, linewidthunstable = 3)

display(plt)

