using LinearAlgebra
using NLsolve
include("functions.jl")

A = zeros(4, 4)
A[1, 2] = -10
A[2, 1] = -10
A[3, 1] = 1.1
A[1, 3] = 1.1
A[4, 1] = 1.1
A[1, 4] = 1.1
A[3, 4] = -3.0
A[4, 3] = -3.0

M = zeros(4, 4, 4)

spectrum, V = eigen(A)
W = inv(V)
pos = argmax(spectrum)
lead = spectrum[pos]
A = A ./ lead
spectrum, V = eigen(A)
vmax = V[:, pos]
wmax = W[pos, :]

pars = (A = A, M = M, bbar = 0., b = wmax*0.000, U0 = 0., s = 0.5, n = 3)
x0 = [0., 0., 0., 0.]

rfs(x, p) = (d12 = x[1]-x[2], s34 = x[3]+x[4],
             x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4],
             p = dot(x, vmax))
prob = BifurcationProblem(F, x0, pars, (@lens _.U0),
                          record_from_solution = rfs)

pmin = 0.
pmax = 4.5
opts = ContinuationPar(p_min = pmin, p_max = pmax, n_inversion = 4,
                       max_steps = 500, ds = 0.001, dsmin = 0.0001,
                       dsmax = 0.01)
br = continuation(prob, PALC(), opts; normC = norminf, bothside = true)

diagram = bifurcationdiagram(prob, PALC(), 3, (args...) -> opts; nev = 8,
                             bothside = true, dsmin = 0.0001)
@show diagram
plt = plot(layout = (2, 2), size = (1200, 750*1.3), grid = false, legend = false,
           xlims = (0.75, 4.0), ylims = (-1., 3.))
plot!(plt[1], diagram, vars = (:param, :x1))
plot!(plt[2], diagram, vars = (:param, :x3))


M[4, 3, 1] = 2.
M[3, 4, 1] = 2.
pars = (A = A, M = M, bbar = 0., b = wmax*0.000, U0 = 0., s = 0.5, n = 3)

prob = BifurcationProblem(F, x0, pars, (@lens _.U0),
                          record_from_solution = rfs)

pmin = 0.
pmax = 4.5
opts = ContinuationPar(p_min = pmin, p_max = pmax, n_inversion = 4)

diagram = bifurcationdiagram(prob, PALC(), 3, (args...) -> opts; nev = 8,
                             bothside = true, dsmin = 0.0001)

plot!(plt[3], diagram, vars = (:param, :x1))
plot!(plt[4], diagram, vars = (:param, :x3))

display(plt)

