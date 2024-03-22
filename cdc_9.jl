using NLsolve
using Random

Random.seed!(64)
# 95
# Para todo rndm: 18, 197
# Root 47 da resultados extraños en ambas implementaciones de la función

include("functions.jl")

spectrum = [1, 0.3, 0.01, 0.015, 0.001, 0.02, 0.0013, 0.1, 0.03]
D = Diagonal(spectrum)
#vmt = randn(9)
#V = [vmt nullspace(vmt')]
V = randn(9, 9)
W = inv(V)
A = V * D * W
vmax = V[:, 1]
wmax = W[1, :]

#M = 2*randn(9, 9, 9)
M = 0.80*randn(9, 9, 9)
# Para todo rndm: 1.3, 1.0

U0 = 0.00
b = zeros(3)
s = 0.
bo = nullspace(wmax')[:, 1]
@show bo
@show dot(bo, wmax)

n1m0b0 = (A = A, M = zeros(9, 9, 9), bbar = 1., b = zeros(9)  , U0 = U0, s = s, n = 1)
n1mrb0 = (A = A, M = M             , bbar = 1., b = zeros(9)  , U0 = U0, s = s, n = 1)
n1mrbo = (A = A, M = M             , bbar = 1., b = 0.005*bo  , U0 = U0, s = s, n = 1)
n1mrbp = (A = A, M = M             , bbar = 1., b = -0.005*wmax, U0 = U0, s = s, n = 1)
n2m0b0 = (A = A, M = zeros(9, 9, 9), bbar = 1., b = zeros(9)  , U0 = U0, s = s, n = 2)
n2mrb0 = (A = A, M = M             , bbar = 1., b = zeros(9)  , U0 = U0, s = s, n = 2)
n2mrbo = (A = A, M = M             , bbar = 1., b = 0.005*bo  , U0 = U0, s = s, n = 2)
n2mrbp = (A = A, M = M             , bbar = 1., b = -0.005*wmax, U0 = U0, s = s, n = 2)
n3m0b0 = (A = A, M = zeros(9, 9, 9), bbar = 1., b = zeros(9)  , U0 = U0, s = s, n = 3)
n3mrb0 = (A = A, M = M             , bbar = 1., b = zeros(9)  , U0 = U0, s = s, n = 3)
n3mrbo = (A = A, M = M             , bbar = 1., b = 0.005*bo  , U0 = U0, s = s, n = 3)
n3mrbp = (A = A, M = M             , bbar = 1., b = -0.005*wmax, U0 = U0, s = s, n = 3)

opars = (n1m0b0 = n1m0b0,
         n1mrb0 = n1mrb0,
         n1mrbo = n1mrbo,
         n1mrbp = n1mrbp,
         n2m0b0 = n2m0b0,
         n2mrb0 = n2mrb0,
         n2mrbo = n2mrbo,
         n2mrbp = n2mrbp,
         n3m0b0 = n3m0b0,
         n3mrb0 = n3mrb0,
         n3mrbo = n3mrbo,
         n3mrbp = n3mrbp,
        )

ds = 0.005
dsmin = 0.001
dsmax = 0.01

plt = plot(layout = (3, 4), size = (1600, 1600), grid = false,
           xlims = (0.5, 1.5), ylims = (-1.5, 1.5))
for (ii, key) in enumerate(keys(opars))
    @show ii

    if opars[key].b != zeros(3)
        sol = nlsolve(x -> F(x, opars[key]), zeros(9))
        x0 = sol.zero
        #@show x0
        #@show dot(x0, wmax)
        @show isapprox(F(x0, opars[key]), zeros(9))
    else
        x0 = zeros(9)
    end
    rfs(x, p) = (x1 = x[1], x2 = x[2], x3 = x[3], p = dot(x, wmax))
    prob = BifurcationProblem(F, x0, opars[key], (@lens _.U0),
                              record_from_solution = rfs)

    U0_max = 1.55
    opts = ContinuationPar(p_min = 0.00, p_max = U0_max, n_inversion = 28,
                           ds = ds, dsmin = dsmin, dsmax = dsmax,
                           max_steps = 5000)
    br = continuation(prob, PALC(), opts; normC = norminf, bothside = true,
                      ds = ds, dsmin = dsmin, dsmax = dsmax)
    diagram = bifurcationdiagram(prob, PALC(), 2, (args...) -> opts,
                                 nev = 20, bothside = true,
                                 ds = ds, dsmin = dsmin, dsmax = dsmax)
    #col = [stb ? :black : :blue for stb in br.stable]
    if ii in [1 5 9]
        #plot_diagram(plt, ii, diagram, (:param, :p), (0.75, 1.5), (-0.75, 0.75),
                     #:black, :red, 5, 3)
        plot!(plt[ii], diagram, vars = (:param, :p), linewidthunstable = 3,
              linewidthstable = 5,
              legend = false)
        #plot!(plt[ii], br.branch.param, br.branch.p, ylabel = "", grid = false,
              #color = col,
              #ylims = (-0.75, 0.75), xlims = (0.75, 1.5))
    else
        plot!(plt[ii], diagram, vars = (:param, :p), legend = false,
              linewidthunstable = 3, linewidthstable = 5)
        #plot_diagram(plt, ii, diagram, (:param, :p), (0.75, 1.5), (-0.75, 0.75),
                     #(:black, :blue), (5, 3)) 
    end
    
    if opars[key].b != zeros(3)
        pars = opars[key]
        cpars = (A = pars[:A], M = pars[:M], s = pars[:s], n = pars[:n],
                 b = pars[:b], U0 = U0_max, bbar = 1.)
        sol = nlsolve(x -> F(x, cpars), zeros(9); method = :anderson)
        if !sol.f_converged
            throw(ErrorException("Root could not be found."))
        else
            x0 = sol.zero
        end
        @show isapprox(F(x0, cpars), zeros(9))
        prob = BifurcationProblem(F, x0, cpars, (@lens _.U0),
                                  record_from_solution = rfs)
        opts = ContinuationPar(p_min = 0.00, p_max = U0_max, n_inversion = 28,
                               ds = ds, dsmin = dsmin, dsmax = dsmax,
                               max_steps = 5000)
        br = continuation(prob, PALC(), opts; normC = norminf, bothside = true,
                          ds = ds, dsmin = dsmin, dsmax = dsmax)
        diagram = bifurcationdiagram(prob, PALC(), 2, (args...) -> opts,
                                     nev = 20, bothside = true,
                                     ds = ds,
                                     dsmin = dsmin,
                                     dsmax = dsmax)
        #plot_diagram(plt, ii, diagram, (:param, :p), (0.75, 1.5), (-0.75, 0.75),
                     #(:black, :blue), (5, 3)) 
        plot!(plt[ii], diagram, vars = (:param, :p), linewidthstable = 5,
              linewidthunstagle = 3,
              legend = false)
    end


    display(plt)
end

