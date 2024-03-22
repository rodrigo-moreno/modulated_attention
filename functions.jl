using Parameters, Setfield, Plots
using BifurcationKit
const BK = BifurcationKit
using LinearAlgebra



################################################################################
#################################################################################
#################################################################################
#### FUNCTION DEFINITION
function S(x, s)
    """
    Implementation of sigmoid function that satisfies:
    - S(0, s) = 0
    - S'(0, s) = 1
    """
    numer = tanh.(x - s) .+ tanh(s)
    denom = 1 - tanh(s)^2
    return numer / denom
end;


function S(x, s)
    numer = tanh.(x .- s) .+ tanh(s)
    denom = 1 - tanh(s)^2
    return numer / denom
end;


function F(u::AbstractVector{T}, pars) where T
    """
    Dynamical system. Due to a lack of order-3 tensor products un Julia, this
    was implemented using slices for each specific case. The general equation
    is:
        dot x = -x + S(((U0 + M x^3) odot A)x, s) + b
    where U0 and A are nxn matrices, M an nxnxn tensor, and x and b vectors
    in R^n.

    Notation on indices:
    - A: Element A[i, j] represents the effect of node j on node i.
    - M: Element M[i, j, k] represents the effect of node k on the regulation
      of i by j. In other words, M[i, j, k] modifies A[i, j].
    - U0: If number, the same U0 is used for all interactions. If Matrix,
      each interaction has a specific U0. The elements U0[i, :] are the ones
      for the incomming interactions to x[i].
    """
    @unpack U0, b, bbar, A, M, s, n = pars
    u = real.(u)

    du = zeros(T, size(u))
    for (ii, vv) in enumerate(du)
        Mv = @view M[ii, :, :]
        Av = @view A[ii, :]
        du[ii] = -u[ii] + S(((U0 .+ Mv*(u.^n)) .* Av)' * u, s) + bbar*b[ii]
        #du[ii] = -u[ii] + S(U0 * Av' * u, s) + b[ii]
    end
    return du
end;


#function F(u::AbstractVector{T}, pars) where T
    #@unpack U0, b, bbar, A, M, s, n = pars
    #u = real.(u)
#
    #Mtilde = zeros(T, length(u), length(u))
    #for ii in 1:length(u)
        #for jj in 1:length(u)
            #Mij = @view M[ii, jj, :]
            #Mtilde[ii, jj] = Mij' * u.^n
        #end
    #end
#
    ##Atilde = U0.*A + Mtilde
    #Atilde = A .* (U0 .+ Mtilde)
    #du = -u + S(Atilde*u, s) + bbar*b
#end;


function plot_diagram(plt, index, diagram, variables, xlims, ylims,
                      cols, widths)
    main = diagram.γ
    #@show variables
    #@show typeof(main)
    #@show main.param
    col = [stb ? cols[1] : cols[2] for stb in main.stable]
    width = [stb ? widths[1] : widths[2] for stb in main.stable]
    plot!(plt[index], main.param, main.p,
          grid = false, color = col, lw = width, legend = false)
    for sp in main.specialpoint
        scatter!(plt[index], [sp.param], [sp.printsol.p])
    end
    for branch in diagram.child
        col = [stb ? cols[1] : cols[2] for stb in branch.γ.stable]
        width = [stb ? widths[1] : widths[2] for stb in branch.γ.stable]
        plot!(plt[index], branch.γ.param, branch.γ.p, color = col, lw = width,
              legend = false, xlims = xlims, ylims = ylims)
        for sp in branch.γ.specialpoint
            scatter!(plt[index], [sp.param], [sp.printsol.p])
        end
    end
end;

