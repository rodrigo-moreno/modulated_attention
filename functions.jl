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
end


function S(x, s)
    numer = tanh.(x .- s) .+ tanh(s)
    denom = 1 - tanh(s)^2
    return numer / denom
end


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
    @unpack U0, b, A, M, s = pars
    if U0 isa Number
        dims = length(u)
        U0 = U0 * ones(dims, dims)
    end

    du = zeros(T, size(u))
    for (ii, vv) in enumerate(du)
        Mv = @view M[:, :, ii]
        Av = @view A[ii, :]
        Uv = @view U0[ii, :]
        du[ii] = -u[ii] + S(((Uv .+ Mv*(u.^3)) .* Av)' * u, s) + b[ii]
    end
    return du
end


function Jc(u::AbstractVector{T}, pars) where T
    """
    General case of the Jacobian calculation. Can probably be optimized A LOT,
    but will stay like this for now.

    Maths are nasty to read. Check manual notes for legibility, and function
    documentation for clarity on indices.
    """
    @unpack U0, b, A, M, s = pars
    if U0 isa Number
        dims = length(u)
        U0 = U0 * ones(dims, dims)
    end

    N = length(u)
    J = zeros(T, length(u), length(u))
    for ii in 1:length(u)
        Mv = @view M[:, :, ii]
        Av = @view A[ii, :]
        Uv = @view U0[ii, :]
        for jj in 1:length(u)
            foo = Sp(((Uv .+ Mv*u) .* Av)' * u, s)
            sum1 = [jj != kk ? M[ii, jj, kk] : 0 for kk in 1:N]
            sum1 = A[ii, jj]*u[jj]^3 * sum(sum1)
            #@show sum1
            sum2 = [jj != kk ? A[ii, kk]*M[ii, kk, jj]*u[kk] : 0 for kk in 1:N]
            sum2 = 3*u[jj]^2 * sum(sum2)
            #@show sum2
            der = A[ii, jj]*(U0[ii, jj] + 4*M[ii, jj, jj]*u[jj]^3) + sum1 + sum2
            #@show der
            if ii == jj
                J[ii, jj] = foo*der - 1
            else
                J[ii, jj] = foo*der
            end
        end
    end
    return J
end


function F!(du, u, pars)
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
    @unpack U0, b, A, M, s = pars
    if U0 isa Number
        dims = length(u)
        U0 = U0 * ones(dims, dims)
    end

    for (ii, vv) in enumerate(du)
        Mv = @view M[:, :, ii]
        Av = @view A[ii, :]
        Uv = @view U0[ii, :]
        du[ii] = -u[ii] + S(((Uv .+ Mv*u) .* Av)' * u, s) + b[ii]
    end
end


function project(a, b)
    """
    Projects vector a into vector b
    """
    p = (dot(a, b) / norm(b)) * (b / norm(b))
    return p
end

