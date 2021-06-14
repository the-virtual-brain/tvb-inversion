

using Distributions
using LinearAlgebra: mul!

function MPR(h, h_Store, sim_len, nCoeff, sc_mat, ti, tf, J, delta, coup_st, eta, skip_, count_)

  N = size(sc_mat,1)
  pos = 1
  counter = 1

  pi_ratio = delta/π
  pisquared = π^2
  stepfactor = h/2.0

  R = zeros(Int64((sim_len)/h_Store), N)
  V = zeros(Int64((sim_len)/h_Store), N)
  r = rand(N)*3.0 .- 0.0
  v = rand(N)*3.0 .- 2.5

  coup = zeros(N)

  rTemp = similar(coup)
  vTemp = similar(coup)
  rLeft = similar(coup)
  vLeft = similar(coup)
  rRight = similar(coup)
  vRight = similar(coup)

  rNoise = sqrt(h) * sqrt(2.0 * nCoeff * 0.005)
  vNoise = sqrt(h) * sqrt(2.0 * nCoeff * 0.01)

  noise = Matrix{Float64}(undef, N, 2)
  n1 = view(noise, :, 1)
  n2 = view(noise, :, 2)

  for t in ti+h:h:tf

    mul!(coup, sc_mat, r)

    noise .= rand(Normal(0,1),N,2)

    @. rLeft = pi_ratio + 2.0*r*v
    @. vLeft = v*v - pisquared*r*r + eta + r*J + coup_st*coup

    mul!(rTemp, rNoise, n1)
    mul!(vTemp, vNoise, n2)

    rTemp .+= r .+ h.*rLeft
    vTemp .+= v .+ h.*vLeft

    @. rRight = pi_ratio + 2.0*rTemp*vTemp
    @. vRight = vTemp*vTemp - pisquared*rTemp*rTemp + eta + rTemp*J + coup_st*coup

    r .+= stepfactor .* (rLeft .+ rRight) .+ rNoise .* n1
    v .+= stepfactor .* (vLeft .+ vRight) .+ vNoise .* n2

    for (i,x) in enumerate(r)
      if x < 0
        r[i] = 0.0
      end
    end

    if t > skip_
      if mod(counter, count_) == 0
        V[pos, :] .= v
        R[pos, :] .= r
        pos = pos + 1
      end
      counter = counter + 1
    end

    if isnan(v[1])
      break
    end

  end
  return V,R
end
