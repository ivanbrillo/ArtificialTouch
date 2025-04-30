def simulate_neuron(a,b,c,d,vr,ur,dt,t,input_curr, p = 1 , off = 0, offu = False):
    v = []
    u = []
    # Loop to solve the numerical problem
    for i in range(len(t)) :
        vr = vr + dt * (0.04 * (vr ** 2) + p * 5 * vr + 140 + off - ur + input_curr[i])
        if offu is True :
            ur = ur + dt * a * (b * vr + 65)
        else :
            ur = ur + dt * a * (b * vr - ur)

        if vr > 30 :
            v.append (vr)
            vr = c
            ur = ur + d
        else:
            v.append (vr)
        u.append (ur)
    return v,u




