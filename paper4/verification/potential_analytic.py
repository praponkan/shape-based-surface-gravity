#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analytic potential of uniform ellipsoid (verify library). SI, metres."""
import numpy as np
from scipy import integrate
G=6.674e-11
def ellipsoid_potential(x,y,z,a,b,c,rho):
    # surface/exterior point; substitution u=t^2 removes sqrt singularity
    def integrand_t(t):
        u=t*t
        Delta=np.sqrt((a*a+u)*(b*b+u)*(c*c+u))
        return (1 - x*x/(a*a+u) - y*y/(b*b+u) - z*z/(c*c+u))/Delta*2*t
    val,_=integrate.quad(integrand_t,0,np.inf,limit=400)
    return -np.pi*G*rho*a*b*c*val
