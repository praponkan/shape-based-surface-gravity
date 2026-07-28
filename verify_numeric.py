"""
verify_numeric.py - INDEPENDENT numerical check of the O(t^4) coefficients
==========================================================================
This confirms r2, r3, r22, v2, v3, v22 WITHOUT using sympy's symbolic engine
at all. It evaluates the surface area and mean surface gravity of the ellipsoid
by high-precision numerical quadrature (mpmath), fits the Taylor coefficients
along one shape direction, and compares to the closed-form rationals derived by
wstudy_theory_o4.py.

Because the method (numerical integration + polynomial fit) is completely
different from the symbolic derivation, agreement rules out an engine-specific
bug -- this is exactly how the earlier r22 = -31/630 error was caught (the true
value 2/315 differs already in the first digit).

Only needs mpmath.  Pure ASCII.  Increase DPS / add fit points for tighter
agreement (down to full rational reconstruction).

Expected closed-form values (from wstudy_theory_o4.py):
   r2=4/15   r3=-8/63   r22=2/315
   v2=-76/375   v3=1544/39375   v22=2014/65625
"""
import mpmath as mp

DPS = 30                        # raise to 40-50 for more digits (slower)
mp.mp.dps = DPS
FIT_TS = [mp.mpf(k)/100 for k in (-10, -8, -6, -4, -2, 2, 4, 6, 8, 10)]
DEG = 8                         # polynomial fit degree (>= 5)
DIRECTION = (mp.mpf(3), mp.mpf(-1))    # unit shape direction (s1,s2); s3=-(s1+s2)

EXPECT = {"r2": "4/15", "r3": "-8/63", "r22": "2/315",
          "v2": "-76/375", "v3": "1544/39375", "v22": "2014/65625"}


def r_and_DV(s1, s2):
    """Return (r, D_V) for the ellipsoid with log-axes (s1,s2,-(s1+s2))."""
    s3 = -(s1+s2)
    a = [mp.e**s1, mp.e**s2, mp.e**s3]
    a2 = [z*z for z in a]

    def dS(T, P):
        sT, cT, cP, sP = mp.sin(T), mp.cos(T), mp.cos(P), mp.sin(P)
        Xt = [a[0]*cT*cP, a[1]*cT*sP, -a[2]*sT]
        Xp = [-a[0]*sT*sP, a[1]*sT*cP, 0]
        E = sum(z*z for z in Xt); G = sum(z*z for z in Xp)
        F = sum(Xt[i]*Xp[i] for i in range(3))
        return mp.sqrt(E*G - F*F)
    A = mp.quad(lambda T: mp.quad(lambda P: dS(T, P), [0, mp.pi, 2*mp.pi]),
                [0, mp.pi/2, mp.pi])
    r = A/(4*mp.pi)

    al = [mp.quad(lambda uu: 1/((a2[i]+uu)*mp.sqrt((a2[0]+uu)*(a2[1]+uu)*(a2[2]+uu))),
                  [0, 1, 10, 100, mp.inf]) for i in range(3)]

    def gI(T, P):
        sT, cT, cP, sP = mp.sin(T), mp.cos(T), mp.cos(P), mp.sin(P)
        x = [a[0]*sT*cP, a[1]*sT*sP, a[2]*cT]
        gg = mp.sqrt(sum(al[i]**2*x[i]**2 for i in range(3)))
        Xt = [a[0]*cT*cP, a[1]*cT*sP, -a[2]*sT]
        Xp = [-a[0]*sT*sP, a[1]*sT*cP, 0]
        E = sum(z*z for z in Xt); G = sum(z*z for z in Xp)
        F = sum(Xt[i]*Xp[i] for i in range(3))
        return gg*mp.sqrt(E*G - F*F)
    J = mp.quad(lambda T: mp.quad(lambda P: gI(T, P), [0, mp.pi, 2*mp.pi]),
                [0, mp.pi/2, mp.pi])
    DV = mp.mpf(3)/2*J/A
    return r, DV


def main():
    s1u, s2u = DIRECTION
    s3u = -(s1u+s2u)
    p2u = s1u**2 + s2u**2 + s3u**2
    p3u = s1u**3 + s2u**3 + s3u**3

    print("evaluating %d quadrature points at %d digits..." % (len(FIT_TS), DPS))
    R = []; V = []
    for tt in FIT_TS:
        r, v = r_and_DV(tt*s1u, tt*s2u)
        R.append(r-1); V.append(v-1)

    # least-squares polynomial fit (no constant term: r(0)-1 = 0)
    M = mp.matrix([[tt**k for k in range(1, DEG+1)] for tt in FIT_TS])
    MT = M.T
    def solve(ys):
        return mp.lu_solve(MT*M, MT*mp.matrix(ys))
    cr = solve(R); cv = solve(V)
    # coeff indices: cr[0]=t^1, cr[1]=t^2, cr[2]=t^3, cr[3]=t^4
    got = {"r2": cr[1]/p2u, "r3": cr[2]/p3u, "r22": cr[3]/p2u**2,
           "v2": cv[1]/p2u, "v3": cv[2]/p3u, "v22": cv[3]/p2u**2}

    print("\nINDEPENDENT numeric reconstruction (mpmath quadrature; NO sympy):")
    print("%-5s %-22s %-16s %-14s %s" % ("name", "numeric", "closed form", "value", "|diff|"))
    allok = True
    for name in ["r2", "r3", "r22", "v2", "v3", "v22"]:
        x = got[name]; n, d = EXPECT[name].split('/'); ev = mp.mpf(n)/mp.mpf(d)
        diff = abs(x-ev)
        ok = diff < mp.mpf(10)**(-5)
        allok = allok and ok
        print("%-5s %-22s %-16s %-14s %s  %s"
              % (name, mp.nstr(x, 14), EXPECT[name], mp.nstr(ev, 12),
                 mp.nstr(diff, 2), "OK" if ok else "<-- CHECK"))
    print("\n%s" % ("ALL COEFFICIENTS CONFIRMED (agree to < 1e-5)."
                    if allok else
                    "Some disagree beyond 1e-5: raise DPS/points, or investigate."))
    print("(Residuals ~1e-6 are polynomial-fit truncation, not error; raise DPS")
    print(" and shrink the fit points for tighter agreement / exact reconstruction.)")


if __name__ == "__main__":
    main()

