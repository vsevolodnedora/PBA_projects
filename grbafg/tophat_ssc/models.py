"""
    Contains Synchrotron and SSC models
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, special as sc
import math
import sys
import naima
from naima.models import Synchrotron, InverseCompton, ExponentialCutoffBrokenPowerLaw
import astropy.units as u

import PyBlastAfterglowMag as PBA

erg2TeV = 0.624151

# source properties (shock properties at a given time)
class Source:
    ''' Highly relativsitc blastwave '''
    p = 2.2
    gm = 30571.5
    gM = 18712985.24
    gc = 10245374.32
    u_e_prime = 601.
    u_b_prile = 60.1
    eint = 4.24e+35
    B = 38.
    Gamma = 1000.
    Gamma_sh = 1500.
    t_e = 5.e-5
    nprime = 4002
    n_e = 2.8e35
    em_pl = 3.46e-20

# constaints in cgs
class cgs:
    mec2 = 8.187e-7  # ergs
    mec2eV = 5.11e5  # eV
    mpc2 = 1.5032e-3  # ergs
    eV2Hz = 2.418e14
    eV2erg = 1.602e12
    kB = 1.3807e-16  # erg/K
    h = 6.6261e-27  # erg*sec
    me = 9.1094e-28  # g
    mp = 1.6726e-24  # g
    G = 6.6726e-8  # dyne cm^2/g^2
    Msun = 1.989e33  # g
    Lsun = 3.826e33  # erg/s
    Rsun = 6.960e10  # cm
    pc = 3.085678e18  # cm
    e = 4.8032e-10  # statcoulonb
    re = 2.8179e-13  # cm
    sigmaT = 6.6525e-25  # cm^2
    sigmaSB = 5.6705e-5  # erg/(cm^2 s K^4 )
    Bcr = 4.414e13  # G
    c = 2.99792e10  # cm/s
    ckm = 299792  # km/s
    H0 = 67.4  # km s^-1 Mpc^-1 (Planck Collaboration 2018)
    omegaM = 0.315  # (Planck Collaboration 2018)

# WSPN99 formualation from PyBlastAfterglow
class SYN_WSPN99:
    def __init__(self, source: Source):
        self.source = source

    @staticmethod
    def _interpolated_xi(p):
        p_xi = np.array([
            1.44449807, 1.48206417, 1.48206417, 1.49279734, 1.50353051,
            1.51963027, 1.54109661, 1.55361864, 1.58402929, 1.61944876,
            1.60012905, 1.64842832, 1.66989466, 1.70746076, 1.71282735,
            1.73429369, 1.76327325, 1.79869271, 1.82552564, 1.84699198,
            1.88992467, 1.92212418, 1.95432369, 1.97579004, 2.01872272,
            2.05628882, 2.0992215, 2.14215419, 2.19045346, 2.23875273,
            2.29778517, 2.36003756, 2.42121664, 2.48561566, 2.55001469,
            2.60904713, 2.67881274, 2.74213845, 2.80761079, 2.8709365,
            2.93748216, 3.00402782, 3.06198695, 3.13711915, 3.19937154,
            3.2659172, 3.3254863, 3.39471525, 3.45428435, 3.52780657,
            3.60186545, 3.66626448, 3.73066351, 3.78969595, 3.85838824,
            3.93352044, 4.00435937, 4.06875839, 4.13852401, 4.20184971,
            4.27268864, 4.33708767, 4.40685328, 4.47661889, 4.54101792,
            4.61078353, 4.67625588, 4.74494817, 4.80398061, 4.87911281,
            4.93277866, 5.00254428, 5.09377623, 5.19037477, 5.28697331,
            5.38357185, 5.48017039, 5.57676893, 5.67336747, 5.76996601,
            5.86656455, 5.95242991, 5.9577965
        ])
        Xi = np.array([
            0.99441363, 0.99511912, 0.97241493, 0.95724356, 0.94207218,
            0.92309915, 0.90708165, 0.89361622, 0.8763981, 0.84773936,
            0.86391198, 0.8286159, 0.81232812, 0.7972488, 0.78083371,
            0.7650865, 0.74646758, 0.73108898, 0.71613501, 0.70336097,
            0.68853431, 0.67473396, 0.66093361, 0.6474388, 0.63513483,
            0.622398, 0.60928316, 0.59724948, 0.5847657, 0.5697232,
            0.55760059, 0.54334114, 0.5303826, 0.51719725, 0.50509306,
            0.49560125, 0.48484889, 0.47591959, 0.46802836, 0.46036041,
            0.4506277, 0.44403032, 0.43897477, 0.43224107, 0.4270633,
            0.4210065, 0.41605377, 0.40991608, 0.40481919, 0.39895571,
            0.39424369, 0.39060851, 0.38715353, 0.38344588, 0.37884159,
            0.3742702, 0.37030754, 0.36766342, 0.36473138, 0.36236108,
            0.35942552, 0.35705168, 0.35493051, 0.35118761, 0.34881378,
            0.34588174, 0.34328815, 0.34030559, 0.33821967, 0.33617096,
            0.33293143, 0.33032374, 0.328078, 0.32577859, 0.3231188,
            0.32027882, 0.31815961, 0.31568001, 0.31320041, 0.31072081,
            0.30824122, 0.68419991, 0.30534679
        ])
        return np.interp(p, xp=p_xi, fp=Xi)

    @staticmethod
    def _interpolated_phi(p):
        p_phi = np.array([
            1.00621662, 1.04516293, 1.13098906, 1.21681519, 1.25720396,
            1.35312728, 1.46419639, 1.58608392, 1.62070287, 1.69643181,
            1.81759811, 1.85798688, 1.9589588, 2.09022229, 2.23158298,
            2.27702034, 2.38520454, 2.42342963, 2.53449874, 2.5849847,
            2.68595662, 2.73644258, 2.83034646, 2.88285186, 2.99896957,
            3.04440693, 3.16052464, 3.205962, 3.31198252, 3.36751707,
            3.47353759, 3.51897495, 3.63509266, 3.68053002, 3.79664773,
            3.8420851, 3.94810561, 3.99859157, 4.10966068, 4.15509805,
            4.27121575, 4.31665312, 4.42267363, 4.48022763, 4.58639239,
            4.63976326, 4.73871574, 4.79122114, 4.89219306, 4.94873733,
            5.02682228, 5.0890883, 5.14462286, 5.25064337, 5.28598354,
            5.39200406, 5.44249002, 5.51821896, 5.57375351, 5.60909369,
            5.7151142, 5.7726682, 5.87666927, 5.93725242
        ])
        phi = np.array([
            0.41141367, 0.41748522, 0.43521089, 0.45395245, 0.46103901,
            0.48129824, 0.50341068, 0.52512949, 0.53192933, 0.5467828,
            0.56618001, 0.5703882, 0.58454897, 0.60117452, 0.61418181,
            0.61787897, 0.62681073, 0.62896427, 0.63753144, 0.64207209,
            0.6459046, 0.6483288, 0.65228415, 0.65489336, 0.66095157,
            0.6610931, 0.66491634, 0.66708966, 0.6690566, 0.67080045,
            0.67386795, 0.67502537, 0.67698614, 0.67797425, 0.68010433,
            0.68092312, 0.68289006, 0.684637, 0.68482304, 0.68581115,
            0.68675602, 0.68672823, 0.68903381, 0.69069177, 0.69280375,
            0.69221962, 0.69358135, 0.69375242, 0.69399543, 0.69406244,
            0.6961142, 0.69472159, 0.69638078, 0.69648525, 0.69629432,
            0.69639879, 0.69670654, 0.69784544, 0.69730352, 0.69812849,
            0.69806364, 0.69823162, 0.69796483, 0.69792778
        ])
        return np.interp(p, xp=p_phi, fp=phi)

    def sync_em_abs(self):
        ss = self.source
        emissivity = np.sqrt(3.0) / 4.0 * self._interpolated_phi(ss.p) * np.pi * ss.nprime \
                   * cgs.e * cgs.e * cgs.e * ss.B / (cgs.me * cgs.c * cgs.c)
        Xp = self._interpolated_xi(ss.p)

        nu_m = 3.0 / (4.0 * np.pi) * Xp * ss.gm * ss.gm * cgs.e * ss.B / (cgs.me * cgs.c)
        nu_c = 0.286 * 3. * ss.gc * ss.gc * cgs.e * ss.B / (4.0 * np.pi * cgs.me * cgs.c)

        freq_arr = np.logspace(11., 27., 200)
        em_arr = np.zeros_like(freq_arr)
        abs_arr = np.zeros_like(freq_arr)
        for ifreq, freq in enumerate(freq_arr):
            if (nu_m <= nu_c):  # slow cooling
                if (freq < nu_m):
                    scaling = pow(freq / nu_m, 1.0 / 3.0)
                elif (freq >= nu_m and freq < nu_c):
                    scaling = pow(freq / nu_m, -1.0 * (ss.p - 1.0) / 2.0)
                else:
                    scaling = pow(nu_c / nu_m, -1.0 * (ss.p - 1.0) / 2.0) * pow(freq / nu_c, -1.0 * ss.p / 2.0)
            else:
                if (freq < nu_c):
                    scaling = pow(freq / nu_c, 1.0 / 3.0)
                elif (freq >= nu_c and freq < nu_m):
                    scaling = pow(freq / nu_c, -1.0 / 2.0)
                else:
                    scaling = pow(nu_m / nu_c, -1.0 / 2.0) * pow(freq / nu_m, -ss.p / 2.0)

            abs = np.sqrt(3) * pow(cgs.e, 3) * (ss.p - 1) * (ss.p + 2) * ss.nprime * ss.B \
                  / (16 * np.pi * cgs.me * cgs.me * cgs.c * cgs.c * ss.gm * freq * freq)
            if (freq < nu_m):
                abs_scaling = pow(freq / nu_m, 1.0 / 3.0)
            else:
                abs_scaling = pow(freq / nu_m, -0.5 * ss.p)

            em_pl = emissivity * scaling
            abs_pl = abs * abs_scaling

            em_arr[ifreq] = em_pl
            abs_arr[ifreq] = abs_pl

        return (freq_arr, em_arr, abs_arr)

# JOH06 formulation from PyBlastAfterglow
class SYN_JOH:
    def __init__(self, source: Source):
        self.source = source

    def sync_em_abs(self):
        ss = self.source
        p = ss.p
        gamToNuFactor = (3.0 / (4.0 * np.pi)) * (cgs.e * ss.B) / (cgs.me * cgs.c)
        XpF = 0.455 + 0.08 * p
        XpS = 0.06 + 0.28 * p
        phipF = 1.89 - 0.935 * p + 0.17 * (p * p)
        phipS = 0.54 + 0.08 * p
        kappa1 = 2.37 - 0.3 * p
        kappa2 = 14.7 - 8.68 * p + 1.4 * (p * p)
        kappa3 = 6.94 - 3.844 * p + 0.62 * (p * p)
        kappa4 = 3.5 - 0.2 * p
        kappa13 = -kappa1 / 3.0
        kappa12 = kappa1 / 2.0
        kappa11 = -1.0 / kappa1
        kappa2p = kappa2 * (p - 1.0) / 2.0
        kappa12inv = -1.0 / kappa2
        kappa33 = -kappa3 / 3.
        kappa3p = kappa3 * (p - 1.0) / 2.0
        kappa13inv = -1.0 / kappa3
        kappa42 = kappa4 / 2.0
        kappa14 = -1.0 / kappa4

        freq_arr = np.logspace(11., 27., 200)
        em_arr = np.zeros_like(freq_arr)
        abs_arr = np.zeros_like(freq_arr)

        for ifreq, freq in enumerate(freq_arr):
            if (ss.gm < ss.gc): # slow cooling
                nu_m = XpS * ss.gm * ss.gm * gamToNuFactor
                nu_c = XpS * ss.gc * ss.gc * gamToNuFactor
                emissivity = 11.17 * (p - 1.0) / (3.0 * p - 1.0) * (0.54 + 0.08 * p) \
                             * cgs.e * cgs.e * cgs.e * ss.nprime * ss.B / (cgs.me * cgs.c * cgs.c)
                scaling = pow(pow(freq / nu_m, kappa33) + pow(freq / nu_m, kappa3p), kappa13inv) \
                        * pow(1. + pow(freq / nu_c, kappa42), kappa14)

                _alpha = 7.8 * phipS * pow(XpS, -(4 + p) / 2.) * (p + 2) * (p - 1) \
                       * cgs.e / cgs.mp / (p + 2 / 3.)
                abs = _alpha * ss.nprime * cgs.mp * pow(ss.gm, -5) / ss.B
                if (freq <= nu_m):
                    abs_scaling = pow(freq / nu_m, -5 / 3.)
                elif ((nu_m < freq) and (freq <= nu_c)):
                    abs_scaling = pow(freq / nu_m, -(p + 4) / 2)
                elif (nu_c < freq):
                    abs_scaling = pow(nu_c / nu_m, -(p + 4) / 2) * pow(freq / nu_c, -(p + 5) / 2)
                else:
                    raise ValueError()

            else: # fast cooling
                nu_m = XpF * ss.gm * ss.gm * gamToNuFactor
                nu_c = XpF * ss.gc * ss.gc * gamToNuFactor
                _phip = 2.234 * phipF
                emissivity = _phip * cgs.e * cgs.e * cgs.e * ss.nprime * ss.B / (cgs.me * cgs.c * cgs.c)
                scaling = pow(pow(freq / nu_c, kappa13) + pow(freq / nu_c, kappa12), kappa11) \
                        * pow(1. + pow(freq / nu_m, kappa2p), kappa12inv)

                _alpha = 11.7 * phipF * pow(XpF, -3) * cgs.e / cgs.mp
                abs = _alpha * (ss.nprime * cgs.mp) * pow(ss.gc, -5) / ss.B
                if (freq <= nu_c):
                    abs_scaling = pow(freq / nu_c, -5 / 3.)
                elif ((nu_c < freq) and (freq <= nu_m)):
                    abs_scaling = pow(freq / nu_c, -3)
                elif (nu_m < freq):
                    abs_scaling = pow(nu_m / nu_c, -3) * pow(freq / nu_m, -(p + 5) / 2)
                else:
                    raise ValueError()

            em_pl = emissivity * scaling
            abs_pl = abs * abs_scaling

            em_arr[ifreq] = em_pl
            abs_arr[ifreq] = abs_pl

        return (freq_arr, em_arr, abs_arr)

# using naima package
class SSC_NAIMA:
    def __init__(self, source : Source):
        self.source = source

    def syn_em(self):
        ss = self.source
        ampl = 1. / u.eV



        e_0 = (max(2., min(ss.gm, ss.gc)) * cgs.mec2) / erg2TeV * u.TeV
        amplitude_slow = ss.n_e * (ss.p - 1) / (max(2., ss.gm) * cgs.mec2) * min(1., (ss.gm / 2.) ** (ss.p - 1))
        amplitude_fast = ss.n_e / (max(2., ss.gc) * cgs.mec2) * min(1., ss.gc / 2.)
        amplitude = amplitude_slow if ss.gm < ss.gc else amplitude_fast
        ebreak = max(2., max(ss.gm, ss.gc)) * cgs.mec2 / erg2TeV * u.TeV
        e_syn_max = max((max(2., min(ss.gm, ss.gc)) + 1) * cgs.mec2, ss.gM * cgs.mec2) / erg2TeV * u.TeV

        ECBPL = ExponentialCutoffBrokenPowerLaw(
            amplitude= 1. / u.eV,  # temporary amplitude
            e_0= 1. * u.TeV,
            e_break=ebreak,
            alpha_1=ss.p if ss.gm < ss.gc else 2.,
            alpha_2=ss.p + 1,
            e_cutoff=e_syn_max
        )
        ener = np.logspace(2, np.log10(20 * 1e13), 100) * u.eV
        eldis = ECBPL(ener)
        plt.loglog(ener, eldis)
        plt.show()
        ra = naima.utils.trapz_loglog(ener * eldis, ener) / naima.utils.trapz_loglog(eldis, ener)

        rat = (1. * ss.Gamma * cgs.mp * cgs.c * cgs.c) / erg2TeV * u.TeV
        emin = rat / ra * 1e9 * u.eV
        SYN = Synchrotron(ECBPL, B=ss.B * u.G, Eemin=emin, Eemax=e_syn_max, nEed=20)
        self.Wesyn = SYN.compute_We(Eemin=emin, Eemax=e_syn_max)

        # electrons = ExponentialCutoffBrokenPowerLaw(
        #     amplitude=amplitude  * u.erg,
        #     e_0=e_0 * u.erg,
        #     e_break=ebreak  * u.erg,
        #     alpha_1=ss.p if ss.gm < ss.gc else 2.,
        #     alpha_2=ss.p + 1,
        #     e_cutoff=e_syn_max * u.erg,
        #     beta=2.0,
        # )

        SYN = Synchrotron(
            electrons,
            B=ss.B * u.G,
            Eemax=electrons.e_cutoff,
            Eemin=electrons.e_0,
            nEed=50,
        )

        freq_arr = np.logspace(11., 27., 200)
        e = freq_arr / cgs.eV2Hz * cgs.eV2erg
        flux = SYN.flux(energy=e,distance=0) * cgs.eV2erg

        return (freq_arr, flux)



# ------------------- TASKS ---------------------
def compare_syn(source : Source):
    # bret = SSC_Berretta(source = source)
    wspn = SYN_WSPN99(source = source)
    john = SYN_JOH(source = source)
    naim = SSC_NAIMA(source = source)

    # freq_arr, em_arr, abs_arr = bret.sync_em_abs()
    # plt.loglog(freq_arr,em_arr,color='blue',ls='-',label='Em.Berretta')
    # plt.loglog(freq_arr,abs_arr,color='blue',ls='--',label='Abs.Berretta')

    # freq_arr, em_arr = mcgr.sync_em_abs()
    # plt.loglog(freq_arr,em_arr,color='pink',ls='-',label='Em.McGrath')
    # plt.loglog(freq_arr,abs_arr,color='green',ls='--',label='Abs.McGrath')

    freq_arr, em_arr = naim.syn_em()
    plt.loglog(freq_arr,em_arr,color='pink',ls='-',label='Em.McGrath')
    # plt.loglog(freq_arr,abs_arr,color='green',ls='--',label='Abs.McGrath')

    freq_arr, em_arr, abs_arr = wspn.sync_em_abs()
    plt.loglog(freq_arr,em_arr,color='green',ls='-',label='Em.WSPN')
    plt.loglog(freq_arr,abs_arr,color='green',ls='--',label='Abs.WSPN')

    freq_arr, em_arr, abs_arr = john.sync_em_abs()
    plt.loglog(freq_arr,em_arr,color='red',ls='-',label='Em.WSPN')
    plt.loglog(freq_arr,abs_arr,color='red',ls='--',label='Abs.WSPN')

    plt.grid()
    plt.legend()
    plt.show()

# -----------------------------------------------
if __name__ == '__main__':
    # load PBA instance to interface with code output for blastwave dynamics
    pba = PBA.interface.PyBlastAfterglow(workingdir=os.getcwd()+'/',
                                         readparfileforpaths=True,
                                         verbose=False)

    # pba.GRB.get_dyn_arr(v_n="")
    source = Source()
    compare_syn(source=source)