import copy
import os
import sys
import h5py
import numpy as np


def kinetic_energy(velocity, mass):
    solar_m = 1.989e+33
    c = 2.9979e10
    """ velocity [c] mass [Msun] -> Ek [ergs]"""
    return mass * solar_m * (velocity * c) ** 2

class LoadFile():

    def __init__(
            self,
            path
    ):
        if not os.path.isdir(path):
            raise IOError("no data dir: {}".format(path))
        self.rootpath = path

    def load_tot_mass(self, fname="/total_flux.dat"):
        times, flux, masses = np.loadtxt(self.rootpath + fname, unpack=True)
        return (times, flux, masses)

    def load_vinf_hist(self, fname="/hist_vel_inf.dat"):
        hist1 = np.loadtxt(self.rootpath + fname)
        assert len(hist1) > 1
        vinf, mass = hist1[:, 0], hist1[:, 1]
        return (vinf, mass)

    def load_theta_hist(self, fname="/hist_theta.dat"):
        hist1 = np.loadtxt(self.rootpath + fname)
        theta, mass = hist1[:, 0], hist1[:, 1]
        return (theta, mass)

    def load_Ye_hist(self, fname="hist_Y_e.dat"):
        hist1 = np.loadtxt(self.rootpath + fname)
        ye, mass = hist1[:, 0], hist1[:, 1]
        return (ye, mass)

    def load_logrho_hist(self, fname="hist_logrho.dat"):
        hist1 = np.loadtxt(self.rootpath + fname)
        logrho, mass = hist1[:, 0], hist1[:, 1]
        return (logrho, mass)

    def load_corr_file(self, fname="corr_vel_inf_theta.h5"):

        # load the corr file
        dfile = h5py.File(self.rootpath + fname, mode="r")
        # for key in dfile.keys():
        #     print(key, np.array(dfile[key]).shape)

        # extract arrays of data
        thetas = np.array(dfile["theta"])
        vinfs = np.array(dfile["vel_inf"])
        mass = np.array(dfile["mass"])  # [i_theta, i_vinf]

        # if not len(mass[:, 0]) == len(thetas):
        #     raise ValueError("Histogram mismatch :: mass[:, 0]={} theta={}".format(len(mass[:, 0]), len(thetas)))
        #
        # if not len(mass[0, :]) == len(vinfs):
        #     # vinfs = vinfs[2:]
        #     if not len(mass[0, :]) == len(vinfs):
        #         raise ValueError("Histogram mismatch :: mass[0, :]={} vinf={}".format(len(mass[0, :]), len(vinfs)))
        # print("Loaded")
        return (thetas, vinfs, mass)

    def load_time_corr_vinf(self, fname="timecorr_vel_inf.h5"):

        dfile = h5py.File(self.rootpath + fname, mode="r")
        for key in dfile.keys():
            print(key, np.array(dfile[key]).shape)

        # extract arrays of data
        times = np.array(dfile["time"])
        vinfs = np.array(dfile["vel_inf"])
        mass = np.array(dfile["mass"]).T  # [i_theta, i_vinf]

        if not len(mass[:, 0]) == len(times):
            raise ValueError("timecorr mismatch :: mass[:, 0]={} times={}".format(len(mass[:, 0]), len(times)))

        if not len(mass[0, :]) == len(vinfs):
            vinfs = vinfs[2:]
            if not len(mass[0, :]) == len(vinfs):
                raise ValueError("Histogram mismatch :: mass[0, :]={} vinf={}".format(len(mass[0, :]), len(vinfs)))

        return (times, vinfs, mass)

    def load_time_corr_theta(self, fname="timecorr_theta.h5"):

        dfile = h5py.File(self.rootpath + fname, mode="r")
        for key in dfile.keys():
            print(key, np.array(dfile[key]).shape)

        # extract arrays of data
        times = np.array(dfile["time"])
        theta = np.array(dfile["theta"])
        mass = np.array(dfile["mass"]).T  # [i_theta, i_vinf]

        if not len(mass[:, 0]) == len(times):
            raise ValueError("timecorr mismatch :: mass[:, 0]={} times={}".format(len(mass[:, 0]), len(times)))

        if not len(mass[0, :]) == len(theta):
            theta = theta[2:]
            if not len(mass[0, :]) == len(theta):
                raise ValueError("Histogram mismatch :: mass[0, :]={} theta={}".format(len(mass[0, :]), len(theta)))

        return (times/1e3, theta, mass)

class Ejecta(LoadFile):

    def __init__(self, path):
        super(Ejecta, self).__init__(path)

    def get_tot_mass(self):
        _, _, masses=self.load_tot_mass()
        return masses[-1]

    def get_hist(self, v_n):
        if v_n == "theta":
            return self.load_theta_hist()
        elif v_n == "vinf" or v_n == "vel_inf":
            return self.load_vinf_hist()
        elif v_n == "Ye" or v_n == "Y_e":
            return self.load_Ye_hist()
        else:
            raise NameError("No data file for hist:{}".format(v_n))

    def get_theta_rms(self):
        """ root mean squared angle of the module_ejecta (1 planes) To get for 2 planes, multiply by 2 """
        theta, theta_M = self.load_theta_hist()
        theta -= np.pi / 2.
        theta_rms = (180. / np.pi) * np.sqrt(np.sum(theta_M * theta ** 2) / np.sum(theta_M))
        value = np.float64(theta_rms)
        return (value) # [degrees]

    def get_total_ek(self):
        hist_vinf, hist_vinf_M = self.load_vinf_hist("hist_vel_inf.dat")
        hist_vinf = hist_vinf[hist_vinf_M > 0.]
        hist_vinf_M = hist_vinf_M[hist_vinf_M > 0.]
        hist_ek = kinetic_energy(hist_vinf, hist_vinf_M)
        return np.sum(hist_ek)

    def get_vej_ave(self, vmin=0.):
        hist_vinf, hist_vinf_M = self.load_vinf_hist("hist_vel_inf.dat")
        hist_vinf = hist_vinf[hist_vinf_M > 0.]
        hist_vinf_M = hist_vinf_M[hist_vinf_M > 0.]
        hist_vinf_M = hist_vinf_M[hist_vinf > vmin]
        hist_vinf = hist_vinf[hist_vinf > vmin]
        assert len(hist_vinf_M) > 0
        vel_ave = np.sum(hist_vinf * hist_vinf_M) / np.sum(hist_vinf_M)
        return vel_ave

    def get_ye_ave(self, ye_min=0.):
        hist_ye, hist_ye_M = self.load_Ye_hist("hist_Y_e.dat")
        hist_ye = hist_ye[hist_ye_M > 0.]
        hist_ye_M = hist_ye_M[hist_ye_M > 0.]
        hist_ye_M = hist_ye_M[hist_ye > ye_min]
        hist_ye = hist_ye[hist_ye > ye_min]
        assert len(hist_ye_M) > 0
        vel_ave = np.sum(hist_ye * hist_ye_M) / np.sum(hist_ye_M)
        return vel_ave

if __name__ == '__main__':
    pass
