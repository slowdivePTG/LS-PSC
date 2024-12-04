import numpy as np


class DataSet:
    """
    A class representing a dataset.

    Parameters
    ---------
    ds : dict
        The dataset.
    y_true : array-like
        The true labels.
    manual_mask : str, optional
        The manual mask. Default is an empty string.
    scored : bool, optional
        Whether the dataset is already scored. Default is False.
    """

    def __init__(self, ds, y_true, manual_mask="", scored=False):
        self.ds = ds.copy()
        self.y_true = y_true

        if scored:  # the dataset is already scored
            self.pred = {}
            for key in ds.keys():
                if "score" in key:
                    k = key.split("_")[0]
                    self.pred[k] = ds[key]
            return

        # available filters in the dataset
        # dr9 has grz filters
        # dr10 has griz filters
        flt_avail = [mag.split("_")[-1] for mag in ds.keys() if "mag" in mag]
        # if the filter is not available, manually mask it
        self.manual_mask = [m for m in manual_mask] + [flt for flt in "griz" if flt not in flt_avail]

        # missing dchisq
        self.ds["missing_dchisq_1"] = self.ds.dchisq_1 == 0
        self.ds["missing_dchisq_gal"] = (
            (self.ds.dchisq_2 == 0) & (self.ds.dchisq_3 == 0) & (self.ds.dchisq_4 == 0) & (self.ds.dchisq_5 == 0)
        )

        # missing apflux
        for flt in "griz":
            flt_masked = 1
            if flt in flt_avail:  # if the filter is available
                for k in range(8):
                    flt_masked &= np.array(self.ds[f"apflux_ivar_{flt}_{k+1}"], dtype=float) <= 0
                self.ds[f"apflux_masked_{flt}"] = flt_masked
                self.ds[f"masked_{flt}"] = self.ds[f"apflux_masked_{flt}"] | (self.ds[f"snr_{flt}"] <= 0)
            else:  # if the filter is not available (i band in dr9 data) - manually mask
                self.ds[f"apflux_masked_{flt}"] = np.ones(len(self.ds), dtype=bool)
                self.ds[f"masked_{flt}"] = np.ones(len(self.ds), dtype=bool)

        # snr
        snr2_flt = {"g": 0, "r": 0, "i": 0, "z": 0}
        for flt in "griz":
            if flt in self.manual_mask:  # keep 0 if manually masked
                continue
            snr2_flt[flt] = np.where(
                self.ds[f"snr_{flt}"] > 0,
                self.ds[f"snr_{flt}"] ** 2,
                0,
            )
        snr2 = (snr2_flt["g"] + snr2_flt["r"] + snr2_flt["i"] + snr2_flt["z"]) / 4
        # assert np.all(snr2 > 0), "SNR is not positive"
        if not np.all(snr2 > 0):
            print("SNR is not positive")
        snr = np.where(snr2 > 0, snr2**0.5, 0).ravel()
        self.ds["snr"] = snr

        # white_mag
        flux_white = 0
        for flt in "griz":
            if flt in self.manual_mask:  # skip if manually masked
                continue
            flux = np.where(snr2_flt[flt] > 0, 10 ** (-0.4 * self.ds[f"mag_{flt}"]), 0)
            flux_white += snr2_flt[flt] * flux
        flux_white = np.where(snr2 > 0, flux_white / (snr2 * 4), np.nan)
        self.ds["mag_white"] = -2.5 * np.log10(flux_white)

        # dchisq
        for i in range(1, 5):
            with np.errstate(divide="ignore", invalid="ignore"):
                dchisq_k = np.where(ds[f"dchisq_{i+1}"] > 0, ds[f"dchisq_{i+1}"], 0)
                self.ds[f"dchisq_{i+1}-1_norm"] = np.where(
                    (ds[f"dchisq_1"] > 0),
                    (dchisq_k - ds["dchisq_1"]) / snr2,
                    0,
                )
        forced_PSF = (self.ds.fitbits & (2**0 + 2**12) != 0) & self.ds.missing_dchisq_gal
        not_fit = self.ds.missing_dchisq_1 & self.ds.missing_dchisq_gal
        X_mask_type = np.where(forced_PSF, 2, np.where(not_fit, 1, 0))
        X_dchi2 = [X_mask_type] + [self.ds[f"dchisq_{i+1}-1_norm"] for i in range(1, 5)]

        # X_dchi2 = [self.ds[f"dchisq_{i+1}-1_norm"] for i in range(1, 5)]

        # fiducial - normalized dchisq
        self.X = np.array(X_dchi2).T
        # self.X_seeing = np.array(X_seeing + X_dchi2).T

        # normalized dchisq + AP
        self.X_AP = {}
        self.X_AP["grz"] = np.array(
            X_dchi2 + self.X_flt_ap("g", ds) + self.X_flt_ap("r", ds) + self.X_flt_ap("z", ds)
        ).T

        # normalized dchisq + AP (white = g+r+i+z)
        self.X_white = np.array(X_dchi2 + self.X_white_ap("griz", ds)).T

        # normalized dchisq + AP (g+r) + AP (i+z), trained separately, weighted average
        mask_gr = self.ds[f"masked_g"] & self.ds[f"masked_r"]
        mask_iz = self.ds[f"masked_i"] & self.ds[f"masked_z"]
        self.weighted_X_white_br = np.array(
            [mask_gr] + [mask_iz] + X_dchi2 + self.X_white_ap("gr", ds) + self.X_white_ap("iz", ds)
        ).T

        self.pred = {}

    # aperture phot
    def X_flt_ap(self, flt, table):
        """
        Calculate the aperture photometry for a given filter.

        Parameters
        ---------
        flt : str
            The filter (g, r, i, z)
        table : dict
            The dataset.

        Returns
        -------
        A list of aperture flux ratios.
        """
        N_flt = 8

        if flt in self.manual_mask:
            return [np.nan * np.ones(len(table))] * (N_flt - 1)

        flux = np.empty((N_flt, len(table)))
        mask = np.empty((N_flt, len(table)))
        for i in range(N_flt):
            mask[i] = table[f"apflux_ivar_{flt}_{i+1}"] == 0  # find saturated/not observed pixels
            flux[i] = np.where(
                mask[i],
                0,  # set to 0 if masked
                table[f"apflux_{flt}_{i+1}"],
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            return [
                np.where(
                    mask[i + 1],
                    np.nan,  # NaN if masked
                    np.where(
                        flux[i + 1] >= table[f"apflux_ivar_{flt}_{i+2}"] ** -0.5 * 3,
                        flux[i] / flux[i + 1],
                        np.nan,  # NaN if not 3-sigma detection
                    ),
                )
                for i in range(N_flt - 1)
            ]

    # white flux
    def X_white_ap(self, flts, table):
        """
        Calculate the white flux for a given set of filters.

        Parameters
        ---------
        flts : str
            The filters (a combination of g, r, i, z)
        table : dict
            The dataset.

        Returns:
        A list of white aperture flux ratios.
        """
        N_flt = 8
        white_flux = []
        for i in range(N_flt):
            apflux_w, w = 0, 0
            for flt in flts:
                if flt in self.manual_mask:  # skip if manually masked
                    continue
                fl = table[f"apflux_{flt}_{i+1}"]
                ivar = table[f"apflux_ivar_{flt}_{i+1}"]
                ivar_cal = np.where(ivar > 0, ivar, 0)
                snr2 = fl**2 * ivar_cal  # weight = snr^2
                apflux_w += np.where(fl > 0, fl, 0) * snr2
                w += snr2
            with np.errstate(divide="ignore", invalid="ignore"):
                white_flux.append(np.where(w > 0, apflux_w / w, 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            return [
                np.where(
                    white_flux[i + 1] > 0,
                    (white_flux[i] / white_flux[i + 1]),
                    np.nan,
                )
                for i in range(N_flt - 1)
            ]
