import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import MultipleLocator

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Ariel",
        "font.size": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
    }
)


class ModelComparisonMetrics:
    """
    Class to compare the performance of different models
    """

    def __init__(self, dataset, models=["grz", "white", "hybrid"], cv_idx=None, **kwargs):
        """
        Initialize the ModelComparisonMetrics object.

        Parameters
        ----------
        dataset : dataset.DataSet
            The dataset object used for training and testing.
        models : list of str, optional
            List of models to be used (default: ["grz", "white", "hybrid"]).
        cv_idx : list of int or None, optional
            List of indices for cross-validation (default: None).
        **kwargs : dict
            Additional keyword arguments.
            - snr_lim : list of float, tuple, or None, optional
                List of SNR limits (default: None).
            - snr_binsize : float, optional
                SNR binsize (default: 0.5).
            - mag_lim : list of float, tuple, or None, optional
                List of r-band magnitude limits (default: None).
            - mag_binsize : float, optional
                Magnitude binsize (default: 1.0).
        """
        self._color_s = "#fc8d62"
        self._color_g = "#8da0cb"

        self._color_model = dict(
            grz="#984ea3", white="#377eb8", White="#377eb8", hybrid="#4daf4a", Hybrid="#4daf4a", LS="black"
        )

        self._label_model = dict(
            grz=r"$grz$",
            white=r"$\mathrm{White}$",
            White=r"$\mathrm{White}$",
            hybrid=r"$\mathrm{Hybrid}$",
            Hybrid=r"$\mathrm{Hybrid}$",
            LS=r"$\mathrm{LS}$",
        )

        self.models = models
        self.dataset = dataset
        self.cv_idx = cv_idx

        self.bins_lim = dict(snr=None, mag=None)
        self.bins_size = dict(snr=kwargs.get("snr_binsize", 0.5), mag=kwargs.get("mag_binsize", 1.0))
        self.bins = dict(snr=None, mag=None)

        for key in ["snr", "mag"]:
            self.bins_lim[key] = kwargs.get(key + "_lim", None)
            self.bins_size[key] = kwargs.get(key + "_binsize", None)
            if self.bins_lim[key] is not None:
                self.bins[key] = np.arange(
                    self.bins_lim[key][0], self.bins_lim[key][1] + self.bins_size[key], self.bins_size[key]
                )
        self.bins_results = {}

    def data_binning(self, idx=None, bins_by=None, verbose=True, **kwargs):
        """
        Bin the data based on a specified criterion.

        Parameters:
        ----------
        idx : numpy.ndarray, optional
            Boolean array indicating which data points to include in the binning. Default is None, which includes all data points.
        bins_by : str, optional
            The criterion to use for binning. Must be either 'snr' or 'mag'. Default is None.
        verbose : bool, optional
            Whether to print the number of stars and galaxies in each bin. Default is True.
        **kwargs : dict, optional
            Additional keyword arguments:
            - lim : tuple, optional
                The lower and upper limits of the bins. Default is None.
            - binsize : float, optional
                The size of each bin. Default is the value specified in self.bins_size for the given bins_by criterion.

        Raises:
        -------
        ValueError
            If bins_by is not 'snr' or 'mag'.

        Returns:
        -------
        dict
            A dictionary containing the bins and bin indices.
            - n_bin : int
                The number of bins.
            - bins : numpy.ndarray
                The bin edges.
            - bin_idx : numpy.ndarray
                A boolean array indicating the indices of the dataset in each bin.
            - title : str
                The title of the plot ('snr' or 'mag').
        """

        if idx is None:
            idx = np.ones(len(self.dataset.ds), dtype=bool)
        y_true = self.dataset.y_true
        ds = self.dataset.ds

        if bins_by is not None:
            if bins_by == "snr":
                data_to_bin = np.log10(ds.snr)
                title = r"$\log(\mathrm{S/N})$"
            elif bins_by == "mag":
                data_to_bin = ds.mag_r
                title = r"$r\ [\mathrm{mag}]$"
            else:
                raise ValueError("bins_by must be 'snr' or 'mag'")
        else:
            data_to_bin = None
            title = None

        lim = kwargs.get("lim", None)
        binsize = kwargs.get("binsize", self.bins_size.get(bins_by, None))
        if lim is None:
            bins = self.bins.get(bins_by, None)
        else:
            assert binsize is not None, "binsize must be provided"
            bins = np.arange(lim[0], lim[1] + binsize, binsize)
        if bins is None:  # the entire dataset
            n_bin = 1
        else:  # -inf and inf are added to the bins
            bins = np.concatenate(([-np.inf], bins, [np.inf]))
            n_bin = len(bins) - 1

        bin_idx = np.ones((n_bin, len(ds)), dtype=bool)

        N_star, N_gal = np.zeros(n_bin, dtype=int), np.zeros(n_bin, dtype=int)
        N_tot = np.zeros(n_bin, dtype=int)

        for j in range(n_bin):
            if bins_by is not None:
                bin_idx[j] = (data_to_bin > bins[j]) & (data_to_bin <= bins[j + 1]) & idx & (~np.isinf(data_to_bin))
            else:
                bin_idx[j] = idx
            N_star[j] = (y_true & bin_idx[j]).sum()
            N_gal[j] = (~y_true & bin_idx[j]).sum()
            N_tot[j] = bin_idx[j].sum()
            if verbose:
                print(f"Bin {j}: Star = {N_star[j]}; Galaxy = {N_gal[j]}")

        Bins = dict(n_bin=n_bin, bins=bins, bin_idx=bin_idx, title=title, N_star=N_star, N_gal=N_gal, N_tot=N_tot)
        self.bins_results[bins_by] = Bins
        return Bins

    def plot_score_hist(self, idx=None, bins_by=None, **kwargs):
        """
        Plots a histogram of the score distribution.

        Parameters:
        -----------
        idx : numpy.ndarray, optional
            Boolean array indicating which data points to include in the histogram. Default is None, which includes all data points.
        bins_by : str, optional
            The criterion to use for binning. Must be either 'snr' or 'mag'. Default is None.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the data_binning method.
            - hist_bins : int, optional
            - lim : tuple, optional
            - binsize : float, optional

        Returns:
        --------
        matplotlib.axes.Axes
            The axes object containing the histogram.

        """
        # Get the true labels from the dataset
        y_true = self.dataset.y_true

        # Perform data binning
        self.data_binning(idx=idx, bins_by=bins_by, **kwargs)
        bins = self.bins_results[bins_by]["bins"]
        bin_idx = self.bins_results[bins_by]["bin_idx"]
        title = self.bins_results[bins_by]["title"]

        # Calculate the number of columns and rows for subplots
        n_col = self.bins_results[bins_by]["n_bin"]
        n_row = len(self.models)

        # Create subplots
        _, ax = plt.subplots(
            ncols=n_col,
            nrows=n_row,
            figsize=(1 + n_col * 3, n_row * 3),
            sharey=True,
            sharex=True,
            constrained_layout=True,
        )

        ax = ax.ravel().reshape((n_row, n_col))

        for j, arg in enumerate(bin_idx):
            for k, p in enumerate([self.dataset.pred[model] for model in self.models]):
                ax[k, j].hist(
                    p[y_true & arg],
                    histtype="step",
                    label=r"$\mathrm{Star}$",
                    color=self._color_s,
                    bins=kwargs.get("hist_bins", 50),
                    range=([0, 1]),
                    log=True,
                )
                ax[k, j].hist(
                    p[~y_true & arg],
                    histtype="step",
                    label=r"$\mathrm{Galaxy}$",
                    color=self._color_g,
                    bins=kwargs.get("hist_bins", 50),
                    range=([0, 1]),
                    log=True,
                )
        if n_col > 1:
            for j in range(n_col):
                ax[-1, j].set_xlabel(r"$\mathrm{Score}$")
                if j == 0:
                    ax[0, j].set_title(f"{title} $\le$ ${bins[j+1]:.1f}$", fontsize=22.5)
                elif j == n_col - 1:
                    ax[0, j].set_title(f"{title} $>$ ${bins[j]:.1f}$", fontsize=22.5)
                else:
                    ax[0, j].set_title(f"${bins[j]:.1f}$ $<$ {title} $\le$ ${bins[j+1]:.1f}$", fontsize=22.5)
        for k in range(n_row):
            ax[k, 0].xaxis.set_major_locator(MultipleLocator(0.5))
            ax[k, 0].xaxis.set_minor_locator(MultipleLocator(0.05))
            import matplotlib.ticker as ticker

            ax[k, 0].yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
            ax[k, 0].yaxis.set_minor_locator(
                ticker.LogLocator(base=10, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=10)
            )
            ax[k, 0].set_ylabel(r"$\mathrm{Counts}$")
            ax[k, -1].text(
                1.05,
                0.5,
                self._label_model.get(self.models[k], self.models[k]),
                transform=ax[k, -1].transAxes,
                rotation=90,
                va="center",
            )

        ax[0, -1].legend(loc="upper right", fontsize=15)

        plt.ylim(0.8, None)
        return ax

    def plot_ROC(self, idx=None, bins_by=None, **kwargs):
        """
        Plot the Receiver Operating Characteristic (ROC) curve for the training set in the cross-validation or
        for the test set in the bootstrapping.

        Parameters
        ----------
        idx : int or None, optional
            Index of the data points to include in the ROC curve. Default is None.
        bins_by : str or None, optional
            Binning method for data points. Default is None.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the data_binning method.
            - cv_idx : list of int, optional
            - lim : tuple, optional
            - binsize : float, optional

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes object containing the ROC curve plot.
        """
        from sklearn.metrics import roc_curve

        # Get the true labels from the dataset
        y_true = self.dataset.y_true

        # Perform data binning
        self.data_binning(idx=idx, bins_by=bins_by, **kwargs)
        bins = self.bins_results[bins_by]["bins"]
        bin_idx = self.bins_results[bins_by]["bin_idx"]
        title = self.bins_results[bins_by]["title"]

        # Calculate the number of columns and rows for subplots
        n_col = self.bins_results[bins_by]["n_bin"]

        # Create subplots
        _, ax = plt.subplots(
            ncols=n_col,
            nrows=1,
            figsize=(1 + 3 * n_col, 1 + 3),
            sharey=True,
            sharex=True,
            constrained_layout=True,
        )

        cv_idx = kwargs.get("cv_idx", self.cv_idx)

        ax = np.atleast_1d(ax)

        for j, arg in enumerate(bin_idx):
            for k, p in enumerate([self.dataset.pred[model] for model in self.models]):
                # overall ROC curve
                fpr, tpr, _ = roc_curve(y_true[arg], p[arg])
                ax[j].plot(
                    fpr,
                    tpr,
                    label=self._label_model.get(self.models[k], self.models[k]),
                    color=self._color_model.get(self.models[k], "black"),
                    lw=2,
                )
                # each fold in the cross-validation
                if cv_idx is not None:
                    for l in range(len(cv_idx)):
                        eval_idx = np.array([False] * len(self.dataset.ds))
                        eval_idx[cv_idx[l]] = True
                        fpr, tpr, _ = roc_curve(y_true[arg & eval_idx], p[arg & eval_idx])
                        ax[j].plot(fpr, tpr, color=self._color_model.get(self.models[k], "black"), lw=0.5, alpha=0.25)
            # LS classifier
            try:
                LS_star = np.array(self.dataset.ds.type)[arg] == "PSF"
                ax[j].scatter(
                    ((~y_true[arg]) & LS_star).sum() / (~y_true[arg]).sum(),
                    (y_true[arg] & LS_star).sum() / y_true[arg].sum(),
                    marker="*",
                    edgecolor="k",
                    facecolor="none",
                    s=250,
                    zorder=100,
                )
            except:
                print("No LS classifier")
            ax[j].set_xscale("log")
            ax[j].axvline(0.005, color="0.8", linestyle="-.", linewidth=5, zorder=-10)
            ax[j].set_xlabel(r"$\mathrm{FPR}$")
        ax[0].set_ylabel(r"$\mathrm{TPR}$")
        ax[0].legend(prop={"size": 15}, loc=4)

        # if all elements in bins are integers
        if n_col > 1:
            if all(x == np.round(x) for x in bins[1:-1]):
                digits = 0
            else:
                digits = 1

            for j in range(n_col):
                if j == 0:
                    ax[j].set_title(f"{title} $\le$ ${bins[j+1]:.{digits}f}$", fontsize=22.5)
                elif j == n_col - 1:
                    ax[j].set_title(f"{title} $>$ ${bins[j]:.{digits}f}$", fontsize=22.5)
                else:
                    ax[j].set_title(f"${bins[j]:.{digits}f}$ $<$ {title} $\le$ ${bins[j+1]:.{digits}f}$", fontsize=22.5)
        return ax

    def plot_FPFN(self, idx=None, bins_by=None, thresh=0.5, **kwargs):
        """
        Plot the FPR/FNR for different models.

        Parameters:
        ----------
        idx : int or None, optional
            Index of the data to be plotted. If None, all data will be plotted.
        bins_by : str or None, optional
            Variable to bin the data by. If None, no binning will be performed.
        thresh : float, optional
            Threshold value for classification. Default is 0.5.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the data_binning method.

        Returns:
        -------
        ax : matplotlib.axes.Axes
            The matplotlib axes object containing the plotted FPR/FNR.
        """

        # Perform data binning
        self.data_binning(idx=idx, bins_by=bins_by, **kwargs)
        bins = self.bins_results[bins_by]["bins"][1:-1]
        bins_center = (bins[1:] + bins[:-1]) / 2
        bin_idx = self.bins_results[bins_by]["bin_idx"][1:-1]
        xlabel = self.bins_results[bins_by]["title"]

        fig, ax = plt.subplots(2, 1, figsize=(6, 7), sharex=True, constrained_layout=True)

        cv_idx = kwargs.get("cv_idx", self.cv_idx)

        models = np.append(["LS"], self.models)
        for l, model in enumerate(models):
            Acc = dict(tpr=[], tnr=[], acc=[])
            Acc_err = dict(tpr=[], tnr=[], acc=[])
            for arg in bin_idx:
                star = self.dataset.y_true[arg]
                Acc["tpr"].append((star & (self.dataset.pred[model][arg] > thresh)).sum() / (star).sum())
                Acc["tnr"].append(((~star) & (~(self.dataset.pred[model][arg] > thresh))).sum() / (~star).sum())
                if cv_idx is not None:
                    tpr, tnr = [], []
                    for j in range(len(cv_idx)):
                        eval_idx = np.array([False] * len(self.dataset.ds))
                        eval_idx[cv_idx[j]] = True
                        star = self.dataset.y_true[arg & eval_idx]
                        tpr.append(
                            (star & (self.dataset.pred[model][arg & eval_idx] > thresh)).sum() / (star).sum()
                        )
                        tnr.append(
                            ((~star) & (~(self.dataset.pred[model][arg & eval_idx] > thresh))).sum() / (~star).sum()
                        )
                    Acc_err["tpr"].append(np.std(tpr, ddof=1))
                    Acc_err["tnr"].append(np.std(tnr, ddof=1))
                else:
                    Acc_err["tpr"].append(0)
                    Acc_err["tnr"].append(0)

            for k, key in enumerate(["tpr", "tnr"]):
                ax[k].errorbar(
                    bins_center + 0.075 * (bins_center[1] - bins_center[0]) * (l - len(models) / 2 + 0.5),
                    1 - np.array(Acc[key]),
                    yerr=Acc_err[key],
                    color=self._color_model.get(model, "black"),
                    marker="s" if model == "LS" else "o",
                    ms=6,
                    lw=0.75,
                    ls="--" if model == "LS" else "-",
                    label=self._label_model.get(model, model),
                )

        ax[1].set_xlabel(xlabel)
        ax[0].set_ylabel(r"$\mathrm{FNR}$")
        ax[1].set_ylabel(r"$\mathrm{FPR}$")
        # ax[2].set_ylabel(r"$\mathrm{Accuracy}$")
        ax[0].legend(prop={"size": 17.5})

        return ax
