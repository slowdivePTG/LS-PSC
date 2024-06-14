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

color_s = "#fc8d62"
color_g = "#8da0cb"

color_model = dict(grz="#984ea3", white="#377eb8", White="#377eb8", hybrid="#4daf4a", Hybrid="#4daf4a", LS="black")

label_model = dict(
    grz=r"$grz$",
    white=r"$\mathrm{White}$",
    White=r"$\mathrm{White}$",
    hybrid=r"$\mathrm{Hybrid}$",
    Hybrid=r"$\mathrm{Hybrid}$",
    LS=r"$\mathrm{LS}$",
)


def score_hist(dataset, models, idx=[], snr_lim=2.5):
    """
    histogram of score distribution
    """
    if len(idx) == 0:
        idx = np.array([True] * len(dataset.ds))
    y_true = dataset.y_true[idx]
    ds = dataset.ds[idx]

    bins = snr_lim - np.arange(3) / 2 - 0.25

    _, ax = plt.subplots(
        3,
        len(models),
        figsize=(2.5 * len(models), 6.5),
        sharey="row",
        sharex=True,
        constrained_layout=True,
    )

    for j, b in enumerate(bins):
        if j == 0:
            arg = ds.snr > 10 ** (b - 0.25)  # highest S/N
        else:
            arg = np.abs(np.log10(ds.snr) - b) < 0.25  # binned S/N
        print(f"Star: {(y_true & arg).sum()}; Galaxy: {(~y_true & arg).sum()}")

        for k, p in enumerate([dataset.pred[model][idx] for model in models]):
            ax[j, k].hist(
                p[y_true & arg],
                histtype="step",
                label="S (log(S/N):{:.1f}-{:.1f})".format(b - 0.25, b + 0.25),
                color=color_s,
                bins=50,
                range=([0, 1]),
                log=True,
            )
            ax[j, k].hist(
                p[~y_true & arg],
                histtype="step",
                label="G (log(S/N):{:.1f}-{:.1f})".format(b - 0.25, b + 0.25),
                color=color_g,
                bins=50,
                range=([0, 1]),
                log=True,
            )
            ax[j, k].tick_params(labelsize=15)
            ax[j, k].tick_params(axis="y", which="minor")
            if k == 0:
                ax[j, k].legend(prop={"size": 10})
            if j == 0:
                ax[j, k].set_title(label_model.get(models[k], models[k]))
            elif j == 2:
                ax[j, k].set_xlabel("score")

    plt.xlim(-0.05, 1.05)
    plt.ylim(1, None)
    plt.show()


def ROC_cv(
    train_set,
    models=[],
    idx_train=None,
    bins_by="SNR",
    binsize=None,
    snr_lim=2.5,
    mag_lim=19.5,
    cv_idx=None,
):
    """
    ROC curve for training set in the cross-validation
    """

    if idx_train is None:
        idx_train = np.array([True] * len(train_set.ds))

    from sklearn.metrics import roc_curve

    if bins_by == None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        y_true = train_set.y_true.astype(bool)[idx_train]

        try:
            # LS classifier
            LS_star = train_set.ds.type.astype(str)[idx_train] == "PSF"

            ax.scatter(
                ((~y_true) & LS_star).sum() / (~y_true).sum(),
                (y_true & LS_star).sum() / y_true.sum(),
                marker="*",
                edgecolor="k",
                facecolor="none",
                s=250,
                zorder=100,
            )
        except:
            print("No LS classifier")

        # CV results
        for j, model in enumerate(models):
            fpr, tpr, _ = roc_curve(y_true, train_set.pred[model][idx_train])
            ax.plot(fpr, tpr, label=label_model.get(model, model), color=color_model.get(model, "black"))
            if cv_idx:
                for l in range(len(cv_idx)):
                    eval_idx = np.array([False] * len(train_set.ds))
                    eval_idx[cv_idx[l]] = True
                    y_true = np.array(train_set.y_true)[idx_train & eval_idx]
                    fpr, tpr, _ = roc_curve(y_true, train_set.pred[model][idx_train & eval_idx])
                    ax.plot(fpr, tpr, color=color_model.get(model, "black"), lw=0.25)

        ax.set_xlim(6.5e-4, 4.5e-1)
        ax.set_xscale("log")

        ax.set_xlabel(r"$\mathrm{FPR}$")
        ax.set_ylabel(r"$\mathrm{TPR}$")

        ax.axvline(0.005, color="0.8", linestyle="-.", linewidth=5, zorder=-10)
        # ax[0].set_ylim(0.32, 1.01)

        ax.legend(prop={"size": 15}, loc=4)
        return ax

    if bins_by == "SNR":
        bins = snr_lim - np.arange(4) / 2
        if binsize == None:
            binsize = 0.5
    elif bins_by == "mag":
        bins = np.arange(4) + mag_lim
        if binsize == None:
            binsize = 1

    fig, ax = plt.subplots(
        1,
        len(bins),
        figsize=(14, 4),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )
    ax = ax.ravel()
    for k, b in enumerate(bins):
        # print the number of stars and galaxies in each bin
        if bins_by == "SNR":
            arg = np.abs(np.log10(train_set.ds.snr) - b) < binsize / 2
            y_true = np.array(train_set.y_true)[idx_train & arg]
            print(
                "Training set, log(S/N) = {:.1f} - {:.1f}: S = {:.0f}, G = {:.0f}".format(
                    b - binsize / 2,
                    b + binsize / 2,
                    y_true.sum(),
                    (~y_true).sum(),
                ),
            )
            ax[k].set_title(
                "${:.1f}".format(b + binsize / 2) + " > \log(\mathrm{S/N}) > " + "{:.1f}$".format(b - binsize / 2),
                fontsize=22.5,
            )
        elif bins_by == "mag":
            arg = np.abs(train_set.ds.mag_r.astype(float) - b) < binsize / 2
            y_true = np.array(train_set.y_true)[idx_train & arg]
            print(
                "Training set, mag = {:.0f} - {:.0f}: S = {:.0f}, G = {:.0f}".format(
                    b - binsize / 2,
                    b + binsize / 2,
                    y_true.sum(),
                    (~y_true).sum(),
                ),
            )
            ax[k].set_title(
                "${:.0f}".format(b - binsize / 2) + " < r\ [\mathrm{mag}] < " + "{:.0f}$".format(b + binsize / 2),
                fontsize=22.5,
            )
        try:
            # LS classifier
            LS_star = np.array(train_set.ds.type)[idx_train][arg] == "PSF"

            ax[k].scatter(
                ((~y_true) & LS_star).sum() / (~y_true).sum(),
                (y_true & LS_star).sum() / y_true.sum(),
                marker="*",
                edgecolor="k",
                facecolor="none",
                s=250,
                zorder=100,
            )
        except:
            print("No LS classifier")

        # CV results
        for j, model in enumerate(models):
            y_true = np.array(train_set.y_true)[idx_train & arg]
            fpr, tpr, _ = roc_curve(y_true, train_set.pred[model][idx_train & arg])
            ax[k].plot(fpr, tpr, label=label_model.get(model, model), color=color_model.get(model, "black"), lw=1)
            if cv_idx:
                for l in range(len(cv_idx)):
                    eval_idx = np.array([False] * len(train_set.ds))
                    eval_idx[cv_idx[l]] = True
                    y_true = np.array(train_set.y_true)[idx_train & arg & eval_idx]
                    fpr, tpr, _ = roc_curve(
                        y_true,
                        train_set.pred[model][idx_train & arg & eval_idx],
                    )
                    ax[k].plot(fpr, tpr, color=color_model.get(model, "black"), lw=0.1)

    ax[0].set_xlim(6.5e-4, 4.5e-1)
    ax[0].set_xscale("log")

    ax[0].set_ylabel(r"$\mathrm{TPR}$")

    for a in ax:
        a.axvline(0.005, color="0.8", linestyle="-.", linewidth=5, zorder=-10)
        a.set_xlabel(r"$\mathrm{FPR}$")
    # ax[0].set_ylim(0.32, 1.01)

    ax[0].legend(prop={"size": 15}, loc=4)
    # plt.show()
    return ax


def ROC_train_test(
    train_set,
    test_set=None,
    models=[],
    idx_train=[],
    idx_test=[],
    bins_by="SNR",
    binsize=None,
    snr_lim=2.5,
    mag_lim=19.5,
):
    """
    ROC curve for training and test set
    """
    if not bins_by in ["SNR", "mag"]:
        raise IndexError("Modes not supported")

    if len(idx_train) == 0:
        idx_train = np.array([True] * len(train_set.ds))
    if (len(idx_test) == 0) & (test_set != None):
        idx_test = np.array([True] * len(test_set.ds))

    from sklearn.metrics import roc_curve

    if bins_by == "SNR":
        bins = snr_lim - np.arange(4) / 2
        if binsize == None:
            binsize = 0.5
    elif bins_by == "mag":
        bins = np.arange(4) + mag_lim
        if binsize == None:
            binsize = 1

    fig, ax = plt.subplots(
        1,
        len(bins),
        figsize=(14, 4),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )
    ax = ax.ravel()

    for k, b in enumerate(bins):
        if bins_by == "SNR":
            log_snr = np.log10(train_set.ds.snr)
            arg = np.abs(log_snr - b) < binsize / 2
            y_true = train_set.y_true.values[idx_train & arg]
            print(
                "Training set, log(S/N) = {:.1f} - {:.1f}: S = {:.0f}, G = {:.0f}".format(
                    b - binsize / 2,
                    b + binsize / 2,
                    y_true.sum(),
                    (~y_true).sum(),
                ),
            )
            ax[k].set_title(
                "${:.1f}".format(b + binsize / 2) + " > \log(\mathrm{S/N}) > " + "{:.1f}$".format(b - binsize / 2),
                fontsize=22.5,
            )
        elif bins_by == "mag":
            mag = np.array(train_set.ds.mag_r)
            arg = np.abs(mag - b) < binsize / 2
            y_true = train_set.y_true.values[idx_train & arg]
            print(
                "Training set, mag = {:.0f} - {:.0f}: S = {:.0f}, G = {:.0f}".format(
                    b - binsize / 2,
                    b + binsize / 2,
                    y_true.sum(),
                    (~y_true).sum(),
                ),
            )
            ax[k].set_title(
                "${:.0f}".format(b - binsize / 2) + " < r\ [\mathrm{mag}] < " + "{:.0f}$".format(b + binsize / 2),
                fontsize=22.5,
            )

        for j, model in enumerate(models):
            fpr, tpr, _ = roc_curve(y_true, train_set.pred[model][idx_train][arg])
            ax[k].plot(fpr, tpr, label=label_model.get(model, model), color=color_model.get(model, "black"))
        try:
            # LS classifier
            LS_star = np.array(train_set.ds.type)[idx_train][arg] == "PSF"

            ax[k].scatter(
                ((~y_true) & LS_star).sum() / (~y_true).sum(),
                (y_true & LS_star).sum() / y_true.sum(),
                marker="*",
                edgecolor="k",
                facecolor="none",
                s=250,
                zorder=100,
            )
        except:
            print("No LS classifier")
    if not test_set == None:
        for k, b in enumerate(bins):
            if bins_by == "SNR":
                log_snr = np.log10(test_set.ds[idx_test].snr)
                arg = np.abs(log_snr - b) < binsize / 2
                y_true = np.array(test_set.y_true)[idx_test][arg]
                print(
                    "Test set, log(S/N) = {:.1f} - {:.1f}: S = {:.0f}, G = {:.0f}".format(
                        b - binsize / 2,
                        b + binsize / 2,
                        y_true.sum(),
                        (~y_true).sum(),
                    ),
                )
            elif bins_by == "mag":
                mag = test_set.ds[idx_test].mag_r
                arg = np.abs(mag - b) < binsize / 2
                y_true = np.array(test_set.y_true)[idx_test][arg]
                print(
                    "Test set, mag = {:.0f} - {:.0f}: S = {:.0f}, G = {:.0f}".format(
                        b - binsize / 2,
                        b + binsize / 2,
                        y_true.sum(),
                        (~y_true).sum(),
                    ),
                )

            for j, model in enumerate(models):
                fpr, tpr, _ = roc_curve(y_true, test_set.pred[model][idx_test][arg])
                ax[k].plot(fpr, tpr, linestyle="--", color=color_model.get(model, "black"))
            try:
                # LS classifier
                LS_star = np.array(test_set.ds.type)[idx_test][arg] == "PSF"

                ax[k].scatter(
                    ((~y_true) & LS_star).sum() / (~y_true).sum(),
                    (y_true & LS_star).sum() / y_true.sum(),
                    marker="*",
                    edgecolor="k",
                    facecolor="none",
                    linestyle="--",
                    s=250,
                    zorder=100,
                )
            except:
                print("No LS classifier")

    ax[0].set_xlim(6.5e-4, 4.5e-1)
    ax[0].set_xscale("log")

    ax[0].set_ylabel(r"$\mathrm{TPR}$")

    for a in ax:
        a.axvline(0.005, color="0.8", linestyle="-.", linewidth=5, zorder=-10)
        a.set_xlabel(r"$\mathrm{FPR}$")
    # ax[0].set_ylim(0.32, 1.01)

    ax[0].legend(prop={"size": 15}, loc=4)
    # plt.show()
    return ax
