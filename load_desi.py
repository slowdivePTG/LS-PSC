import pandas as pd
import numpy as np

def load_desi(verbose=False, footprint="all"):

    desi_clean = pd.read_parquet("./Data/ls_dr10_desi_edr_clean.parquet")

    LRG_bit = 2**0
    ELG_bit = 2**1
    QSO_bit = 2**2
    BGS_bit = 2**60
    MWS_bit = 2**61
    SCND_bit = 2**62

    zpix_cat = pd.read_csv("./Data/DESI_survey.csv")

    survey_info = pd.merge(desi_clean, zpix_cat, left_on="targetid", right_on="TARGETID")

    is_LRG = (
        (survey_info.SV1_DESI_TARGET & LRG_bit != 0)
        | (survey_info.SV2_DESI_TARGET & LRG_bit != 0)
        | (survey_info.SV3_DESI_TARGET & LRG_bit != 0)
    )
    is_ELG = (
        (survey_info.SV1_DESI_TARGET & ELG_bit != 0)
        | (survey_info.SV2_DESI_TARGET & ELG_bit != 0)
        | (survey_info.SV3_DESI_TARGET & ELG_bit != 0)
    )
    is_QSO = (
        (survey_info.SV1_DESI_TARGET & QSO_bit != 0)
        | (survey_info.SV2_DESI_TARGET & QSO_bit != 0)
        | (survey_info.SV3_DESI_TARGET & QSO_bit != 0)
    )
    is_BGS = (
        (survey_info.SV1_DESI_TARGET & BGS_bit != 0)
        | (survey_info.SV2_DESI_TARGET & BGS_bit != 0)
        | (survey_info.SV3_DESI_TARGET & BGS_bit != 0)
    )
    is_MWS = (
        (survey_info.SV1_DESI_TARGET & MWS_bit != 0)
        | (survey_info.SV2_DESI_TARGET & MWS_bit != 0)
        | (survey_info.SV3_DESI_TARGET & MWS_bit != 0)
    )
    is_SCND = (
        (survey_info.SV1_DESI_TARGET & SCND_bit != 0)
        | (survey_info.SV2_DESI_TARGET & SCND_bit != 0)
        | (survey_info.SV3_DESI_TARGET & SCND_bit != 0)
    )

    survey_info["survey_type"] = None
    survey_info.loc[is_SCND, "survey_type"] = "SCND"
    survey_info.loc[is_BGS, "survey_type"] = "BGS"
    survey_info.loc[is_MWS, "survey_type"] = "MWS"
    survey_info.loc[is_LRG, "survey_type"] = "LRG"
    survey_info.loc[is_ELG, "survey_type"] = "ELG"
    survey_info.loc[is_QSO, "survey_type"] = "QSO"

    for survey_type in ["SCND", "BGS", "MWS", "LRG", "ELG", "QSO"]:
        print(survey_type, (survey_info.survey_type == survey_type).sum())
    print(
        "LRG + red BGS (r - z > 1 mag):",
        (is_LRG | (is_BGS & (survey_info.r_z > 1))).sum(),
    )

    sga2020 = pd.read_csv("./Data/sga2020_ls_dr10.csv")
    ls_DUP = pd.read_csv("./Data/ls_dr10_DUP.csv")

    desi_clean["sga"] = desi_clean.ls_id.isin(
        sga2020.ls_id
    )  # | (desi_clean.fitbits & 2**9 != 0)
    desi_clean["frozen"] = desi_clean.fitbits & 2**4 != 0
    desi_clean["dup"] = desi_clean.ls_id.isin(ls_DUP.ls_id)

    ls_release = desi_clean.ls_id.values >> 40
    desi_clean["footprint"] = np.where(ls_release >= 10000, "south", "north")

    for flt in "griz":
        flt_masked = 1
        for k in range(8):
            flt_masked &= desi_clean[f"apflux_ivar_{flt}_{k+1}"] <= 0
        desi_clean[f"apflux_masked_{flt}"] = flt_masked

    desi_south = ls_release >= 10000
    desi_north = ls_release < 10000

    desi_qso = desi_clean.targetid.isin(
        survey_info[survey_info.survey_type == "QSO"].targetid
    )  # targeted as QSOs

    if verbose:
        print(f"Total: {len(desi_clean)} Stars/galaxies")
        print(f"Not targeted as a QSO: {(~desi_qso).sum()} Stars/galaxies")
        print()
        print(f"ls_dr10 south: {(desi_south).sum()} Stars/galaxies")
        print(f"    Not targeted as a QSO: {(desi_south & ~desi_qso).sum()} Stars/galaxies")
        print(f"    Not targeted as a QSO + LS4 footprint: {((desi_clean.dec < 20) & ~desi_qso).sum()} Stars/galaxies")
        print()
        print(f"ls_dr10 north: {desi_north.sum()} Stars/galaxies")
        print(f"    Not targeted as a QSO: {(desi_north & ~desi_qso).sum()} Stars/galaxies")

    if footprint == "south":
        return desi_clean[desi_south & ~desi_qso]
    elif footprint == "north":
        return desi_clean[desi_north & ~desi_qso]
    else:
        return desi_clean[~desi_qso]

if __name__ == "__main__":
    desi_clean = load_desi(verbose=True, footprint="south")