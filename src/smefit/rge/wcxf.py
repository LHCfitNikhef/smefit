# -*- coding: utf-8 -*-
# Dictionary translating from the smefit basis to the Warsaw basis in the WCxf
import numpy as np

# Values at MZ
alpha_s = 0.118
mw = 80.387
mz = 91.1876
gs = np.sqrt(4 * np.pi * alpha_s)
sw = np.sqrt(1 - mw**2 / mz**2)
cw = np.sqrt(1 - sw**2)


# This creates a dictionary to go from SMEFiT basis to Warsaw.
# In particular, for each operator, it tells you which Wilson coefficients
# need to be switched on in the Warsaw basis and the corresponding values.
# These are basically the equations between operators.
# O_{SMEFiT} = sum_i value_i * O_{Warsaw, i}

# Note on index ordering for same field 4-fermion operators (uu, dd, qq1, qq3, etc):
# these take the "smaller index first pair" ("_1122") as the standard slot
# and overwrites the swapped ordering ("_2211") from it. So for these operators,
# the first pair must have the lowest index first. If we wrote "uu_2211" instead,
# "uu_1122" would be silently zeroed out by wilson's symmetrisation.

wcxf_translate = {
    # Bosonic
    "OWWW": {"wc": ["W"], "value": [-1.0]},
    "OpBox": {"wc": ["phiBox"]},
    "OpD": {"wc": ["phiD"]},
    "OpWB": {"wc": ["phiWB"]},
    "OpG": {"wc": ["phiG"]},
    "OpW": {"wc": ["phiW"]},
    "OpB": {"wc": ["phiB"]},
    "Op": {"wc": ["phi"]},
    # Dipoles
    "OtG": {"wc": ["uG_33"], "value": ["-gs"]},
    "OtW": {"wc": ["uB_33", "uW_33"], "value": [-cw / sw, -1.0]},
    "OtZ": {"wc": ["uB_33"], "value": [1.0 / sw]},
    # Quark Currents
    "O3pq": {"wc": ["phiq1_11", "phiq1_22", "phiq3_11", "phiq3_22"]},
    "OpqMi": {"wc": ["phiq1_11", "phiq1_22"]},
    "O3pQ3": {"wc": ["phiq1_33", "phiq3_33"]},
    "OpQM": {"wc": ["phiq1_33"]},
    "Opt": {"wc": ["phiu_33"]},
    "Opui": {"wc": ["phiu_11", "phiu_22"]},
    "Opdi": {"wc": ["phid_11", "phid_22", "phid_33"]},
    # Lepton Currents
    "Opl1": {"wc": ["phil1_11"]},
    "Opl2": {"wc": ["phil1_22"]},
    "Opl3": {"wc": ["phil1_33"]},
    "O3pl1": {"wc": ["phil3_11"]},
    "O3pl2": {"wc": ["phil3_22"]},
    "O3pl3": {"wc": ["phil3_33"]},
    "Ope": {"wc": ["phie_11"]},
    "Opmu": {"wc": ["phie_22"]},
    "Opta": {"wc": ["phie_33"]},
    # Yukawas
    "Otp": {"wc": ["uphi_33"]},
    "Ocp": {"wc": ["uphi_22"]},
    "Obp": {"wc": ["dphi_33"]},
    "Otap": {"wc": ["ephi_33"]},
    "Omup": {"wc": ["ephi_22"]},
    # 2L2H quark operators
    "O81qq": {
        "wc": ["qq1_1331", "qq1_2332", "qq1_1133", "qq1_2233", "qq3_1331", "qq3_2332"],
        "value": [1.0 / 4.0, 1.0 / 4.0, -1.0 / 6.0, -1.0 / 6.0, 1.0 / 4.0, 1.0 / 4.0],
    },
    "O11qq": {"wc": ["qq1_1133", "qq1_2233"]},
    "O83qq": {
        "wc": ["qq1_1331", "qq1_2332", "qq3_1133", "qq3_2233", "qq3_1331", "qq3_2332"],
        "value": [3.0 / 4.0, 3.0 / 4.0, -1.0 / 6.0, -1.0 / 6.0, -1.0 / 4.0, -1.0 / 4.0],
    },
    "O13qq": {"wc": ["qq3_1133", "qq3_2233"]},
    "O8qt": {"wc": ["qu8_1133", "qu8_2233"]},
    "O1qt": {"wc": ["qu1_1133", "qu1_2233"]},
    "O8ut": {
        "wc": ["uu_1133", "uu_2233", "uu_1331", "uu_2332"],
        "value": [-1.0 / 6.0, -1.0 / 6.0, 1.0 / 2.0, 1.0 / 2.0],
    },
    "O1ut": {"wc": ["uu_1133", "uu_2233"]},
    "O8qu": {"wc": ["qu8_3311", "qu8_3322"]},
    "O1qu": {"wc": ["qu1_3311", "qu1_3322"]},
    "O8dt": {"wc": ["ud8_3311", "ud8_3322", "ud8_3333"]},
    "O1dt": {"wc": ["ud1_3311", "ud1_3322", "ud1_3333"]},
    "O8qd": {"wc": ["qd8_3311", "qd8_3322", "qd8_3333"]},
    "O1qd": {"wc": ["qd1_3311", "qd1_3322", "qd1_3333"]},
    # 4H quark operators
    "OQQ1": {"wc": ["qq1_3333"], "value": [1.0 / 2.0]},
    "OQQ8": {"wc": ["qq1_3333", "qq3_3333"], "value": [1.0 / 24.0, 1.0 / 8.0]},
    "OQt1": {"wc": ["qu1_3333"]},
    "OQt8": {"wc": ["qu8_3333"]},
    "Ott1": {"wc": ["uu_3333"]},
    # 4 light quarks
    "Oqq1": {"wc": ["qq1_1122", "qq1_1111", "qq1_2222"]},
    "Oqq1p": {"wc": ["qq1_1221", "qq1_1111", "qq1_2222"]},
    "Oqq3": {"wc": ["qq3_1122", "qq3_1111", "qq3_2222"]},
    "Oqq3p": {"wc": ["qq3_1221", "qq3_1111", "qq3_2222"]},
    "Ouu": {"wc": ["uu_1122", "uu_1111", "uu_2222"]},
    "Ouup": {"wc": ["uu_1221", "uu_1111", "uu_2222"]},
    "Odd": {
        "wc": ["dd_1122", "dd_1133", "dd_2233", "dd_1111", "dd_2222", "dd_3333"],
    },
    "Oddp": {
        "wc": ["dd_1221", "dd_1331", "dd_2332", "dd_1111", "dd_2222", "dd_3333"],
    },
    "Oqu1": {"wc": ["qu1_1111", "qu1_1122", "qu1_2211", "qu1_2222"]},
    "Oud1": {
        "wc": ["ud1_1111", "ud1_1122", "ud1_1133", "ud1_2211", "ud1_2222", "ud1_2233"],
    },
    "Oqd1": {
        "wc": ["qd1_1111", "qd1_1122", "qd1_1133", "qd1_2211", "qd1_2222", "qd1_2233"],
    },
    # 4 leptons
    # left-left leptons
    "Oll1221": {
        "wc": ["ll_1221"],
        "value": [2.0],
    },  # Notice the factor of 1/2. See table 28 of arXiv:2012.11343
    "Oll1331": {
        "wc": ["ll_1331"],
        "value": [2.0],
    },
    "Oll2332": {"wc": ["ll_2332"], "value": [2.0]},
    "Oll1111": {"wc": ["ll_1111"]},
    "Oll1122": {"wc": ["ll_1122"], "value": [2.0]},
    "Oll1133": {"wc": ["ll_1133"], "value": [2.0]},
    "Oll2233": {"wc": ["ll_2233"], "value": [2.0]},
    "Oll2222": {"wc": ["ll_2222"]},
    "Oll3333": {"wc": ["ll_3333"]},
    # left-right leptons
    "Ole1111": {"wc": ["le_1111"]},
    "Ole2222": {"wc": ["le_2222"]},
    "Ole3333": {"wc": ["le_3333"]},
    "Ole1133": {"wc": ["le_1133"]},
    "Ole1122": {"wc": ["le_1122"]},
    "Ole2233": {"wc": ["le_2233"]},
    "Ole3322": {"wc": ["le_3322"]},
    "Ole3311": {"wc": ["le_3311"]},
    "Ole2211": {"wc": ["le_2211"]},
    # right-right leptons
    "Oee1111": {"wc": ["ee_1111"]},
    "Oee2222": {"wc": ["ee_2222"]},
    "Oee3333": {"wc": ["ee_3333"]},
    "Oee1122": {
        "wc": ["ee_1122"],
        "value": [4.0],
    },  # Factor of 4 to agree with SMEFTsim general convention.
    "Oee1133": {"wc": ["ee_1133"], "value": [4.0]},
    "Oee2233": {"wc": ["ee_2233"], "value": [4.0]},
    # 2 quark 2 lepton operators
    "Oeu": {"wc": ["eu_1111", "eu_1122"]},
    "Omuu": {"wc": ["eu_2211", "eu_2222"]},
    "Otau": {"wc": ["eu_3311", "eu_3322"]},
    "Oed": {"wc": ["ed_1111", "ed_1122"]},
    "Omud": {"wc": ["ed_2211", "ed_2222"]},
    "Otad": {"wc": ["ed_3311", "ed_3322"]},
    "Oeb": {"wc": ["ed_1133"]},
    "Omub": {"wc": ["ed_2233"]},
    "Otab": {"wc": ["ed_3333"]},
    "Otl1": {"wc": ["lu_1133"]},
    "Otl2": {"wc": ["lu_2233"]},
    "Otl3": {"wc": ["lu_3333"]},
    "Ote": {"wc": ["eu_1133"]},
    "Otmu": {"wc": ["eu_2233"]},
    "Otta": {"wc": ["eu_3333"]},
    "Oql13": {"wc": ["lq1_1111", "lq1_1122", "lq3_1111", "lq3_1122"]},
    "Oql23": {"wc": ["lq1_2211", "lq1_2222", "lq3_2211", "lq3_2222"]},
    "Oql33": {"wc": ["lq1_3311", "lq1_3322", "lq3_3311", "lq3_3322"]},
    "Oql1M": {"wc": ["lq1_1111", "lq1_1122"]},
    "Oql2M": {"wc": ["lq1_2211", "lq1_2222"]},
    "Oql3M": {"wc": ["lq1_3311", "lq1_3322"]},
    "OQl13": {"wc": ["lq1_1133", "lq3_1133"]},
    "OQl23": {"wc": ["lq1_2233", "lq3_2233"]},
    "OQl33": {"wc": ["lq1_3333", "lq3_3333"]},
    "OQl1M": {"wc": ["lq1_1133"]},
    "OQl2M": {"wc": ["lq1_2233"]},
    "OQl3M": {"wc": ["lq1_3333"]},
    "Ol1u": {"wc": ["lu_1111", "lu_1122"]},
    "Ol2u": {"wc": ["lu_2211", "lu_2222"]},
    "Ol3u": {"wc": ["lu_3311", "lu_3322"]},
    "Ol1d": {"wc": ["ld_1111", "ld_1122"]},
    "Ol2d": {"wc": ["ld_2211", "ld_2222"]},
    "Ol3d": {"wc": ["ld_3311", "ld_3322"]},
    "Ol1b": {"wc": ["ld_1133"]},
    "Ol2b": {"wc": ["ld_2233"]},
    "Ol3b": {"wc": ["ld_3333"]},
    "Oqe": {"wc": ["qe_1111", "qe_2211"]},
    "Oqmu": {"wc": ["qe_1122", "qe_2222"]},
    "Oqta": {"wc": ["qe_1133", "qe_2233"]},
    "OQe": {"wc": ["qe_3311"]},
    "OQmu": {"wc": ["qe_3322"]},
    "OQta": {"wc": ["qe_3333"]},
}

# This creates a dictionary to go from Warsaw to SMEFiT.
# In particular, given a point in the coefficient space of the Warsaw basis,
# it tells you how to combine them to get the coefficients in the SMEFiT basis.
# These are basically the equations between Wilson coefficients.
# C_{SMEFiT} = sum_i coeff_i * C_{Warsaw, i}
# Note that the flavour structure is assumed to hold and therefore the values
# are inferred only from the 11 components.
inverse_wcxf_translate = {
    # Bosonic
    "OWWW": {"wc": ["W"], "coeff": [-1.0]},
    "OpBox": {"wc": ["phiBox"]},
    "OpD": {"wc": ["phiD"]},
    "OpWB": {"wc": ["phiWB"]},
    "OpG": {"wc": ["phiG"]},
    "OpW": {"wc": ["phiW"]},
    "OpB": {"wc": ["phiB"]},
    "Op": {"wc": ["phi"]},
    # Dipoles
    "OtG": {"wc": ["uG_33"], "coeff": ["-1/gs"]},
    "OtW": {"wc": ["uW_33"], "coeff": [-1.0]},
    "OtZ": {"wc": ["uB_33", "uW_33"], "coeff": [sw, -cw]},
    # Quark Currents
    "O3pq": {"wc": ["phiq3_11"]},
    "OpqMi": {"wc": ["phiq1_11", "phiq3_11"], "coeff": [1.0, -1.0]},
    "O3pQ3": {"wc": ["phiq3_33"]},
    "OpQM": {"wc": ["phiq1_33", "phiq3_33"], "coeff": [1.0, -1.0]},
    "Opt": {"wc": ["phiu_33"]},
    "Opui": {"wc": ["phiu_11"]},
    "Opdi": {"wc": ["phid_11"]},
    # Lepton Currents
    "Opl1": {"wc": ["phil1_11"]},
    "Opl2": {"wc": ["phil1_22"]},
    "Opl3": {"wc": ["phil1_33"]},
    "O3pl1": {"wc": ["phil3_11"]},
    "O3pl2": {"wc": ["phil3_22"]},
    "O3pl3": {"wc": ["phil3_33"]},
    "Ope": {"wc": ["phie_11"]},
    "Opmu": {"wc": ["phie_22"]},
    "Opta": {"wc": ["phie_33"]},
    # Yukawas
    "Otp": {"wc": ["uphi_33"]},
    "Ocp": {"wc": ["uphi_22"]},
    "Obp": {"wc": ["dphi_33"]},
    "Otap": {"wc": ["ephi_33"]},
    "Omup": {"wc": ["ephi_22"]},
    # 2L2H quark operators
    "O81qq": {"wc": ["qq1_1331", "qq3_1331"], "coeff": [1.0, 3.0]},
    "O11qq": {
        "wc": ["qq1_1133", "qq1_1331", "qq3_1331"],
        "coeff": [1.0, 1.0 / 6.0, 1.0 / 2.0],
    },
    "O83qq": {
        "wc": ["qq1_1331", "qq3_1331"],
        "coeff": [1.0, -1.0],
    },
    "O13qq": {
        "wc": ["qq3_1133", "qq1_1331", "qq3_1331"],
        "coeff": [1.0, 1.0 / 6.0, -1.0 / 6.0],
    },
    "O8qt": {"wc": ["qu8_1133"]},
    "O1qt": {"wc": ["qu1_1133"]},
    "O8ut": {"wc": ["uu_1331"], "coeff": [2.0]},
    "O1ut": {"wc": ["uu_1133", "uu_1331"], "coeff": [1.0, 1.0 / 3.0]},
    "O8qu": {"wc": ["qu8_3311"]},
    "O1qu": {"wc": ["qu1_3311"]},
    "O8dt": {"wc": ["ud8_3311"]},
    "O1dt": {"wc": ["ud1_3311"]},
    "O8qd": {"wc": ["qd8_3311"]},
    "O1qd": {"wc": ["qd1_3311"]},
    # 4H quark operators
    "OQQ1": {"wc": ["qq1_3333", "qq3_3333"], "coeff": [2.0, -2.0 / 3.0]},
    "OQQ8": {"wc": ["qq3_3333"], "coeff": [8.0]},
    "OQt1": {"wc": ["qu1_3333"]},
    "OQt8": {"wc": ["qu8_3333"]},
    "Ott1": {"wc": ["uu_3333"]},
    # 4 light quarks
    "Oqq1": {"wc": ["qq1_1122"]},
    "Oqq1p": {"wc": ["qq1_1221"]},
    "Oqq3": {"wc": ["qq3_1122"]},
    "Oqq3p": {"wc": ["qq3_1221"]},
    "Ouu": {"wc": ["uu_1122"]},
    "Ouup": {"wc": ["uu_1221"]},
    "Odd": {"wc": ["dd_1122"]},
    "Oddp": {"wc": ["dd_1221"]},
    "Oqu1": {"wc": ["qu1_1122"]},
    "Oud1": {"wc": ["ud1_1122"]},
    "Oqd1": {"wc": ["qd1_1122"]},
    # 4 leptons
    # left-left leptons
    "Oll1221": {
        "wc": ["ll_1221"],
        "coeff": [1.0 / 2.0],
    },  # Notice the factor of 1/2. See table 28 of arXiv:2012.11343
    "Oll1331": {
        "wc": ["ll_1331"],
        "coeff": [1.0 / 2.0],
    },
    "Oll1111": {"wc": ["ll_1111"]},
    "Oll1122": {
        "wc": ["ll_1122"],
        "coeff": [1.0 / 2.0],
    },
    "Oll2332": {
        "wc": ["ll_2332"],
        "coeff": [1.0 / 2.0],
    },  # Notice the factor of 1/2. See table 28 of arXiv:2012.11343
    "Oll1133": {"wc": ["ll_1133"], "coeff": [1.0 / 2.0]},
    "Oll2233": {"wc": ["ll_2233"], "coeff": [1.0 / 2.0]},
    "Oll2222": {"wc": ["ll_2222"]},
    "Oll3333": {"wc": ["ll_3333"]},
    # left-right leptons
    "Ole1111": {"wc": ["le_1111"]},
    "Ole2222": {"wc": ["le_2222"]},
    "Ole3333": {"wc": ["le_3333"]},
    "Ole1133": {"wc": ["le_1133"]},
    "Ole1122": {"wc": ["le_1122"]},
    "Ole2233": {"wc": ["le_2233"]},
    "Ole3322": {"wc": ["le_3322"]},
    "Ole3311": {"wc": ["le_3311"]},
    "Ole2211": {"wc": ["le_2211"]},
    # right-right leptons
    "Oee1111": {"wc": ["ee_1111"]},
    "Oee2222": {"wc": ["ee_2222"]},
    "Oee3333": {"wc": ["ee_3333"]},
    "Oee1122": {
        "wc": ["ee_1122"],
        "coeff": [1.0 / 4.0],
    },  # Factor 1/4 to agree with SMEFTsim general convention.
    "Oee1133": {"wc": ["ee_1133"], "coeff": [1.0 / 4.0]},
    "Oee2233": {"wc": ["ee_2233"], "coeff": [1.0 / 4.0]},
    # 2 quark 2 lepton operators
    "Oeu": {"wc": ["eu_1111"]},
    "Omuu": {"wc": ["eu_2211"]},
    "Otau": {"wc": ["eu_3311"]},
    "Oed": {"wc": ["ed_1111"]},
    "Omud": {"wc": ["ed_2211"]},
    "Otad": {"wc": ["ed_3311"]},
    "Oeb": {"wc": ["ed_1133"]},
    "Omub": {"wc": ["ed_2233"]},
    "Otab": {"wc": ["ed_3333"]},
    "Otl1": {"wc": ["lu_1133"]},
    "Otl2": {"wc": ["lu_2233"]},
    "Otl3": {"wc": ["lu_3333"]},
    "Ote": {"wc": ["eu_1133"]},
    "Otmu": {"wc": ["eu_2233"]},
    "Otta": {"wc": ["eu_3333"]},
    "Oql13": {"wc": ["lq3_1111"]},
    "Oql23": {"wc": ["lq3_2211"]},
    "Oql33": {"wc": ["lq3_3311"]},
    "Oql1M": {"wc": ["lq1_1111", "lq3_1111"], "coeff": [1.0, -1.0]},
    "Oql2M": {"wc": ["lq1_2211", "lq3_2211"], "coeff": [1.0, -1.0]},
    "Oql3M": {"wc": ["lq1_3311", "lq3_3311"], "coeff": [1.0, -1.0]},
    "OQl13": {"wc": ["lq3_1133"]},
    "OQl23": {"wc": ["lq3_2233"]},
    "OQl33": {"wc": ["lq3_3333"]},
    "OQl1M": {"wc": ["lq1_1133", "lq3_1133"], "coeff": [1.0, -1.0]},
    "OQl2M": {"wc": ["lq1_2233", "lq3_2233"], "coeff": [1.0, -1.0]},
    "OQl3M": {"wc": ["lq1_3333", "lq3_3333"], "coeff": [1.0, -1.0]},
    "Ol1u": {"wc": ["lu_1111"]},
    "Ol2u": {"wc": ["lu_2211"]},
    "Ol3u": {"wc": ["lu_3311"]},
    "Ol1d": {"wc": ["ld_1111"]},
    "Ol2d": {"wc": ["ld_2211"]},
    "Ol3d": {"wc": ["ld_3311"]},
    "Ol1b": {"wc": ["ld_1133"]},
    "Ol2b": {"wc": ["ld_2233"]},
    "Ol3b": {"wc": ["ld_3333"]},
    "Oqe": {"wc": ["qe_1111"]},
    "Oqmu": {"wc": ["qe_1122"]},
    "Oqta": {"wc": ["qe_1133"]},
    "OQe": {"wc": ["qe_3311"]},
    "OQmu": {"wc": ["qe_3322"]},
    "OQta": {"wc": ["qe_3333"]},
}
