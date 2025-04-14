import glob

import numpy as np
from astropy.table import QTable

# import astropy.units as u

from measure_extinction.extdata import ExtData

def prettyname(name):
    return name


if __name__ == "__main__":

    for ctype in ["", "_forecor"]:

        otab = QTable(
            # fmt: off
            names=("name", "EBV", "EBV_unc", "AV", "AV_unc", "RV", "RV_unc", "NHI", "NHI_unc",
                   "C1", "C1_unc", "C2", "C2_unc", "B3", "B3_unc", "C4", "C4_unc",
                   "x0", "x0_unc", "gamma", "gamma_unc"),
            dtype=("S", "f", "f", "f", "f", "f", "f", "f", "f", "f", "f", "f",
                   "f", "f", "f", "f", "f", "f", "f", "f", "f"),
            # fmt:on
        )

        otab_lat = QTable(
            # fmt: off
            names=("Name", 
                   "$E(B-V)$", "$A(V)$", "$R(V)$", "$N_\mathrm{SMC}(HI)$",
                   "$C_1$", "$C_2$", "$B_3$", "$C_4$"),
            dtype=("S", "S", "S", "S", "S", "S", "S", "S", "S")
            # fmt:on
        )


        otab_lat2 = QTable(
            # fmt: off
            names=("Name", "$x_o$", "$\gamma$"),
            dtype=("S", "S", "S")
            # fmt:on
        )

        colnames = ["EBV", "AV", "RV", "NHI"]
        fm90names = ["C1", "C2", "B3", "C4", "XO", "GAMMA"]

        files = ["highebv", "highebv_bumps", "highebv_flat", "lowebv", "aves"]
        tags = [r"$E(B-V)_\mathrm{SMC} \geq 0.1, Steep with Weak/Absent Bump",
                r"$E(B-V)_\mathrm{SMC} \geq 0.1, Significant Bump",
                r"$E(B-V)_\mathrm{SMC} \geq 0.1, Flat",
                r"$E(B-V)_\mathrm{SMC} < 0.1, Weak/Absent Bump",
                r"Averages"]
        
        files = ["all"]
        tages = ["All"]
        for cfile, ctag in zip(files, tags):

            filename = f"data/m31_good.dat"

            f = open(filename, "r")
            file_lines = list(f)
            starnames = []
            for line in file_lines:
                if (line.find("#") != 0) & (len(line) > 0):
                    name = line.rstrip()
                    starnames.append(name)
            # starnames = np.sort(starnames)

            for cname in starnames:
                ctype = ""
                cpath = "jan2025_ext_curve_fits_and_figs_forecor/forecor_ext_curve_FM90_fits/"
                cfile = f"{cpath}/{cname}_elvebv_ext_FM90.fits_ext_forecor_FM90_5000.fits"
                if cname in ["m31_e5", "m31_e8"]:
                    cfile = cfile.replace("_5000", "")

                edata = ExtData(filename=cfile)
                if "LOGHI" not in edata.columns:
                    #print(edata.columns)
                    edata.columns["NHI"] = (0.0, 0.0)
                    #edata.columns["AV"] = (0.0, 0.0)
                    #edata.columns["EBV"] = (0.0, 0.0)
                if "AV" not in edata.columns:
                    print("no AV")
                    exit()
                    av = edata.columns["EBV"][0] * edata.columns["RV"][0]
                    # fmt: off
                    av_unc = ((edata.columns["EBV"][0] / edata.columns["EBV"][0]) ** 2 
                            + (edata.columns["RV"][0] / edata.columns["RV"][0]) ** 2)
                    # fmt: on
                    av_unc = av * np.sqrt(av_unc)
                    edata.columns["AV"] = (av, av_unc)

                if "NHI" not in edata.columns:
                    hip = 10 ** (edata.columns["LOGHI"][0] + edata.columns["LOGHI"][1])
                    him = 10 ** (edata.columns["LOGHI"][0] - edata.columns["LOGHI"][1])
                    edata.columns["NHI"] = (10 ** edata.columns["LOGHI"][0], 0.5 * (hip - him))

                print(edata.columns)

                # get foreground subtraction plus / minus fits
                if ctype == "_forecor":
                    pedata = ExtData(filename=cfile.replace("forecor", "forecor_plus"))
                    medata = ExtData(filename=cfile.replace("forecor", "forecor_minus"))
                    if "HI" not in pedata.columns:
                        pedata.columns["NHI"] = (10 ** pedata.columns["LOGHI"][0], 0.0)
                        medata.columns["NHI"] = (10 ** medata.columns["LOGHI"][0], 0.0)

                rdata = []
                rdata_lat = []
                rdata_lat2 = []
                rdata.append(cname)
                pcname = prettyname(cname)
                print(cname, pcname)
                rdata_lat.append(pcname)
                rdata_lat2.append(pcname)
                for ccol in colnames:
                    val = edata.columns[ccol][0]
                    unc = edata.columns[ccol][1]
                    if ctype == "_forecor":
                        # fmt: off
                        unc = np.sqrt((unc ** 2) 
                                    + ((0.5 * np.absolute(medata.columns[ccol][0] - pedata.columns[ccol][0]))) ** 2)
                        # fmt: on
                    rdata.append(val)
                    rdata.append(unc)
                    if ccol == "NHI":
                        val /= 1e21
                        unc /= 1e21
                    rdata_lat.append(fr"${val:.2f} \pm {unc:.2f}$")

                for ccol in fm90names:
                    val = edata.fm90_p50_fit[ccol][0]
                    unc = edata.fm90_p50_fit[ccol][1]
                    if ctype == "_forecor":
                        # fmt: off
                        unc = np.sqrt((unc ** 2) 
                                    + ((0.5 * np.absolute(medata.fm90_p50_fit[ccol][0] - pedata.fm90_p50_fit[ccol][0]))) ** 2)
                        # fmt: on
                    rdata.append(val)
                    rdata.append(unc)
                    if ccol not in ["XO", "GAMMA"]:
                        rdata_lat.append(fr"${val:.2f} \pm {unc:.2f}$")
                    if ccol in ["XO", "GAMMA"]:
                        rdata_lat2.append(fr"${val:.2f} \pm {unc:.2f}$")

                otab.add_row(rdata)
                otab_lat.add_row(rdata_lat)
                otab_lat2.add_row(rdata_lat2)

        basestr = "C25_m31"
        otab.write(
            f"tables/{basestr}{ctype}_ensemble_params.dat", format="ascii.ipac", overwrite=True
        )

        otab_lat.write(
            f"tables/{basestr}{ctype}_ensemble_dust_params.tex",
            format="aastex",
            col_align="lcccc",
            latexdict={
                "caption": r"Extinction Parameters \label{tab_ext_col_param}",
            },
            overwrite=True,
        )

        otab_lat2.write(
            f"tables/{basestr}{ctype}_ensemble_bump_params.tex",
            format="aastex",
            col_align="lcc",
            latexdict={
                "caption": r"Detailed Bump Parameters \label{tab_ext_bump_params}",
            },
            overwrite=True,
        )