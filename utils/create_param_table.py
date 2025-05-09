import os
import numpy as np
from astropy.table import QTable

from measure_extinction.extdata import ExtData
from measure_extinction.extdata import conv55toAv, conv55toRv, conv55toEbv


def prettyname(name):
    return name.split("_")[1]


if __name__ == "__main__":

    for ctype in [""]:

        otab = QTable(
            # fmt: off
            names=("name", 
                   "AV", "AV_unc", "RV", "RV_unc", "NHI", "NHI_unc",
                   "C2", "C2_unc", "B3", "B3_unc", "C4", "C4_unc",
                   "x0", "x0_unc", "gamma", "gamma_unc"),
            dtype=("S", 
                   "f", "f", "f", "f", "f", "f",
                   "f", "f", "f", "f", "f", "f",
                   "f", "f", "f", "f"),
            # fmt:on
        )

        otab_lat = QTable(
            # fmt: off
            names=("Name", 
                   r"$A(V)$", r"$R(V)$", r"$log[N(HI)]$", r"$A(V)_\mathrm{MW}$", r"$log[N(HI)]_\mathrm{MW}$"),
            dtype=("S", "S", "S", "S", "S", "S")
            # fmt:on
        )


        otab_lat2 = QTable(
            # fmt: off
            names=("Name", r"$C_2$", r"$B_3$", r"$C_4$", r"$x_o$", r"$\gamma$"),
            dtype=("S", "S", "S", "S", "S", "S")
            # fmt:on
        )

        otab_lat3 = QTable(
            # fmt: off
            names=("Name", r"$\log(T_\mathrm{eff})$", r"$\log(g)$", r"$\log(Z)$",
                   r"$v_\mathrm{vturb}$"),
            dtype=("S", "S", "S", "S", "S")
            # fmt:on
        )


        colnames = ["Av", "Rv", "logHI_exgal", "fore_Av", "logHI_MW"]
        fm90names = ["C2", "B3", "C4", "xo", "gamma"]
        stellnames = ["logTeff", "logg", "logZ", "vturb"]
        
        files = ["all"]
        tags = ["All"]
        for cfile, ctag in zip(files, tags):

            filename = "data/m31_exts.dat"

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
                cpath = "exts"
                cfile = f"{cpath}/{cname}_mefit_ext.fits"
                edata = ExtData(filename=cfile)
                fdata = edata.fit_params["MCMC"]

                rdata = []
                rdata_lat = []
                rdata_lat2 = []
                rdata_lat3 = []
                rdata.append(cname)
                pcname = prettyname(cname)
                print(pcname)
                rdata_lat.append(pcname)
                rdata_lat2.append(pcname)
                rdata_lat3.append(pcname)

                for ccol in colnames:
                    idx, = np.where(fdata["name"] == ccol)
                    val = fdata[idx]["value"].data[0]
                    unc = fdata[idx]["unc"].data[0]
                    if ccol == "logHI_exgal":
                        sval = val
                        sunc = unc
                        val = 10 ** val
                        unc *= np.log(10.) * val
                        if unc / val > 0.5:
                            val = 0.0
                            unc = 0.0
                            rdata_lat.append(fr"\nodata")
                        else:
                            rdata_lat.append(fr"${sval:.2f} \pm {sunc:.2f}$")
                    elif ccol in ["fore_Av", "logHI_MW"]:
                        rdata_lat.append(fr"${val:.2f}$")
                    else:
                        rdata_lat.append(fr"${val:.2f} \pm {unc:.2f}$")
                    if ccol not in ["fore_Av", "logHI_MW"]:
                        rdata.append(val)
                        rdata.append(unc)

                for ccol in fm90names:
                    idx, = np.where(fdata["name"] == ccol)
                    val = fdata[idx]["value"].data[0]
                    unc = fdata[idx]["unc"].data[0]
                    rdata.append(val)
                    rdata.append(unc)
                    if ccol == "xo":
                        tstr = fr"${val:.3f} \pm {unc:.3f}$"
                    else:
                        tstr = fr"${val:.2f} \pm {unc:.2f}$"
                    rdata_lat2.append(tstr)
                otab_lat2.add_row(rdata_lat2)

                for ccol in stellnames:
                    idx, = np.where(fdata["name"] == ccol)
                    val = fdata[idx]["value"].data[0]
                    unc = fdata[idx]["unc"].data[0]
                    if ccol in ["logTeff", "logg", "logZ"]:
                        tstr = fr"${val:.3f} \pm {unc:.3f}$"
                    else:
                        tstr = fr"${val:.2f} \pm {unc:.2f}$"
                    rdata_lat3.append(tstr)
                otab_lat3.add_row(rdata_lat3)

                otab.add_row(rdata)
                otab_lat.add_row(rdata_lat)

        basestr = "C25_m31"
        otab.write(
            f"tables/{basestr}{ctype}_ensemble_params.dat", format="ascii.ipac", overwrite=True
        )

        otab_lat.write(
            f"tables/{basestr}{ctype}_ensemble_dust_params.tex",
            format="aastex",
            col_align="lccccccc",
            latexdict={
                "caption": r"Column Parameters \label{tab_ext_col_param}",
            },
            overwrite=True,
        )

        otab_lat2.write(
            f"tables/{basestr}{ctype}_ensemble_fm90_params.tex",
            format="aastex",
            col_align="lcccccc",
            latexdict={
                "caption": r"FM90 Parameters \label{tab_ext_fm90_params}",
            },
            overwrite=True,
        )

        otab_lat3.write(
            f"tables/{basestr}{ctype}_ensemble_stell_params.tex",
            format="aastex",
            col_align="lcccc",
            latexdict={
                "caption": r"Stellar Parameters \label{tab_ext_stell_params}",
            },
            overwrite=True,
        )