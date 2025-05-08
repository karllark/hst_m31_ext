Code for HST M31 UV extinction project
======================================

Routines for HST M31 UV extinction curves program.
PI: G. Clayton

In Development!
---------------

Active development.
Everything changing.
Use at your own risk.

Contributors
------------
Karl Gordon

License
-------

This code is licensed under a 3-clause BSD style license (see the
``LICENSE`` file).

Extinction Curves
-----------------

Extinction curves created by running the `fitstars` bash script.  This fits the
STIS spectra and the available photometry (mostly HST) using `utils/fit_model.py`.
The foreground extinction is inlcuded in the fitting using the MW velocity integrated
HI columns converted to A(V) using the high-latitude N(HI)/A(V) ratio and a R(V) = 3.1
is assumed.

Most of the extinction curves are created relative to the F475W band as V band photometry
does not exist for them.   For these curves, utils/convert_to_av_norm.py is used to 
convert the curves to E(lambda - V) from E(lambda - F475W) using the fit R(V) value and 
the dust_extinction G23 R(V) dependent extinction model.

Figures
------- 

2. UV-NIR spectra/photometry of all stars: plotting/plot_spec_stack.py

3. UV-NIR extinction curves: plotting/plot_uvoptir_ext.py

Tables
------

3. Stellar parameters: utils/create_param_table.py

4. Column parameters: utils/create_param_table.py

5. FM90 parameters: utils/create_param_table.py