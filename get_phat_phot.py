
import math

import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
import pyvo as vo


def get_phat_single_star(radecstr, tap_service):
    """
    Get the PHAT photmoetry for the star closest to the
    coordinates specified.  Using the NOAO data lab.
    """

    # put into a coordinate object to handle the unit conversions
    coord = SkyCoord(radecstr, unit=(u.hourangle, u.degree), frame='icrs')

    query_str = """SELECT * FROM phat_v2.phot_mod
                WHERE CONTAINS(POINT('ICRS',ra,dec),
                CIRCLE('ICRS',%s,%s,0.00027))=1""" % (coord.ra.degree,
                                                      coord.dec.degree)

    # query_str = """SELECT * FROM phat_v2.phot_mod
    #            WHERE q3c_radial_query(ra,dec,{0},{1},{2})
    #            """.format(coord.ra.degree, coord.dec.degree, 1./60)

    # query_str = """SELECT *
    #       FROM phat_v2.phot_mod WHERE q3c_radial_query(ra,dec,11.27188291499695,41.697835806211046,0.016666666666666666)
    #       """

    print(query_str)
    exit()

    result = tap_service.run_sync(query_str)

    print(result)
    exit()

    tap_results = tap_service.run_async(query_str)
    # tap_results = tap_service.search(query_str)

    if len(tap_results) > 0:
        return (tap_results[0], len(tap_results))
    else:
        return (None, 0)


if __name__ == "__main__":

    # initialize the GAIA TAP service
    tap_service = vo.dal.TAPService("http://datalab.noao.edu/tap")
    # tap_service = vo.dal.SCSService("http://datalab.noao.edu/tap")
    # print(tap_service.description)

    # resultset = tap_service.search("SELECT TOP 1 * FROM phat_v2.phot_mod")
    # print(resultset)
    # exit()

    # get the list of starnames
    itablename = 'm31_stars_coords.dat'
    # itablename = 'data/mwext_small.dat'
    snames = Table.read(itablename,
                        format='ascii.commented_header', guess=False)
    print(snames)

    # query GAIA and create a table
    # ot = Table(names=('name', 'parallax', 'parallax_error', 'nfound',
    #                  'G_mag', 'G_flux', 'G_flux_error',
    #                  'AG', 'AG_lower', 'AG_upper'),
    #           dtype=('S15', 'f', 'f', 'i', 'f', 'f', 'f', 'f', 'f', 'f'))

    for sname, cra, cdec in zip(snames['name'], snames['ra'], snames['dec']):
        # print('trying ', sname)
        coord = SkyCoord('{} {}'.format(cra, cdec),
                         unit=(u.hourangle, u.degree), frame='icrs')
        print(sname, coord.ra.degree, coord.dec.degree)

        # get the GAIA info for one star
        # sres, nres = get_phat_single_star('{} {}'.format(cra, cdec),
        #                                  tap_service)
        #if nres == 0:
        #    print(sname, 'not found')
            # ot.add_row((sname, 0.0, 0.0, 0, 0.0, 0.0, 0.0))
        #else:
        #    print(sres)

            # ot.add_row((sname, sres['parallax'], sres['parallax_error'],
            #            nres,
            #            sres['phot_g_mean_mag'],
            #            sres['phot_g_mean_flux'],
            #            sres['phot_g_mean_flux_error'],
            #            sres['a_g_val'],
            #            sres['a_g_percentile_lower'],
            #            sres['a_g_percentile_upper']))

    # output the resulting table
    # otablename = itablename.replace('.dat', '_gaia.dat')
    # ot.write(otablename, format='ascii.commented_header', overwrite=True)
