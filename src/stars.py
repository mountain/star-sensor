# -*- coding: utf-8 -*-

import datetime

from pytz import utc

from skyfield.api import Star, Loader
from skyfield.data import hipparcos
from skyfield.api import Topos


load = Loader('.', expire=False)
with load.open('./hip_main.dat.gz') as f:
    df = hipparcos.load_dataframe(f)

planets = load('de421.bsp')
sun = planets['sun']
venus = planets['venus']
mercury = planets['mercury']
earth = planets['earth']
moon = planets['moon']
mars = planets['mars']

north_polar = Star(ra_hours=0, dec_degrees=90.0)
south_polar = Star(ra_hours=0, dec_degrees=-90.0)
march_equinox = Star(ra_hours=0, dec_degrees=0.0)
septb_equinox = Star(ra_hours=12, dec_degrees=0.0)
june_solstice = Star(ra_hours=6, dec_degrees=0.0)
decm_solstice = Star(ra_hours=18, dec_degrees=0.0)

ts = load.timescale()

filtered = df[df['magnitude'] < 6.0]
filtered = filtered[filtered['ra_degrees'].notnull()]
filtered = filtered[filtered['dec_degrees'].notnull()]
bright_stars_count = len(filtered)
print('bright stars count:', bright_stars_count)
bright_stars = Star.from_dataframe(filtered)


def get_place(lat, lng, tms):
    return (earth + Topos(latitude=lat, longitude=lng, latitude_degrees=True, longitude_degrees=True)).at(tms)


def get_time(tms=None):
    if tms is None:
        tms = ts.now()
    else:
        dt = datetime.datetime.utcfromtimestamp(tms)
        dt = dt.replace(tzinfo=utc)
        tms = ts.utc(dt)

    return tms


def get_time_by_tt(tt):
    return ts.tt_jd(tt)
