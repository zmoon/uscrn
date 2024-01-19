===
API
===

Get data
--------

Download selected data for all sites and return as a :class:`pandas.DataFrame`.
In the returned :class:`~pandas.DataFrame`,
and those from `the readers <#read>`__ as well,
variable attributes such as long name and units are stored in :attr:`~pandas.DataFrame.attrs`.

.. autosummary::
   :toctree: api/

   uscrn.get_data

Convert to xarray
-----------------

Convert to :class:`xarray.Dataset`,
automatically adding a soil depth dimension if applicable.

.. autosummary::
   :toctree: api/

   uscrn.to_xarray


Read
----

Read data from a single file and return as a :class:`pandas.DataFrame`.

.. autosummary::
   :toctree: api/

   uscrn.read
   uscrn.read_daily
   uscrn.read_hourly
   uscrn.read_monthly
   uscrn.read_subhourly


Metadata
--------

Load site metadata as a :class:`pandas.DataFrame`.

.. autosummary::
   :toctree: api/

   uscrn.load_meta
