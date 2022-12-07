import yt

yt.enable_parallelism()

ds = yt.load("RD0035/RD0035")
v, c = ds.find_max(("gas", "density"))
print(v, c)
p = yt.ProjectionPlot(ds, "x", ("gas", "density"))
p.save()

