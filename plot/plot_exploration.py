import io

import matplotlib.tri as tri
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def to_numpy(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    array = data.reshape((int(h), int(w), -1))
    return array


def contour(results: pd.DataFrame, criteria: str):
    x = np.log10(results["learning_rate"])
    y = np.log10(results["weight_decay"])
    z = results[criteria]
    xi = np.linspace(-6, -2, 100)
    yi = np.linspace(-5, 1, 100)
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)
    fig, ax1 = plt.subplots()
    ax1.contour(xi, yi, zi, levels=14, linewidths=0.5, colors="k")
    cntr1 = ax1.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")
    fig.colorbar(cntr1, ax=ax1)
    ax1.plot(x, y, "ko", ms=3)
    ax1.set(xlim=(-6, -2), ylim=(-5, 0))
    ax1.set_xlabel("Learning rate ($10^x$)")
    ax1.set_ylabel("Weight decay ($10^y$)")
    ax1.set_title(f"{criteria} exploration")
    fig.tight_layout()
    return to_numpy(fig)
