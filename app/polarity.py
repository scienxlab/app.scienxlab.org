#-*- coding: utf-8 -*-
import io
from collections import namedtuple
import base64

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap


matplotlib.use('Agg')

def get_colour(cmap, frac):
    """
    Decide whether to make white or black labels.
    """
    cmap = plt.get_cmap(cmap)
    return 'k' if (np.mean(cmap(frac)[:3]) > 0.5) else 'w'


def rotate_phase(w, phi, degrees=False):
    """
    Performs a phase rotation of wavelet or wavelet bank using:
    The analytic signal can be written in the form S(t) = A(t)exp(j*theta(t))
    where A(t) = magnitude(hilbert(w(t))) and theta(t) = angle(hilbert(w(t))
    then a constant phase rotation phi would produce the analytic signal
    S(t) = A(t)exp(j*(theta(t) + phi)). To get the non analytic signal
    we take real(S(t)) == A(t)cos(theta(t) + phi)
    == A(t)(cos(theta(t))cos(phi) - sin(theta(t))sin(phi)) <= trig idenity
    == w(t)cos(phi) - h(t)sin(phi)
    A = w(t)Cos(phi) - h(t)Sin(phi)
    Where w(t) is the wavelet and h(t) is its Hilbert transform.
    Args:
        w (ndarray): The wavelet vector, can be a 2D wavelet bank.
        phi (float): The phase rotation angle (in radians) to apply.
        degrees (bool): If phi is in degrees not radians.
    Returns:
        The phase rotated signal (or bank of signals).
    """
    if degrees:
        phi = phi * np.pi / 180.0
    a = scipy.signal.hilbert(w, axis=0)
    w = (np.real(a) * np.cos(phi) - np.imag(a) * np.sin(phi))
    return w


def ricker(duration, dt, f, return_t=False):
    """
    FROM BRUGES https://github.com/agile-geoscience/bruges/blob/master/bruges/filters/wavelets.py

    Also known as the mexican hat wavelet, models the function:

    .. math::
        A =  (1 - 2 \pi^2 f^2 t^2) e^{-\pi^2 f^2 t^2}
    If you pass a 1D array of frequencies, you get a wavelet bank in return.
    Args:
        duration (float): The length in seconds of the wavelet.
        dt (float): The sample interval in seconds (often one of  0.001, 0.002,
            or 0.004).
        f (ndarray): Centre frequency of the wavelet in Hz. If a sequence is
            passed, you will get a 2D array in return, one row per frequency.
        return_t (bool): If True, then the function returns a tuple of
            wavelet, time-basis, where time is the range from -duration/2 to
            duration/2 in steps of dt.
    Returns:
        ndarray. Ricker wavelet(s) with centre frequency f sampled on t.
    .. plot::
        plt.plot(bruges.filters.ricker(.5, 0.002, 40))

    """
    f = np.asanyarray(f).reshape(-1, 1)
    t = np.arange(-duration/2, duration/2, dt)
    pft2 = (np.pi * f * t)**2
    w = np.squeeze((1 - (2 * pft2)) * np.exp(-pft2))

    if return_t:
        RickerWavelet = namedtuple('RickerWavelet', ['amplitude', 'time'])
        return RickerWavelet(w, t)
    else:
        return w


def make_synthetic(size=256, top=0.4, base=0.6, value=1, freq=25, phase=0):
    """Make a synthetic. Return the wavelet, the model, the RC, and the synthetic.
    """
    v = np.ones(size) - value

    v[int(top*size):int(base*size)] = value
    rc = np.diff(v)

    w = ricker(0.256, 0.001, freq)

    if phase != 0:
        w = rotate_phase(w, phase, degrees=True)

    syn = np.convolve(rc, w, mode='same')
    return w, v, rc, syn


def _make_synthetic(imps=(0, 1, 0), thicks=(4, 2, 4), freq=25, phase=0, noise=0):

    v = np.ones(256) * imps[-1]

    stops = np.array([0] + list(np.cumsum(thicks)))
    stops = stops / stops[-1]

    for imp, (top, bot) in zip(imps, zip(stops, stops[1:])):
        v[int(top*256):int(bot*256)] = imp

    rc = np.diff(v)

    if noise:
        rcn = rc + np.random.choice([1, -1], size=rc.size) * np.random.power(0.33, size=rc.size) * 0.1
    else:
        rcn = rc

    w = ricker(0.256, 0.001, freq)

    if phase != 0:
        w = rotate_phase(w, phase, degrees=True)

    syn = np.convolve(rcn, w, mode='same')

    pos = 256 * (stops[:-1] + stops[1:]) / 2

    return w, v, rc, rcn, syn, pos


def polarity_cartoon(layer='hard',
                     polarity='normal',
                     freq='med',
                     phase=0,
                     style='vd',
                     cmap=None,
                     fmt='png',
                     ):
    """
    Plot a polarity cartoon.
    """
    freqs = {'vhi': 60, 'hi': 30, 'med': 15, 'lo': 7.5,
             'vhigh': 60, 'high': 30, 'medium': 15, 'low': 7.5,
             'mod': 15, 'mid': 15}
    backgr = 'soft' if layer == 'hard' else 'hard'
    value = 1 if layer == 'hard' else 0
    size, top, base = 256, 0.4, 0.6

    _, v, _, syn = make_synthetic(size, top, base, value, freq=freqs[freq], phase=phase)

    if polarity.lower() not in ['normal', 'seg', 'usa', 'us', 'canada']:
        syn *= -1

    if style == 'ramp':
        # cbar is a ramp.
        cbar = np.linspace(1, -1, size).reshape(-1, 1)
    else:
        # cbar is the synthetic.
        cbar = syn.reshape(-1, 1)

    gs = {'width_ratios':[2,2,2,1]}
    fig, axs = plt.subplots(ncols=4,
                            figsize=(6, 4),
                            gridspec_kw=gs,
                            facecolor='w', sharey=True,
                            )

    # Plot the earth model.
    ax = axs[0]
    cmap_ = 'Greys'
    ax.imshow(v.reshape(-1, 1), aspect='auto', cmap=cmap_, vmin=-1.5, vmax=1.5)
    ax.axhline(top*size, c='w', lw=4)
    ax.axhline(base*size, c='w', lw=4)
    ax.axvline(0.55, c='w', lw=6)  # Total hack to adjust RHS
    ax.text(0, size/4.75, backgr, ha='center', va='center', color=get_colour(cmap_, (1-value)*256), size=25)
    ax.text(0, size/2+0.75, layer, ha='center', va='center', color=get_colour(cmap_, (value)*256), size=25)

    # Plot the impedance diagram.
    ax = axs[1]
    cmap_ = 'Greys'
    ax.imshow(v.reshape(-1, 1), aspect='auto', cmap=cmap_, vmin=0, vmax=2)
    ax.axvline(-0.5, c=(0.58, 0.58, 0.58), lw=50)
    ax.text(0.45, 2*size/8, 'imp', ha='right', va='center', color='k', size=25)
    #ax.text(0.15, size/8, "→", ha='center', va='center', color='k', size=30, fontproperties=fontprop)
    ax.annotate("", xy=(0.33, size/8), xytext=(0, size/8), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    # Plot the waveform.
    ax = axs[2]
    y = np.arange(syn.size)
    ax.plot(syn, y, 'k')
    ax.fill_betweenx(y, syn, 0, where=syn>0, color='k')
    ax.invert_yaxis()
    ax.text(0.65, size/8, '+', ha='center', va='center', size=30)
    ax.text(-0.65, size/8, '–', ha='center', va='center', size=40)

    # Plot the colourbar.
    ax = axs[3]
    if cmap == "petrel":
        colors = np.loadtxt('./petrel.tsv')
        cmap = LinearSegmentedColormap.from_list('_', colors)
    elif cmap == "petrel_r":
        colors = np.loadtxt('./petrel.tsv')
        cmap = LinearSegmentedColormap.from_list('_', colors[::-1])
    else:
        cmap = cmap or 'gray'
    frac = 1/8
    top_col = get_colour(cmap, frac)
    bot_col = get_colour(cmap, 7*frac)
    ax.imshow(cbar, cmap=cmap, aspect='auto')
    if style == 'ramp':
        ax.text(0, frac*size, '+', ha='center', va='center', color=top_col, size=30)
        ax.text(0, 7*frac*size, '–', ha='center', va='center', color=bot_col, size=40)

    # Make final adjustments to subplots and figure.
    for ax in axs:
        ax.set_axis_off()

    plt.subplots_adjust(left=0.1)

    # Make data to hand back.
    if fmt == 'svg':
        im = io.StringIO()
        plt.savefig(im, format='svg')
        im.seek(0)
        txt = im.getvalue()
    else:
        im = io.BytesIO()
        plt.savefig(im, format='png')
        im.seek(0)
        if fmt == 'raw':
            return im
        txt = base64.b64encode(im.getvalue()).decode('utf8')

    return txt


def plot_synthetic(imps=(0, 1, 0),
                   thicks=(4, 2, 4),
                   polarity='normal',
                   noise=0,
                   freq='med',
                   phase=0,
                   cmap=None,
                   ):
    """
    Plot a synthetic.
    """
    freqs = {'vhi': 60, 'hi': 30, 'med': 15, 'lo': 7.5,
             'vhigh': 60, 'high': 30, 'medium': 15, 'low': 7.5,
             'mod': 15, 'mid': 15}

    if isinstance(freq, str):
        freq = freqs[freq]

    w, v, rc, rcn, syn, pos = _make_synthetic(imps, thicks, freq=freq, phase=phase, noise=noise)

    if polarity.lower() not in ['normal', 'seg', 'usa', 'us', 'canada']:
        syn *= -1
        w *= -1

    if noise:
        gs = {'width_ratios':[2,2,2,2,1,2,1]}
        fig, axs = plt.subplots(ncols=7,
                                figsize=(14, 5),
                                gridspec_kw=gs,
                                facecolor='w',
                                sharey=True,
                                )
    else:
        gs = {'width_ratios':[2,2,2,1,2,1]}
        fig, axs = plt.subplots(ncols=6,
                                figsize=(12, 5),
                                gridspec_kw=gs,
                                facecolor='w',
                                sharey=True,
                                )

    size = 256
    x_ = 0
    y_ = size + 20

    # Earth model.
    ax = axs[0]
    cmap_ = 'viridis_r'
    ax.imshow(v.reshape(-1, 1), aspect='auto', cmap=cmap_, vmin=-1.5, vmax=1.5)
    ax.axvline(0.55, c='w', lw=6)  # Total hack to adjust RHS
    ax.text(x_, y_, 'm', ha='center', va='center', color='k', size=25)

    for imp, p in zip(imps, pos):
        ax.text(0, p, str(imp), ha='center', va='center', color=get_colour(cmap_, imp/max(imps)), size=25)

    # Impedance.
    ax = axs[1]
    cmap_ = 'Greys'
    y = np.arange(v.size)
    ax.plot(v, y, c='b', lw=2)
    ax.text(0.5, -20/256, 'i', ha='center', va='center', color='k', size=25, transform=ax.transAxes)
    ax.text(0.5, 12/256, '−  +', ha='center', va='center', color='gray', size=25, transform=ax.transAxes)

    # Reflection coefficients.
    ax = axs[2]
    y = np.arange(rc.size)
    ax.plot(rc, y, 'k')
    ax.text(x_, y_, 'r', ha='center', va='center', color='k', size=25)
    ax.text(x_, y_ - 32, '−  +', ha='center', va='center', color='gray', size=25)

    if noise:
        # Reflection coefficients + noise.
        ax = axs[3]
        y = np.arange(rcn.size)
        ax.plot(rcn, y, 'k')
        ax.text(x_, y_, 'r + n', ha='center', va='center', color='k', size=25)
        ax.text(x_, y_ - 32, '−  +', ha='center', va='center', color='gray', size=25)
        w_, s_, v_ = 4, 5, 6
    else:
        w_, s_, v_ = 3, 4, 5

    # Wavelet.
    ax = axs[w_]
    y = np.arange(w.size)
    ax.plot(w, y, 'k')
    ax.fill_betweenx(y, w, 0, where=w>0, color='k')
    ax.text(x_, y_, 'w', ha='center', va='center', color='k', size=25)
    ax.text(x_, y_ - 32, '−  +', ha='center', va='center', color='gray', size=25)

    # Synthetic wiggle.
    ax = axs[s_]
    y = np.arange(syn.size)
    ax.plot(syn, y, 'k')
    ax.fill_betweenx(y, syn, 0, where=syn>0, color='k')
    ax.text(x_, y_, 's', ha='center', va='center', color='k', size=25)
    ax.text(x_, y_ - 32, '−  +', ha='center', va='center', color='gray', size=25)

    # Synthetic VD.
    ax = axs[v_]
    cmap = cmap or 'gray'
    ax.imshow(syn.reshape(-1, 1), cmap=cmap, aspect='auto')
    ax.text(x_, y_, 's', ha='center', va='center', color='k', size=25)

    for ax in axs:
        ax.set_axis_off()

    plt.subplots_adjust(left=0.1)
    plt.subplots_adjust(bottom=0.15)

    # Make bytes to hand back.
    im = io.BytesIO()
    plt.savefig(im, format='png')
    im.seek(0)

    return im
