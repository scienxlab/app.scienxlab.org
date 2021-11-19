# -*- coding: utf-8 -*-
import io
import urllib
import requests

from flask import Flask
from flask import make_response
from flask import request, redirect, jsonify, render_template

from polarity import polarity_cartoon, plot_synthetic
from bruges import get_bruges


VERBOTEN = ['jet', 'jet_r', 'hsv', 'hsv_r', 'rainbow', 'rainbow_r', 'gist_rainbow', 'gist_rainbow_r', 'nipy_spectral', 'nipy_spectral_r']


# Error handling.
class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


# The app.
application = Flask(__name__)


# Routes and handlers.
@application.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@application.route('/')
def main():
    return render_template('index.html', title='Home')


@application.route('/about')
def about():
    return render_template('about.html', title='About')


@application.route('/polarity')
def polarity():
    layer = str(request.args.get('layer') or 'hard').lower()
    polarity = str(request.args.get('polarity') or 'normal').lower()
    freq = str(request.args.get('freq') or 'med').lower()
    phase = int(request.args.get('phase') or 0)
    style = str(request.args.get('style') or 'syn').lower()
    cmap = str(request.args.get('cmap') or 'RdBu')
    fmt = 'png'

    if cmap in VERBOTEN:
        return redirect("https://agilescientific.com/blog/2017/12/14/no-more-rainbows", code=302)

    # This returns a BytesIO containing the image.
    txt = polarity_cartoon(layer, polarity, freq, phase, style, cmap, fmt)

    data = {'image': txt,
            'format': fmt,
            'layer': layer,
            'polarity': polarity,
            'freq': freq,
            'phase': phase,
            'style': style,
            'cmap': cmap,
            }

    return render_template('polarity.html', title='Polarity', data=data)


@application.route('/polarity.png')
def polarity_png():
    layer = str(request.args.get('layer') or 'hard')
    polarity = str(request.args.get('polarity') or 'normal')
    freq = str(request.args.get('freq') or 'med')
    phase = int(request.args.get('phase') or 0)
    style = str(request.args.get('style') or 'syn')
    cmap = str(request.args.get('cmap') or 'RdBu')
    fmt = 'raw'

    if cmap in VERBOTEN:
        return redirect("https://agilescientific.com/blog/2017/12/14/no-more-rainbows", code=302)

    # This returns a base64-encoded BytesIO containing the image.
    im = polarity_cartoon(layer, polarity, freq, phase, style, cmap, fmt)

    response = make_response(im.getvalue())
    response.mimetype = 'image/png'
    return response


@application.route('/polarity.svg')
def polarity_svg():
    layer = str(request.args.get('layer') or 'hard')
    polarity = str(request.args.get('polarity') or 'normal')
    freq = str(request.args.get('freq') or 'med')
    phase = int(request.args.get('phase') or 0)
    style = str(request.args.get('style') or 'syn')
    cmap = str(request.args.get('cmap') or 'RdBu')
    fmt = 'svg'

    if cmap in VERBOTEN:
        return redirect("https://agilescientific.com/blog/2017/12/14/no-more-rainbows", code=302)

    # This returns a StringIO containing the SVG data.
    im = polarity_cartoon(layer, polarity, freq, phase, style, cmap, fmt)

    response = make_response(im)
    response.mimetype = 'image/svg+xml'
    return response


@application.route('/synthetic')
def synthetic():
    return render_template('synthetic.html', title='Synthetic')


@application.route('/synthetic.png')
def synthetic_api():
    imps = request.args.get('imps')
    if imps is not None:
        imps = [float(i) for i in imps.split(',')]
    else:
        imps = (0, 1, 0)
    thicks = request.args.get('thicks')
    if thicks is not None:
        thicks = [float(i) for i in thicks.split(',')]
    else:
        thicks = (0.4, 0.2, 0.4)
    polarity = str(request.args.get('polarity') or 'normal')
    noise = int(request.args.get('noise') or 0)
    freq = str(request.args.get('freq') or 'med')
    phase = int(request.args.get('phase') or 0)
    cmap = str(request.args.get('cmap') or 'RdBu')

    if cmap in VERBOTEN:
        return redirect("https://agilescientific.com/blog/2017/12/14/no-more-rainbows", code=302)

    # This returns a BytesIO containing the image.
    im = plot_synthetic(imps, thicks, polarity, noise, freq, phase, cmap)

    response = make_response(im.getvalue())
    response.mimetype = 'image/png'
    return response

##############################################################################
#
# Bruges logo server
#

@application.route('/bruges')
def bruges():
    return render_template('bruges.html', title='Bruges')


@application.route('/bruges.png')
def bruges_png():

    p = float(request.args.get('p') or 0.5)
    n = int(request.args.get('n') or 1)
    style = str(request.args.get('style') or '')

    text = get_bruges(p, n)
    text = urllib.parse.quote_plus(text)

    base_url = "https://chart.googleapis.com/chart"

    # Transparent backgrounds are not possible, unforch.
    # https://developers.google.com/chart/image/docs/chart_params?hl=en#background-fills-chf-[all-charts]
    if style.lower() == 'bubble':
        q = "?chst=d_bubble_text_small&chld=bb|{}|14AFCA|000000"
        query = q.format(text)
    else:
        q = "?chst=d_text_outline&chld=14AFCA|24|h|325396|b|{}"
        query = q.format(text)

    url = base_url + query

    r = requests.get(url)
    b = io.BytesIO(r.content)

    response = make_response(b.getvalue())
    response.mimetype = 'image/png'
    return response


@application.route('/bruges.json')
def bruges_json():

    p = float(request.args.get('p') or 0.5)
    n = int(request.args.get('n') or 1)

    text = get_bruges(p, n)
    dictionary = {'result': text,
                  'p': p,
                  'n': n,
                  }

    return jsonify(dictionary)


@application.route('/bruges.txt')
def bruges_text():

    p = float(request.args.get('p') or 0.5)
    n = int(request.args.get('n') or 1)

    text = get_bruges(p, n)
    return text


if __name__ == "__main__":

    application.run()
