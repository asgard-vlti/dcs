#!/usr/bin/env python3
"""Rolling Baldr intensity-distribution monitor; camera access is read-only."""
import argparse
import json
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.stats import norm


def masks(shape, geometry):
    cx, cy, radius, gap, width = geometry
    y, x = np.indices(shape)
    r = np.hypot(x-cx, y-cy)
    return r <= radius, (r >= radius+gap) & (r <= radius+gap+width)


def fit_pixel(data, prior, mu, bins=50, maxiter=500):
    """Original histogram forward model, with bounded fits and convergence checks.

    mu=None floats mu on its canonical [0, pi] branch. The clear intensity is
    only a soft prior on A, as in post_analysis.py, not an optical calibration.
    """
    data = np.asarray(data, dtype=float)
    if not np.all(np.isfinite(data)) or prior <= 0 or np.std(data) <= 0:
        raise ValueError('Invalid or constant pixel data')
    # Work in clear-pupil units to make the objective independent of ADU scale.
    data = data / prior
    hist, edges = np.histogram(data, bins=bins, density=True)
    width = edges[1]-edges[0]
    phi = np.linspace(-4*np.pi, 4*np.pi, 2000)
    def pdf(params):
        a, b, sigma, noise = params[:4]
        angle = params[4] if mu is None else mu
        weights = norm.pdf(phi, 0, sigma) * (phi[1]-phi[0])
        model, _ = np.histogram(a+b*np.cos(phi+angle), bins=edges, weights=weights)
        model = gaussian_filter1d(model.astype(float), noise/width, mode='constant')
        area = model.sum()*width
        return model/area if area > 0 else np.zeros_like(hist)
    def objective(params):
        return np.mean((pdf(params)-hist)**2)*width**2 + .001*(params[0]-1)**2
    spread = np.std(data)
    start = [1., max(np.ptp(np.percentile(data, [5,95]))/2, .001), .5, max(spread*.1, .001)]
    bounds = [(1e-5, None), (1e-5, None), (.02, 3.), (1e-5, None)]
    if mu is None:
        start.append(np.pi/2)
        bounds.append((0., np.pi))
    result = minimize(objective, start, method='Nelder-Mead', bounds=bounds,
                      options={'maxiter': maxiter, 'xatol': 1e-3, 'fatol': 1e-8})
    a, b, sigma, noise = result.x[:4]
    if not result.success or b > a or sigma >= 2.99:
        raise ValueError('Unconverged or out-of-range fit')
    return (np.array([a*prior, b*prior, sigma, noise*prior]),
            float(result.x[4] if mu is None else mu),
            (edges[:-1]+edges[1:])/2*prior, pdf(result.x)/prior, hist/prior,
            float(result.fun))


def analyse(cube, clear, geometry, threshold, fixed_mu, wavelength_ratio):
    started = time.monotonic()
    pupil, annulus = masks(clear.shape, geometry)
    ann_values = clear[annulus & np.isfinite(clear)]
    if not ann_values.size:
        raise ValueError('Annulus has no finite pixels; adjust geometry')
    bias = float(np.median(ann_values))
    reference = clear.astype(float)-bias
    if not np.any(pupil) or np.nanmax(reference[pupil]) <= 0:
        raise ValueError('No illuminated pupil pixels')
    active = pupil & np.isfinite(reference) & (reference > threshold*np.nanmax(reference[pupil]))
    coords = np.argwhere(active)
    if not len(coords):
        raise ValueError('No active pixels')
    cube = cube.astype(float)-bias
    maps = np.full((5,*clear.shape), np.nan)
    mu_map = np.full(clear.shape, np.nan)
    residual = np.full(clear.shape, np.nan)
    profiles = {}
    for y, x in coords:
        try:
            parameters, mu, centres, model, histogram, quality = fit_pixel(
                cube[:,y,x], reference[y,x], fixed_mu,
                maxiter=900 if fixed_mu is None else 500)
            maps[:4,y,x] = parameters
            maps[4,y,x] = quality
            mu_map[y,x] = mu
            # RMS bin-probability residual, excluding the A-prior penalty.
            width = centres[1]-centres[0]
            residual[y,x] = np.sqrt(np.mean(((model-histogram)*width)**2))
            profiles[(int(y),int(x))] = (centres, model, histogram)
        except ValueError:
            pass
    good = np.isfinite(maps[2])
    count = int(good.sum())
    # Uniform area weighting of accepted pixels; missing pixels can bias this.
    proxy = float(np.exp(-np.mean(maps[2,good]**2)*wavelength_ratio**2)) if count else np.nan
    return dict(maps=maps, mu_map=mu_map, residual=residual, profiles=profiles,
                mu_mode="free per pixel" if fixed_mu is None else "fixed",
                bias=bias, active=active, count=count,
                proxy=proxy, seconds=time.monotonic()-started)


class Camera:
    def __init__(self, path):
        from xaosim.shmlib import shm
        if not Path(path).is_file():
            raise FileNotFoundError(f'Camera stream does not exist: {path}')
        self.stream = shm(str(path), nosem=True)
        self.last = None

    def read(self):
        counter = int(self.stream.get_counter())
        if counter == self.last:
            return None
        if self.last is not None and counter < self.last:
            raise RuntimeError('Camera counter reset; restart monitor')
        frame = self.stream.get_data().copy()
        if int(self.stream.get_counter()) != counter:
            return None
        self.last = counter
        if frame.ndim != 2:
            raise ValueError('Expected a 2D Baldr subframe')
        return counter, frame

    def close(self):
        # We attach to a producer-owned stream; never unlink its file.
        self.stream.close(erase_file=False)


class DemoCamera:
    def __init__(self):
        self.rng = np.random.default_rng(42)
        y, x = np.indices((32,32))
        self.pupil = np.hypot(x-15.5,y-15.5) <= 11
        self.clear = 100 + 1000*self.pupil.astype(float)
        self.counter = 0

    def read(self):
        self.counter += 1
        phase = self.rng.normal(0, .5, self.clear.shape)
        frame = self.clear + 600*self.pupil*np.cos(phase+1.1) + self.rng.normal(0,8,self.clear.shape)
        return self.counter, frame

    def close(self):
        pass


class JaxDemoCamera:
    """Small NumPy simulator demo; the live acquisition path never imports JAX."""
    def __init__(self, settings, sigma=.7):
        from baldrapp.common.baldr_core import get_zwfs_output_field
        from scipy.ndimage import gaussian_filter
        self.propagate = get_zwfs_output_field
        self.smooth = gaussian_filter
        self.settings = settings
        self.sigma = sigma
        self.rng = np.random.default_rng(932)
        y,x = np.indices((12,12))
        self.pupil = np.hypot(x-5.5,y-5.5) <= 4
        self.signal = 1000.*self.pupil
        self.counter = 0
        self.clear = settings['bias_adu']+(self.signal+settings['background_e'])/settings['gain']
        n = settings['clear_frames']
        self.clear += (self.rng.poisson((self.signal+settings['background_e'])*n)/n
                       -(self.signal+settings['background_e'])
                       +self.rng.normal(0,settings['read_noise_e']/np.sqrt(n),self.signal.shape))/settings['gain']

    def read(self):
        self.counter += 1
        phase = self.smooth(self.rng.normal(size=self.signal.shape),self.settings['correlation_px'])
        phase -= phase[self.pupil].mean()
        phase *= self.sigma/max(phase[self.pupil].std(),1e-9)
        field = self.propagate(phase,np.sqrt(self.signal),self.settings['theta'],
                              self.settings['mask_diameter'],fplane_pixels=self.settings['fft_size'])
        electrons = self.rng.poisson(np.abs(field)**2+self.settings['background_e'])
        electrons = electrons+self.rng.normal(0,self.settings['read_noise_e'],self.signal.shape)
        return self.counter, self.settings['bias_adu']+electrons/self.settings['gain']

    def close(self):
        pass


def jax_settings(args):
    return dict(model='full' if args.backend=='jax-full' else 'coherent',
        mask_diameter=args.mask_diameter,theta=args.theta,gain=args.gain,
        bias_adu=args.bias_adu,background_e=args.background_e,read_noise_e=args.read_noise_e,
        clear_frames=args.clear_frames,nodes=args.nodes,fft_size=args.fft_size,
        correlation_px=args.correlation_px,pixelwise=args.pixelwise,maxiter=args.maxiter,
        wfs_nm=args.wfs_nm)


def run(args):
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets
    from astropy.io import fits
    pg.setConfigOptions(imageAxisOrder='row-major')
    app = pg.mkQApp('Baldr phase monitor')

    class Window(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.is_jax = getattr(args,'backend','classic') != 'classic'
            self.epoch = 0
            self.last_submitted_counter = None
            self.last_frame_at = time.monotonic()
            self.clear_count = getattr(args,'clear_frames',None)
            self.setWindowTitle('Baldr rolling phase monitor — '+getattr(args,'backend','classic'))
            self.resize(1250,850)
            self.camera = (JaxDemoCamera(jax_settings(args),args.demo_sigma) if self.is_jax else DemoCamera()) if args.demo else Camera(args.shm or f'/dev/shm/baldr{args.beam}.im.shm')
            self.buffer = deque(maxlen=args.frames)
            self.clear = self.camera.clear.copy() if args.demo else None
            self.geometry = ((5.5,5.5,4,1,1) if self.is_jax else (15.5,15.5,11,1,2)) if args.demo else None
            if args.calibration:
                with fits.open(args.calibration) as hdul:
                    self.clear = np.array(hdul['CLEAR_PUPIL'].data, dtype=float)
                    h = hdul[0].header
                    self.geometry = tuple(h[k] for k in ('CX','CY','RPUPIL','ANNGAP','ANNWID'))
                    self.clear_count = self.clear_count or h.get('NCLEAR')
            self.pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn'))
            self.future = None
            self.running = False
            self.next_fit = 0
            self.result = None
            self.history = []
            self.started = time.monotonic()
            root = QtWidgets.QWidget()
            self.setCentralWidget(root)
            layout = QtWidgets.QVBoxLayout(root)
            controls = QtWidgets.QHBoxLayout()
            layout.addLayout(controls)
            for label, callback in [('Capture clear reference', self.capture), ('Save calibration',self.save_calibration), ('Start / pause fitting',self.toggle), ('Save latest result',self.save_result), ('Reset window',self.reset_window)]:
                button = QtWidgets.QPushButton(label)
                button.clicked.connect(callback)
                controls.addWidget(button)
            geometry_controls = QtWidgets.QHBoxLayout()
            layout.addLayout(geometry_controls)
            self.spins = []
            for label, value in zip(('cx','cy','Pupil radius','Annulus gap','Annulus width'),self.geometry or (16,16,10,1,2)):
                geometry_controls.addWidget(QtWidgets.QLabel(label))
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(0 if label != 'Annulus width' else .1,4096)
                spin.setValue(value)
                spin.valueChanged.connect(self.changed)
                geometry_controls.addWidget(spin)
                self.spins.append(spin)
            self.calibration_spins = {}
            if self.is_jax:
                settings_grid = QtWidgets.QGridLayout()
                layout.addLayout(settings_grid)
                fields = [('mask_diameter','Mask diameter λ/D',.01,10),
                          ('theta','Mask phase [rad]',0,6.283185),
                          ('gain','Gain [e-/ADU]',.0001,100000),
                          ('bias_adu','Bias [ADU]',-1000000,1000000),
                          ('read_noise_e','Read RMS [e-]',0,100000),
                          ('background_e','Photon background [e-]',0,1000000),
                          ('correlation_px','Phase correlation [px]',0,20),
                          ('clear_frames','Clear exposures',0,10000000)]
                for i,(key,label,lo,hi) in enumerate(fields):
                    spin = QtWidgets.QDoubleSpinBox()
                    spin.setDecimals(0 if key=='clear_frames' else 4)
                    spin.setRange(lo,hi)
                    spin.setValue((self.clear_count or 0) if key=='clear_frames' else getattr(args,key))
                    spin.valueChanged.connect(self.calibration_changed)
                    settings_grid.addWidget(QtWidgets.QLabel(label),i//4,2*(i%4))
                    settings_grid.addWidget(spin,i//4,2*(i%4)+1)
                    self.calibration_spins[key] = spin
            pixel_controls = QtWidgets.QHBoxLayout()
            layout.addLayout(pixel_controls)
            self.pixel_x, self.pixel_y = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
            for label, spin in (('Pixel x (column)',self.pixel_x),('y (row)',self.pixel_y)):
                spin.setRange(0,4095)
                pixel_controls.addWidget(QtWidgets.QLabel(label))
                pixel_controls.addWidget(spin)
            inspect = QtWidgets.QPushButton('Inspect pixel')
            inspect.clicked.connect(lambda: self.show_pixel(self.pixel_x.value(),self.pixel_y.value()))
            pixel_controls.addWidget(inspect)
            pixel_controls.addWidget(QtWidgets.QLabel('Or click a pixel in any image'))
            pixel_controls.addStretch()
            self.template_dialog = None
            if self.is_jax:
                check_button = QtWidgets.QPushButton('Inspect spatial check')
                check_button.clicked.connect(self.show_template_check)
                pixel_controls.addWidget(check_button)
            self.pixel_dialog = None
            self.status = QtWidgets.QLabel('Demo clear reference loaded. Click Start / pause fitting.' if args.demo else 'Capture clear reference with mask OUT; insert mask, then Start fitting.')
            layout.addWidget(self.status)
            self.summary = QtWidgets.QLabel('Strehl proxy: — (uniform accepted-pixel weighting; no static-phase correction)')
            if self.is_jax:
                self.summary.setText(f'Waiting for first JAX fit at {args.wfs_nm:g} nm.')
            layout.addWidget(self.summary)
            self.template_summary = QtWidgets.QLabel('')
            layout.addWidget(self.template_summary)
            self.template_history = []
            self.fit_warnings = QtWidgets.QLabel('')
            self.fit_warnings.setWordWrap(True)
            self.fit_warnings.setStyleSheet('color: #d97706')
            layout.addWidget(self.fit_warnings)
            if self.is_jax:
                self.fit_warnings.setText('Experimental JAX fit; phase RMS follows the spatial prior. Clear exposures=0 means unknown; capture a reference or enter its count.')
            self.graphics = pg.GraphicsLayoutWidget()
            layout.addWidget(self.graphics)
            self.images = []
            self.colorbars = []
            self.views = []
            for i, title in enumerate(('Live subframe + pupil','Phase RMS σφ [rad]','Histogram residual RMS [bin probability]')):
                plot = self.graphics.addPlot(row=i//2, col=i%2, title=title)
                plot.setAspectLocked(True)
                item = pg.ImageItem()
                plot.addItem(item)
                bar = pg.ColorBarItem(colorMap=pg.colormap.get('viridis'))
                bar.axis.setWidth(85)
                bar.setImageItem(item, insert_in=plot)
                self.colorbars.append(bar)
                self.images.append(item)
                self.views.append(plot)
            self.rings = []
            for colour in ('c','g','g'):
                ring = pg.PlotCurveItem(pen=pg.mkPen(colour))
                self.views[0].addItem(ring)
                self.rings.append(ring)
            self.trend = self.graphics.addPlot(row=1,col=1,title='Model-based Strehl proxy')
            self.trend.setLabel('bottom','Elapsed time',units='s')
            self.trend.setYRange(0,1)
            self.trend_line = self.trend.plot(pen='y',symbol='o')
            self.flagged_line = self.trend.plot(pen=None,symbol='x',symbolPen='orange',symbolSize=10)
            self.template_line = self.trend.plot(pen='c')
            self.flagged_history = []
            if self.is_jax:
                self.trend.setTitle(f'JAX Strehl at {args.wfs_nm:g} nm — yellow: JAX; cyan: spatial; orange ×: flagged')
                if args.backend=='jax-full':
                    self.views[1].setTitle('Phase RMS [rad] — fitted spatial prior')
            self.graphics.scene().sigMouseClicked.connect(self.pixel_clicked)
            self.draw_rings()
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.tick)
            self.timer.start(args.poll_ms)

        def display_image(self, index, data):
            finite = data[np.isfinite(data)]
            lo,hi = (float(finite.min()),float(finite.max())) if finite.size else (0.,1.)
            if hi <= lo:
                hi = lo+max(abs(lo)*.01,1e-6)
            self.images[index].setImage(data,autoLevels=False,levels=(lo,hi))
            self.colorbars[index].setLevels((lo,hi))

        def current_geometry(self):
            return tuple(s.value() for s in self.spins)

        def draw_rings(self):
            cx,cy,r,g,w = self.current_geometry()
            angle = np.linspace(0,2*np.pi,150)
            for line,radius in zip(self.rings,(r,r+g,r+g+w)):
                line.setData(cx+.5+radius*np.cos(angle),cy+.5+radius*np.sin(angle))

        def changed(self):
            self.invalidate_result()
            self.running = False
            self.result = None
            self.draw_rings()
            self.status.setText('Geometry changed. Start fitting to use these masks.')

        def invalidate_result(self):
            self.epoch += 1
            self.result = None
            for image in self.images[1:]:
                image.clear()
            self.summary.setText('Awaiting a fit of the new window/calibration.')
            self.fit_warnings.clear()
            self.template_summary.clear()
            self.template_history.clear()
            self.template_line.clear()
            if self.template_dialog is not None:
                self.template_dialog.close()
            self.history.clear()
            self.flagged_history.clear()
            self.trend_line.clear()
            self.flagged_line.clear()
            if self.pixel_dialog is not None:
                self.pixel_dialog.close()

        def calibration_changed(self):
            self.invalidate_result()
            self.running = False
            self.buffer.clear()
            self.status.setText('Calibration changed. Start fitting after confirming the reference matches this source.')

        def current_jax_settings(self):
            settings = jax_settings(args)
            for key,spin in self.calibration_spins.items():
                settings[key] = int(spin.value()) if key=='clear_frames' else spin.value()
            return settings

        def reset_window(self):
            self.invalidate_result()
            self.buffer.clear()
            self.last_submitted_counter = None
            self.next_fit = 0
            self.status.setText('Window reset; collecting fresh frames. Previous in-flight fit will be discarded.')

        def capture(self):
            if not self.buffer:
                return
            self.clear_count = len(self.buffer)
            self.clear = np.mean([row[2] for row in self.buffer],axis=0)
            if self.is_jax:
                self.calibration_spins['clear_frames'].blockSignals(True)
                self.calibration_spins['clear_frames'].setValue(self.clear_count)
                self.calibration_spins['clear_frames'].blockSignals(False)
            self.invalidate_result()
            self.running = False
            self.result = None
            self.buffer.clear()
            self.status.setText('Clear reference captured. Insert mask, then Start fitting (buffer resets).')

        def toggle(self):
            if self.clear is None:
                self.status.setText('Capture or load a clear reference first.')
                return
            if self.is_jax and self.current_jax_settings()['clear_frames'] < 1:
                self.status.setText('Clear exposure count is unknown. Capture a reference or enter its actual count.')
                return
            self.running = not self.running
            self.epoch += 1
            if self.running:
                self.buffer.clear()
                self.next_fit = 0
                self.last_submitted_counter = None
            else:
                self.status.setText('Fitting paused; acquisition continues.')

        def tick(self):
            try:
                sample = self.camera.read()
                if sample is not None:
                    counter, frame = sample
                    if self.clear is not None and frame.shape != self.clear.shape:
                        raise ValueError('Camera and clear-reference shapes differ')
                    if self.buffer and frame.shape != self.buffer[-1][2].shape:
                        raise ValueError('Camera shape changed; restart monitor')
                    self.last_frame_at = time.monotonic()
                    self.buffer.append((time.time(),counter,frame.astype(np.float32)))
                    self.display_image(0,frame)
            except Exception as exc:
                self.timer.stop()
                self.running = False
                self.status.setText(f'Acquisition stopped: {exc}')
                return
            if self.future is not None and self.future.done():
                try:
                    result = self.future.result()
                    if self.running and self.epoch == self.snapshot_epoch:
                        self.result = result
                        self.result['window_end'] = self.snapshot_end
                        self.result['window_start'] = self.snapshot_start
                        self.result['geometry'] = self.snapshot_geometry
                        self.result['cube'] = self.snapshot
                        self.result['clear'] = self.snapshot_clear
                        self.result['clear_count'] = self.snapshot_clear_count
                        self.result['counters'] = self.snapshot_counters
                        self.result['timestamps'] = self.snapshot_timestamps
                        for index, data in enumerate((result['maps'][2],result['residual']),start=1):
                            self.display_image(index,data)
                        coverage = result['count']/result['active'].sum()
                        self.history.append((time.monotonic()-self.started,result['proxy']))
                        self.history = self.history[-2000:]
                        self.trend_line.setData(*np.asarray(self.history).T)
                        if self.is_jax and not result['checks_passed']:
                            self.flagged_history.append(self.history[-1])
                        self.flagged_history = self.flagged_history[-2000:]
                        if self.flagged_history:
                            self.flagged_line.setData(*np.asarray(self.flagged_history).T)
                        if self.is_jax:
                            check = result['template_check']
                            self.template_history.append((self.history[-1][0],check['strehl']))
                            self.template_history = self.template_history[-2000:]
                            self.template_line.setData(*np.asarray(self.template_history).T)
                            lo,hi = check['candidate_range']
                            self.template_summary.setText(
                                f"Spatial candidate: {check['strehl']:.3f} | {check['family']} | "
                                f"score range {lo:.3f}–{hi:.3f} | clear compatible: {check['clear_compatible']} | "
                                f"{check['outside_pixels']} outside-pupil pixels")
                            self.fit_warnings.setText('; '.join(result['warnings']) or 'Numerical checks passed; optical model and spatial prior still require validation.')
                        label = 'Flagged Strehl candidate' if self.is_jax and not result['checks_passed'] else 'Strehl proxy'
                        coverage_label = 'active' if self.is_jax else 'valid'
                        self.summary.setText(f"{label}: {result['proxy']:.3f} | {coverage_label} {coverage:.0%} ({result['count']} pixels) | μ {result['mu_mode']} | fit {result['seconds']:.1f} s")
                except Exception as exc:
                    if self.epoch == self.snapshot_epoch:
                        self.running = False
                        self.status.setText(f'Fit failed: {exc}. Press Start to retry.')
                finally:
                    self.future = None
            now = time.monotonic()
            fresh = bool(self.buffer) and self.buffer[-1][1] != self.last_submitted_counter
            stale = now-self.last_frame_at > getattr(args,'stale_seconds',5.)
            if self.running and self.future is None and len(self.buffer) == args.frames and now >= self.next_fit and fresh and not stale:
                self.snapshot = np.stack([row[2] for row in self.buffer])
                self.snapshot_start, self.snapshot_end = self.buffer[0][0],self.buffer[-1][0]
                self.snapshot_geometry = self.current_geometry()
                self.snapshot_epoch = self.epoch
                self.snapshot_clear = self.clear.copy()
                self.snapshot_counters = np.array([row[1] for row in self.buffer],dtype=np.int64)
                self.snapshot_timestamps = np.array([row[0] for row in self.buffer])
                self.snapshot_clear_count = self.current_jax_settings()['clear_frames'] if self.is_jax else self.clear_count
                self.last_submitted_counter = self.buffer[-1][1]
                try:
                    if self.is_jax:
                        from .online_jax import analyse_jax
                        self.future = self.pool.submit(analyse_jax,self.snapshot,self.snapshot_clear,self.snapshot_geometry,args.threshold,self.current_jax_settings())
                    else:
                        self.future = self.pool.submit(analyse,self.snapshot,self.snapshot_clear,self.snapshot_geometry,args.threshold,args.mu,args.wfs_nm/args.science_nm)
                except Exception as exc:
                    self.running = False
                    self.status.setText(f'Cannot start fit worker: {exc}. Restart the monitor.')
                    return
                self.next_fit = now+args.interval
            if self.running:
                span = self.buffer[-1][0]-self.buffer[0][0] if self.buffer else 0
                age = f" | result age {time.time()-self.result['window_end']:.1f}s" if self.result else ''
                self.status.setText(f"Buffer {len(self.buffer)}/{args.frames}, spans {span:.1f}s | {'Camera stale — waiting' if stale else ('Fitting…' if self.future else 'Acquiring')}{age}")

        def pixel_clicked(self, event):
            if event.button() != QtCore.Qt.MouseButton.LeftButton:
                return
            pos = event.scenePos()
            for view in self.views:
                if view.vb.sceneBoundingRect().contains(pos):
                    point = view.vb.mapSceneToView(pos)
                    x,y = int(np.floor(point.x())),int(np.floor(point.y()))
                    if x >= 0 and y >= 0:
                        self.pixel_x.setValue(x)
                        self.pixel_y.setValue(y)
                        self.show_pixel(x,y)
                    return

        def show_pixel(self, x, y):
            result = self.result
            if result is None:
                self.status.setText('Wait for a completed fit before inspecting a pixel.')
                return
            shape = result['maps'].shape[1:]
            if not (0 <= y < shape[0] and 0 <= x < shape[1]):
                self.status.setText('Selected pixel is outside the subframe.')
                return
            if (y,x) not in result['profiles']:
                self.status.setText(f'Pixel ({x}, {y}) is inactive or its fit failed.')
                return
            # Freeze the completed window so curves remain mutually consistent.
            a,b,sigma,noise = result['maps'][:4,y,x]
            mu = result['mu_map'][y,x]
            centres,model,hist = result['profiles'][(y,x)]
            phi = np.linspace(-max(np.pi,3.5*sigma),max(np.pi,3.5*sigma),1000)
            curve = a+b*np.cos(phi+mu)
            if self.pixel_dialog is not None:
                self.pixel_dialog.close()
            dialog = QtWidgets.QDialog(self)
            dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            dialog.destroyed.connect(lambda: self.clear_dialog(dialog))
            self.pixel_dialog = dialog
            dialog.setWindowTitle(f'Pixel ({x}, {y}): intensity / phase fit')
            dialog.resize(950,750)
            layout = QtWidgets.QVBoxLayout(dialog)
            stamp = time.strftime('%H:%M:%S',time.localtime(result['window_end']))
            label = QtWidgets.QLabel(
                f'Window ending {stamp} | A={a:.2f}, B={b:.2f} ADU | '
                f'μ={mu:.3f}, σφ={sigma:.3f} rad | noise={noise:.2f} ADU\n'
                'Phase marginal is inferred N(0, σφ²); intensity histogram is measured. '
                'This popup holds a fixed snapshot.')
            if result.get('backend')=='jax':
                label.setText(label.text()+'\nJAX: intensity PDF uses calibrated shot/read noise; tail bins omitted from this plot. '+('Cosine curve is a coherent-component illustration, not the full forward model. Phase RMS follows the spatial prior.' if result['settings']['model']=='full' else 'Phase/noise are constrained by the optical calibration.'))
            label.setWordWrap(True)
            layout.addWidget(label)
            graphics = pg.GraphicsLayoutWidget()
            layout.addWidget(graphics)
            phase_plot = graphics.addPlot(row=0,col=0,title='Inferred phase distribution')
            trace = graphics.addPlot(row=1,col=0,title='I = A + B cos(φ + μ)')
            intensity_plot = graphics.addPlot(row=1,col=1,title='Intensity distribution')
            for plot in (phase_plot,trace,intensity_plot):
                for axis in ('left','bottom'):
                    plot.getAxis(axis).enableAutoSIPrefix(False)
            phase_plot.setXLink(trace)
            intensity_plot.setYLink(trace)
            phase_plot.plot(phi,norm.pdf(phi,0,sigma),pen='c',fillLevel=0,brush=(0,200,200,60))
            phase_plot.setLabel('left','Probability density',units='1/rad')
            phase_plot.hideAxis('bottom')
            trace.setLabel('bottom','Phase φ',units='rad')
            trace.setLabel('left','Bias-subtracted intensity',units='ADU')
            trace.addLegend()
            trace.plot(phi,curve,pen=pg.mkPen('y',width=2),name='Fitted response')
            if result.get('backend')!='jax':
                upper = trace.plot(phi,curve+noise,pen=None)
                lower = trace.plot(phi,curve-noise,pen=None)
                trace.addItem(pg.FillBetweenItem(upper,lower,brush=(255,220,0,45)))
            elif result['settings']['model']=='full':
                trace.setTitle('Coherent-component illustration only')
                phase_plot.setTitle('Gaussian phase prior at fitted scale')
            trace.showGrid(x=True,y=True,alpha=.25)
            intensity_plot.addLegend()
            intensity_plot.addItem(pg.BarGraphItem(x0=0,y=centres,width=hist,
                height=.9*(centres[1]-centres[0]),brush=(0,200,200,90),pen=None))
            intensity_plot.plot(hist,centres,pen='c',name='Measured histogram')
            intensity_plot.plot(model,centres,pen=pg.mkPen('y',width=2),name='Fitted PDF')
            intensity_plot.setLabel('bottom','Probability density',units='1/ADU')
            intensity_plot.hideAxis('left')
            graphics.ci.layout.setColumnStretchFactor(0,3)
            graphics.ci.layout.setColumnStretchFactor(1,1)
            graphics.ci.layout.setRowStretchFactor(0,1)
            graphics.ci.layout.setRowStretchFactor(1,3)
            dialog.show()

        def clear_dialog(self, dialog):
            if self.pixel_dialog is dialog:
                self.pixel_dialog = None

        def show_template_check(self):
            if self.result is None or 'template_check' not in self.result:
                return
            check = self.result['template_check']
            if self.template_dialog is not None:
                self.template_dialog.close()
            dialog = QtWidgets.QDialog(self)
            dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            dialog.setWindowTitle('Full-subframe spatial check — latest completed window')
            dialog.resize(1100,800)
            self.template_dialog = dialog
            dialog.destroyed.connect(lambda: self.clear_template_dialog(dialog))
            layout = QtWidgets.QVBoxLayout(dialog)
            note = QtWidgets.QLabel('Coherent-reference hypothesis check; score range is not a confidence interval. '
                                   'All finite subframe pixels enter the fit. Flux and offset are fitted.')
            note.setWordWrap(True)
            layout.addWidget(note)
            graphics = pg.GraphicsLayoutWidget()
            layout.addWidget(graphics)
            for i,(key,title) in enumerate((('mean_e','Measured mean [electrons]'),
                    ('best_image_e','Best spatial template [electrons]'),
                    ('residual_sigma','Residual / mean uncertainty'))):
                plot = graphics.addPlot(row=i//2,col=i%2,title=title)
                plot.setAspectLocked(True)
                item = pg.ImageItem(check[key])
                plot.addItem(item)
                bar = pg.ColorBarItem(colorMap=pg.colormap.get('viridis'))
                bar.setImageItem(item,insert_in=plot)
                finite = check[key][np.isfinite(check[key])]
                lo,hi = (float(finite.min()),float(finite.max())) if finite.size else (0.,1.)
                if key=='residual_sigma':
                    hi = max(abs(lo),abs(hi),1.)
                    lo = -hi
                elif hi<=lo:
                    hi = lo+max(abs(lo)*.01,1e-6)
                bar.setLevels((lo,hi))
            profile = graphics.addPlot(row=1,col=1,title='Template score relative to best fit')
            profile.addLegend()
            for values,name,colour in zip(check['delta_score'],check['families'],('c','m')):
                profile.plot(check['grid'],values,pen=colour,name=name)
            profile.addItem(pg.InfiniteLine(pos=9,angle=0,pen=pg.mkPen('w',style=QtCore.Qt.PenStyle.DashLine)))
            profile.addItem(pg.InfiniteLine(pos=self.result['proxy'],angle=90,pen='y'))
            profile.setLabel('bottom','Template Strehl (yellow line: JAX)')
            profile.setLabel('left','Δ score')
            profile.setYRange(0, min(100,max(12,float(np.max(check['delta_score'])))))
            dialog.show()

        def clear_template_dialog(self, dialog):
            if self.template_dialog is dialog:
                self.template_dialog = None

        def save_calibration(self):
            if self.clear is None:
                return
            path,_ = QtWidgets.QFileDialog.getSaveFileName(self,'Save calibration','phase_calibration.fits','FITS (*.fits)')
            if path:
                self.write_fits(path,False)

        def save_result(self):
            if self.result is None:
                return
            path,_ = QtWidgets.QFileDialog.getSaveFileName(self,'Save result','phase_result.fits','FITS (*.fits)')
            if path:
                self.write_fits(path,True)

        def write_fits(self,path,with_result):
            try:
                result = self.result if with_result else None
                primary = fits.PrimaryHDU(result['cube'] if result else None)
                geometry = result['geometry'] if result else self.current_geometry()
                for key,value in zip(('CX','CY','RPUPIL','ANNGAP','ANNWID'),geometry):
                    primary.header[key] = value
                count = result.get('clear_count') if result else (self.current_jax_settings()['clear_frames'] if self.is_jax else self.clear_count)
                if count:
                    primary.header['NCLEAR'] = count
                hdus = [primary,fits.ImageHDU(result['clear'] if result else self.clear,name='CLEAR_PUPIL')]
                if result:
                    for name,data in zip(('A','B','SIGMA_PHI','SIGMA_NOISE','FIT_COST'),result['maps']):
                        hdus.append(fits.ImageHDU(data,name=name))
                    hdus.append(fits.ImageHDU(result['mu_map'],name='MU'))
                    hdus.append(fits.ImageHDU(result['residual'],name='HIST_RMS'))
                    primary.header['MUMODE'] = result['mu_mode']
                    primary.header['HISTRMS'] = 'RMS bin-probability residual; excludes prior'
                    hdus.append(fits.ImageHDU(result['active'].astype(np.uint8),name='ACTIVE'))
                    for key,value in [('BIAS',result['bias']),('WFS_NM',args.wfs_nm),('SCI_NM',args.science_nm),('TSTART',result['window_start']),('TEND',result['window_end']),('NVALID',result['count'])]:
                        primary.header[key] = value
                    if np.isfinite(result['proxy']):
                        primary.header['SPROXY'] = result['proxy']
                    primary.header['COMMENT'] = 'Strehl proxy: exp(-mean accepted phase variance), wavelength scaled.'
                    if result.get('backend')=='jax':
                        self.add_jax_hdus(primary,hdus,result)
                    hdus.extend([fits.ImageHDU(result['counters'],name='FRAME_COUNTERS'),
                                 fits.ImageHDU(result['timestamps'],name='READ_TIMES')])
                fits.HDUList(hdus).writeto(path,overwrite=True)
                self.status.setText(f'Saved {path}')
            except Exception as exc:
                self.status.setText(f'Save failed: {exc}')

        def add_jax_hdus(self,primary,hdus,result):
            j = result['jax_result']
            settings = result['settings']
            del primary.header['COMMENT']
            primary.header['MODEL'] = 'jax-'+settings['model']
            primary.header['CHECKS'] = result['checks_passed']
            primary.header['SUCCESS'] = j['success']
            primary.header['COMMENT'] = 'Experimental JAX Strehl at sensing wavelength; see FIT_INFO for diagnostics.'
            primary.header['COMMENT'] = 'SIGMA_NOISE is read RMS only; A/B are coherent-component diagnostics.'
            for name,key in [('GAIN','gain'),('BIASADU','bias_adu'),('RON_E','read_noise_e'),
                             ('BKG_E','background_e'),('MASKDIAM','mask_diameter'),('THETA','theta'),
                             ('CORRPX','correlation_px'),('NODES','nodes'),('FFTSIZE','fft_size')]:
                primary.header[name] = settings[key]
            hdus.extend([fits.ImageHDU(j['maps']['B0'],name='B0_ELECTRONS'),
                         fits.ImageHDU(j['maps']['CLEAR_FIT'],name='CLEAR_FIT_E')])
            for key in ('model_probabilities','histogram_counts','histogram_edges'):
                hdus.append(fits.ImageHDU(j[key],name=key.upper()))
            info = {k:v for k,v in j.items() if k not in ('maps','active','model_probabilities','histogram_counts','histogram_edges')}
            info['settings'] = settings
            info['online_warnings'] = result['warnings']
            raw = json.dumps(info).encode('utf-8')
            hdus.append(fits.ImageHDU(np.frombuffer(raw,dtype=np.uint8),name='FIT_INFO'))
            check = result['template_check']
            primary.header['STPL'] = check['strehl']
            primary.header['TPLZERO'] = check['clear_compatible']
            primary.header['TPLDIFF'] = check['jax_disagrees']
            metadata = {}
            for key,value in check.items():
                if isinstance(value,np.ndarray):
                    hdus.append(fits.ImageHDU(value.astype(np.uint8) if value.dtype==bool else value,
                                             name='TPL_'+key.upper()))
                else:
                    metadata[key] = value
            raw = json.dumps(metadata).encode('utf-8')
            hdus.append(fits.ImageHDU(np.frombuffer(raw,dtype=np.uint8),name='TEMPLATE_INFO'))

        def closeEvent(self,event):
            self.timer.stop()
            self.camera.close()
            # Shutdown waits for at most the one in-flight bounded fitting job.
            self.pool.shutdown(wait=False,cancel_futures=True)
            event.accept()

    window = Window()
    window.show()
    app.exec()


def main(default_backend="classic"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('classic','jax-full','jax-coherent'),default=default_backend)
    parser.add_argument('--beam',type=int,default=1,choices=(1,2,3,4))
    parser.add_argument('--shm',type=Path)
    parser.add_argument('--frames',type=int,default=500)
    parser.add_argument('--interval',type=float,default=10,help='Minimum seconds between fit starts')
    parser.add_argument('--poll-ms',type=int,default=20,help='Read latest frame every this many ms; not a lossless recorder')
    parser.add_argument('--calibration',type=Path,help='FITS with CLEAR_PUPIL and geometry headers, including original filtered cube')
    parser.add_argument('--threshold',type=float,default=.1,help='Active threshold as fraction of peak bias-subtracted clear pupil')
    parser.add_argument('--mu',type=float,help='Fix mu for all pixels; default fits mu independently per pixel')
    parser.add_argument('--wfs-nm',type=float,default=1250,help='Effective sensing wavelength; default 1250 nm')
    parser.add_argument('--science-nm',type=float,help='Strehl wavelength; defaults to sensing wavelength')
    parser.add_argument('--demo',action='store_true')
    parser.add_argument('--mask-diameter',type=float,help='JAX: diameter in lambda/D at sensing wavelength')
    parser.add_argument('--theta',type=float,default=np.pi/2)
    parser.add_argument('--gain',type=float,help='JAX: electrons per ADU')
    parser.add_argument('--bias-adu',type=float)
    parser.add_argument('--read-noise-e',type=float,help='JAX: total additive noise RMS in electrons')
    parser.add_argument('--background-e',type=float,default=0.)
    parser.add_argument('--clear-frames',type=int,help='Count averaged in a loaded clear reference; captured automatically for new references')
    parser.add_argument('--nodes',type=int,default=256)
    parser.add_argument('--fft-size',type=int,default=129)
    parser.add_argument('--maxiter',type=int,default=120)
    parser.add_argument('--correlation-px',type=float,default=.8)
    parser.add_argument('--pixelwise',action='store_true')
    parser.add_argument('--demo-sigma',type=float,default=.7)
    parser.add_argument('--stale-seconds',type=float,default=5.)
    args = parser.parse_args()
    if args.backend != 'classic':
        if args.demo:
            for key,value in dict(mask_diameter=1.06,gain=1.,bias_adu=100.,read_noise_e=8.,clear_frames=100).items():
                if getattr(args,key) is None:
                    setattr(args,key,value)
        missing = [key for key in ('mask_diameter','gain','bias_adu','read_noise_e') if getattr(args,key) is None]
        if missing:
            parser.error('JAX live mode needs calibrated values for: '+', '.join('--'+key.replace('_','-') for key in missing))
        if args.clear_frames is not None and args.clear_frames<1:
            parser.error('--clear-frames must be positive')
        if args.mu is not None:
            parser.error('--mu applies only to the classic fitter; JAX derives its reference optically')
        if args.mask_diameter<=0 or args.gain<=0 or args.read_noise_e<0 or args.background_e<0 or args.nodes<64 or args.correlation_px<0 or args.maxiter<1:
            parser.error('Invalid JAX optical/noise/fitting settings')
        if args.backend=='jax-full' and args.pixelwise:
            parser.error('Full JAX mode currently fits a common spatial phase scale; omit --pixelwise')
        if args.science_nm is not None and args.science_nm != args.wfs_nm:
            parser.error('Online JAX Strehl is currently reported at the sensing wavelength; omit --science-nm')

    args.science_nm = args.science_nm or args.wfs_nm
    if args.frames < 100 or args.interval <= 0 or args.poll_ms < 1 or not 0 < args.threshold < 1 or args.wfs_nm <= 0 or args.science_nm <= 0 or args.stale_seconds <= 0:
        parser.error('Need frames >=100, positive times/wavelengths and 0 < threshold < 1')
    run(args)


if __name__ == '__main__':
    main()
