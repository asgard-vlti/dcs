"""Independent one-to-four-beam halo Strehl monitor; read-only camera attachment."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from queue import SimpleQueue
import json
from pathlib import Path
import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

from .halo_estimator import measure_halo, reference_from_window
from .monitor import Camera, masks


class FixedViewBox(pg.ViewBox):
    """Disable mouse and keyboard changes; only application code sets the range."""
    def __init__(self):
        super().__init__(enableMenu=False)
        self.setMouseEnabled(x=False,y=False)
        self.disableAutoRange()

    def wheelEvent(self,event,axis=None):
        event.accept()

    def mouseDragEvent(self,event,axis=None):
        event.accept()

    def mouseClickEvent(self,event):
        event.accept()

    def mouseDoubleClickEvent(self,event):
        event.accept()

    def keyPressEvent(self,event):
        event.accept()


class DemoCamera:
    """Energy-conserving illustrative data; not an atmospheric simulation."""
    def __init__(self, beam, args):
        self.args=args
        self.beam=beam
        self.counter=0
        self.mode='Clear (mask OUT)'
        self.rng=np.random.default_rng(beam)
        y,x=np.indices((32,32))
        self.pupil=(x-15.5)**2+(y-15.5)**2<=8**2
        r=np.hypot(x-15.5,y-15.5)
        self.halo=np.where(self.pupil,0,np.exp(-r/7))
        self.halo/=self.halo.sum()

    def read(self):
        self.counter+=1
        s=0. if self.mode.startswith('Clear') else (1. if self.mode.startswith('Reference') else .5+.4*np.sin(self.counter/40+self.beam))
        total=3000*self.pupil.sum()
        signal=3000*self.pupil*(1-.2*s)+total*.2*s*self.halo
        a=self.args
        frame=a.bias_adu+(self.rng.poisson(signal+a.background_e)+self.rng.normal(0,a.read_noise_e,signal.shape))/a.gain
        return self.counter,frame

    def close(self):
        pass


class BeamColumn(QtWidgets.QWidget):
    def __init__(self, beam, path, args, camera_factory=Camera):
        super().__init__()
        self.beam,self.path,self.args=beam,path,args
        self.camera_factory=camera_factory
        self.camera=None
        self.pool=ThreadPoolExecutor(max_workers=1)
        self.task=None
        self.messages=SimpleQueue()
        self.acquisition_record=None
        self.last_attempt=None
        self.motion_fault=False
        self.clear=None
        self.pixel_quality=None
        self.clear_count=0
        self.calibration=None
        self.shape=None
        self.buffer=deque(maxlen=args.frames)
        self.history=deque(maxlen=args.history)
        self.started=time.monotonic()
        self.last_frame_at=None
        self.settings={}
        layout=QtWidgets.QVBoxLayout(self)
        title=QtWidgets.QLabel(f'Beam {beam}')
        title.setStyleSheet('font-weight: bold; font-size: 16px')
        layout.addWidget(title)
        settings_panel=QtWidgets.QWidget()
        grid=QtWidgets.QGridLayout(settings_panel)
        grid.setContentsMargins(0,0,0,0)
        settings_toggle=QtWidgets.QToolButton()
        settings_toggle.setText('Pupil / detector settings ▸')
        settings_toggle.setCheckable(True)
        settings_toggle.toggled.connect(settings_panel.setVisible)
        settings_toggle.toggled.connect(lambda checked:settings_toggle.setText('Pupil / detector settings ▾' if checked else 'Pupil / detector settings ▸'))
        layout.addWidget(settings_toggle)
        layout.addWidget(settings_panel)
        settings_panel.setVisible(False)
        for index,(key,label,value,low,high) in enumerate((
            ('cx','Pupil x',15.5,-4096,4096),('cy','Pupil y',15.5,-4096,4096),
            ('radius','Radius',8,.1,4096),('gain','e⁻/ADU',args.gain,.0001,1e6),
            ('bias_adu','Bias ADU',args.bias_adu,-1e8,1e8),
            ('read_noise_e','Read RMS e⁻',args.read_noise_e,0,1e6),
            ('background_e','Background e⁻',args.background_e,0,1e8))):
            spin=QtWidgets.QDoubleSpinBox()
            spin.setDecimals(4 if key=='gain' else 2)
            spin.setMinimumWidth(65)
            spin.setMaximumWidth(100)
            spin.setRange(low,high)
            spin.setValue(value)
            self.settings[key]=spin
            grid.addWidget(QtWidgets.QLabel(label),index//2,2*(index%2))
            grid.addWidget(spin,index//2,2*(index%2)+1)
            spin.valueChanged.connect(self.settings_changed)
        self.reference_s=QtWidgets.QDoubleSpinBox()
        self.reference_s.setMinimumWidth(65)
        self.reference_s.setMaximumWidth(100)
        self.reference_s.setDecimals(3)
        self.reference_s.setRange(.001,1)
        self.reference_s.setValue(1)
        reference_row=QtWidgets.QHBoxLayout()
        reference_row.addWidget(QtWidgets.QLabel('Reference Strehl'))
        reference_row.addWidget(self.reference_s)
        reference_row.addStretch()
        layout.addLayout(reference_row)
        self.reference_s.setToolTip('Independently known reference Strehl; 1 is an assumption, not a measurement.')
        buttons=QtWidgets.QGridLayout()
        layout.addLayout(buttons)
        for i,(label,callback) in enumerate((('get clear pupil',self.capture_clear),
                ('Calibrate (IN)',self.calibrate),('Reset window',self.reset_window),
                ('Reconnect',self.connect_camera))):
            button=QtWidgets.QPushButton(label)
            button.clicked.connect(callback)
            buttons.addWidget(button,i//2,i%2)
        if args.demo:
            self.mode=QtWidgets.QComboBox()
            self.mode.addItems(['Clear (mask OUT)','Reference (S=1)','Varying Strehl'])
            self.mode.currentTextChanged.connect(self.demo_mode_changed)
            layout.addWidget(self.mode)
        self.weighting=QtWidgets.QCheckBox('Weight suspect pixels from clear reference')
        self.weighting.setChecked(True)
        self.weighting.toggled.connect(self.settings_changed)
        layout.addWidget(self.weighting)
        self.weight_label=QtWidgets.QLabel('Pixel quality: awaiting clear calibration')
        self.weight_label.setWordWrap(True)
        layout.addWidget(self.weight_label)
        self.status=QtWidgets.QLabel('Connecting…')
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(44)
        layout.addWidget(self.status)
        self.readout=QtWidgets.QLabel('Awaiting frames')
        self.readout.setWordWrap(True)
        self.readout.setMinimumHeight(58)
        layout.addWidget(self.readout)
        self.graphics=pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics,1)
        self.pupil_plot=self.graphics.addPlot(row=0,col=0,title='Live pupil [ADU]',viewBox=FixedViewBox())
        self.pupil_plot.setAspectLocked(True)
        self.image=pg.ImageItem()
        self.pupil_plot.addItem(self.image)
        self.image.setColorMap(pg.colormap.get('viridis'))
        self.ring=self.pupil_plot.plot(pen='c')
        self.trend=self.graphics.addPlot(row=1,col=0,title='Calibrated halo Strehl',viewBox=FixedViewBox())
        self.trend.setLabel('bottom','Frame offset (latest = 0)',**{'font-size':'14pt','font-family':'DejaVu Sans','color':'#eeeeee'})
        self.trend.setLabel('left','Strehl estimate')
        self.trend.setYRange(-.1,1.1,padding=0)
        self.trend.setXRange(1-args.history,0,padding=0)
        self.line=self.trend.plot(pen='m')
        self.error=pg.ErrorBarItem(pen=pg.mkPen(150,150,150,90))
        self.trend.addItem(self.error)
        for plot in (self.pupil_plot,self.trend):
            plot.hideButtons()
            plot.setMenuEnabled(False)
            plot.disableAutoRange()
            plot.setTitle(plot.titleLabel.text,size='15pt',color='#eeeeee')
            for side in ('left','bottom'):
                axis=plot.getAxis(side)
                axis.setStyle(tickFont=QtGui.QFont('DejaVu Sans',12))
                axis.setTextPen('#eeeeee')
            plot.getAxis('left').setWidth(85)
            plot.getAxis('bottom').setHeight(60)
        self.pupil_plot.setLabel('bottom','Column',**{'font-size':'14pt'})
        self.pupil_plot.setLabel('left','Row',**{'font-size':'14pt'})
        self.draw_pupil()
        self.connect_camera()

    def config(self):
        return {key:spin.value() for key,spin in self.settings.items()}

    def draw_pupil(self):
        c=self.config()
        a=np.linspace(0,2*np.pi,160)
        self.ring.setData(c['cx']+.5+c['radius']*np.cos(a),c['cy']+.5+c['radius']*np.sin(a))

    def settings_changed(self):
        # Detector or aperture changes invalidate clear and mask-in calibrations.
        self.clear=None
        self.pixel_quality=None
        self.clear_count=0
        self.calibration=None
        self.reset_window()
        self.draw_pupil()
        self.weight_label.setText('Pixel quality: awaiting clear calibration')
        self.status.setText('Settings changed: capture clear, then calibrate mask IN again.')

    def reset_window(self):
        self.buffer.clear()
        self.history.clear()
        self.line.clear()
        self.error.setData(x=np.array([]),y=np.array([]),height=np.array([]))
        self.readout.setText('Awaiting fresh frames')
        self.status.setText('Window reset; existing clear and halo calibrations retained.')

    def demo_mode_changed(self,text):
        if self.camera is not None:
            self.camera.mode=text
        self.reset_window()

    def connect_camera(self):
        if self.camera is not None:
            self.camera.close()
            self.camera=None
        self.clear=None
        self.pixel_quality=None
        self.clear_count=0
        self.calibration=None
        self.shape=None
        self.last_frame_at=None
        self.motion_fault=False
        self.reset_window()
        self.weight_label.setText('Pixel quality: awaiting clear calibration')
        try:
            self.camera=DemoCamera(self.beam,self.args) if self.args.demo else self.camera_factory(self.path)
            if self.args.demo:
                self.camera.mode=self.mode.currentText()
            self.status.setText('Connected. Get clear pupil starts from mask IN.' if self.args.mds else 'Connected. Move mask OUT, then capture fresh clear.')
        except Exception as exc:
            self.status.setText(f'Unavailable: {exc}. Other beams continue; use Reconnect.')

    def measurement(self,frames,calibration=None):
        c=self.config()
        pupil,_=masks(self.clear.shape,(c['cx'],c['cy'],c['radius'],0,0))
        data=(np.asarray(frames,float)-c['bias_adu'])*c['gain']-c['background_e']
        clear=(self.clear-c['bias_adu'])*c['gain']-c['background_e']
        return measure_halo(data,clear,pupil,c['read_noise_e'],self.clear_count,
                            c['background_e'],calibration,
                            weights=self.pixel_quality['weights'] if self.pixel_quality is not None else None)

    def set_controls_enabled(self, enabled):
        for kind in (QtWidgets.QAbstractButton,QtWidgets.QDoubleSpinBox,QtWidgets.QComboBox):
            for control in self.findChildren(kind):
                control.setEnabled(enabled)

    def begin_capture(self, kind):
        from .halo_acquisition import acquire
        if self.task is not None:
            return
        if self.camera is None or self.motion_fault:
            self.status.setText('Check mask position and reconnect before calibration')
            return
        if kind=='reference' and self.clear is None:
            self.status.setText('Capture clear first')
            return
        self.capture_kind=kind
        self.capture_s=self.reference_s.value()
        self.buffer.clear()
        self.set_controls_enabled(False)
        self.status.setText('Starting fresh calibration; ignoring previous rolling buffer')
        factory=self.camera_factory
        if self.args.demo:
            # Demo captures do not send ZMQ or access instrument streams.
            mode='Clear (mask OUT)' if kind=='clear' else self.mode.currentText()
            def factory(path):
                camera=DemoCamera(self.beam,self.args)
                camera.mode=mode
                return camera
        address=self.args.mds if kind=='clear' and not self.args.demo else None
        self.task=self.pool.submit(acquire,self.path,self.args.frames,
            self.args.mask_settle_s,self.args.capture_timeout,mds=address,
            axis=f'{self.args.mask_axis}{self.beam}',offset=self.args.mask_offset_um,
            tolerance=self.args.position_tolerance,motion_timeout=self.args.motion_timeout,
            camera_factory=factory,progress=self.messages.put)

    def capture_clear(self):
        self.begin_capture('clear')

    def calibrate(self):
        self.begin_capture('reference')

    def poll_capture(self):
        while not self.messages.empty():
            self.status.setText(self.messages.get())
        if self.task is None or not self.task.done():
            return
        try:
            captured=self.task.result()
            rows=captured['rows']
            cube=np.stack([row[2] for row in rows])
            self.last_attempt=dict(kind=self.capture_kind,rows=rows,motion=captured['motion'])
            if self.capture_kind=='clear':
                from .pixel_weights import clear_pixel_weights
                cfg=self.config()
                pupil,_=masks(cube.shape[1:],(cfg['cx'],cfg['cy'],cfg['radius'],0,0))
                electrons=(cube-cfg['bias_adu'])*cfg['gain']-cfg['background_e']
                quality=clear_pixel_weights(electrons,pupil,cfg['read_noise_e'],cfg['background_e']) if self.weighting.isChecked() else None
                self.pixel_quality=quality
                weights=quality['weights'] if quality is not None else np.ones(pupil.shape)
                self.weight_label.setText(f'Pixel weights: {int((weights==0).sum())} excluded; '
                    f'{int(((weights>0)&(weights<.99)).sum())} reduced. '
                    f'Mean weight inside {weights[pupil].mean():.1%}, outside {weights[~pupil].mean():.1%}.')
                self.clear=cube.mean(axis=0)
                self.clear_count=len(cube)
                self.calibration=None
                self.acquisition_record=captured['motion']
                self.reset_window()
                self.status.setText('Fresh clear captured; mask restored.' if captured['motion'] else
                                    'Fresh clear captured. Insert mask before Calibrate (IN).')
            else:
                h=self.measurement(cube)
                # Keep the exact attempted reference for export even on failure.
                self.buffer.clear()
                self.buffer.extend(rows)
                self.calibration=reference_from_window(h,self.capture_s)
                self.history.clear()
                self.line.clear()
                self.status.setText(f"Fresh reference calibrated at S={self.calibration['strehl']:.3f}.")
        except Exception as exc:
            if 'MASK RESTORATION UNVERIFIED' in str(exc):
                self.motion_fault=True
                self.calibration=None
                self.history.clear()
                self.line.clear()
                self.error.setData(x=np.array([]),y=np.array([]),height=np.array([]))
            self.status.setText(f'Calibration failed: {exc}'+
                                (' Previous calibration retained.' if self.calibration is not None else ''))
        finally:
            self.task=None
            self.set_controls_enabled(True)

    def tick(self):
        self.poll_capture()
        if self.camera is None:
            return
        try:
            sample=self.camera.read()
            if sample is None:
                if self.last_frame_at is not None and time.monotonic()-self.last_frame_at>self.args.stale_seconds:
                    self.readout.setText('STALE: no new camera frames')
                return
            counter,frame=sample
            frame=np.asarray(frame,float)
            if frame.ndim!=2 or (self.shape is not None and frame.shape!=self.shape):
                raise ValueError('Camera shape changed; reconnect and recalibrate')
            self.shape=frame.shape
            self.pupil_plot.setRange(xRange=(0,frame.shape[1]),yRange=(0,frame.shape[0]),padding=0)
            self.last_frame_at=time.monotonic()
            timestamp=time.time()
            if self.task is None:
                self.buffer.append((timestamp,counter,frame.copy()))
            finite=frame[np.isfinite(frame)]
            levels=(float(finite.min()),float(finite.max())) if finite.size else (0,1)
            if levels[1]<=levels[0]:
                levels=(levels[0],levels[0]+1)
            self.image.setImage(frame,autoLevels=False,levels=levels)
            if self.task is not None:
                self.readout.setText('Calibration acquiring fresh frames; measurement paused')
                return
            if self.motion_fault:
                self.readout.setText('Mask position unverified. Check stage and reconnect; estimates disabled.')
                return
            prefix=f'Buffer {len(self.buffer)}/{self.args.frames}'
            if self.clear is None:
                self.readout.setText(prefix+' | Capture clear first')
                return
            try:
                h=self.measurement(frame,self.calibration)
            except ValueError as exc:
                self.readout.setText(prefix+' | '+str(exc))
                return
            value,error=h['strehl'][0],h['strehl_sigma'][0]
            self.history.append((timestamp,counter,h['ratio'][0],h['fraction'][0],value,error,h['fraction_sigma'][0]))
            if not h['valid'][0]:
                self.readout.setText(prefix+' | Invalid / low total or pupil SNR')
            elif self.calibration is None:
                self.readout.setText(prefix+f" | Outside/inside {h['ratio'][0]:.4f}\nOutside fraction {h['fraction'][0]:.4f}; clear {h['clear_fraction']:.4f}; uncalibrated")
            else:
                self.readout.setText(prefix+f" | S = {value:.3f} ± {error:.3f}\nOutside/inside {h['ratio'][0]:.4f} (statistical error only)")
            rows=np.asarray(self.history)
            x=np.arange(1-len(rows),1)
            calibrated=self.calibration is not None
            self.trend.setTitle('Calibrated halo Strehl' if calibrated else 'Outside fraction — uncalibrated',size='15pt',color='#eeeeee')
            self.trend.setLabel('left','Strehl estimate' if calibrated else 'Outside / total flux',**{'font-size':'14pt','font-family':'DejaVu Sans','color':'#eeeeee'})
            y=rows[:,4] if calibrated else rows[:,3]
            error=rows[:,5] if calibrated else rows[:,6]
            self.line.setData(x,y,connect='finite')
            self.error.setData(x=x,y=y,height=2*error)
        except Exception as exc:
            self.status.setText(f'Acquisition stopped: {exc}. Reconnect this beam.')
            self.close_camera()

    @property
    def started_wall(self):
        # Set on construction in Window; explicit wall origin keeps exports aligned.
        if not hasattr(self,'_started_wall'):
            self._started_wall=time.time()-(time.monotonic()-self.started)
        return self._started_wall

    def save_result(self):
        if self.clear is None or not self.buffer:
            self.status.setText('Need clear reference and acquired frames before saving.')
            return
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,'Save halo result',f'halo_beam{self.beam}.fits','FITS (*.fits)')
        if path:
            try:
                self.write_fits(path)
                self.status.setText(f'Saved {path}')
            except Exception as exc:
                self.status.setText(f'Save failed: {exc}')

    def write_fits(self,path):
        from astropy.io import fits
        cube=np.stack([r[2] for r in self.buffer])
        h=self.measurement(cube,self.calibration)
        primary=fits.PrimaryHDU(cube)
        primary.header['BEAM']=self.beam
        primary.header['NCLEAR']=self.clear_count
        hdus=[primary,fits.ImageHDU(self.clear,name='CLEAR_PUPIL'),
              fits.ImageHDU(np.array([r[0] for r in self.buffer]),name='READ_TIMES'),
              fits.ImageHDU(np.array([r[1] for r in self.buffer],dtype=np.int64),name='FRAME_COUNTERS')]
        if self.pixel_quality is not None:
            for key,value in self.pixel_quality.items():
                hdus.append(fits.ImageHDU(value,name='PIXEL_'+key.upper()))
        if self.last_attempt is not None:
            attempt=self.last_attempt
            hdus.extend([fits.ImageHDU(np.stack([r[2] for r in attempt['rows']]),name='CAL_ATTEMPT'),
                         fits.ImageHDU(np.array([r[1] for r in attempt['rows']],dtype=np.int64),name='CAL_COUNTERS'),
                         fits.ImageHDU(np.array([r[0] for r in attempt['rows']]),name='CAL_TIMES')])
            primary.header['CAL_KIND']=attempt['kind']
        info=dict(settings=self.config(),stream=str(self.path),beam=self.beam,calibration=self.calibration,clear_motion=self.acquisition_record)
        for key,value in h.items():
            if isinstance(value,np.ndarray):
                hdus.append(fits.ImageHDU(value.astype(np.uint8) if value.dtype==bool else value,name='HALO_'+key.upper()))
            else:
                info[key]=value
        hdus.append(fits.ImageHDU(np.frombuffer(json.dumps(info).encode(),dtype=np.uint8),name='HALO_INFO'))
        fits.HDUList(hdus).writeto(path,overwrite=True)

    def close_camera(self):
        if self.camera is not None:
            self.camera.close()
            self.camera=None


class Window(QtWidgets.QMainWindow):
    def __init__(self,args,camera_factory=Camera):
        super().__init__()
        self.setWindowTitle('Baldr — multi-beam halo Strehl')
        self.resize(min(1900,500*len(args.beams)),950)
        root=QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout=QtWidgets.QVBoxLayout(root)
        columns=QtWidgets.QHBoxLayout()
        layout.addLayout(columns,1)
        self.columns=[]
        origin=time.time()
        for beam in args.beams:
            path=Path(args.shm_template.format(beam=beam))
            column=BeamColumn(beam,path,args,camera_factory)
            column._started_wall=origin
            if self.columns:
                column.trend.setXLink(self.columns[0].trend)
            self.columns.append(column)
            columns.addWidget(column,1)
        self.timer=QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(args.poll_ms)

    def tick(self):
        for column in self.columns:
            column.tick()

    def closeEvent(self,event):
        if any(column.task is not None for column in self.columns):
            for column in self.columns:
                if column.task is not None:
                    column.status.setText('Calibration/restoration in progress; wait before closing.')
            event.ignore()
            return
        self.timer.stop()
        for column in self.columns:
            column.close_camera()
            column.pool.shutdown(wait=False)
        event.accept()


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--beams',type=int,nargs='+',choices=(1,2,3,4),default=[1])
    parser.add_argument('--shm-template',default='/dev/shm/baldr{beam}.im.shm')
    parser.add_argument('--frames',type=int,default=100,help='Frames in clear/reference calibration window')
    parser.add_argument('--history',type=int,default=100,help='Rolling plot samples (default 100); separate from calibration frames')
    parser.add_argument('--poll-ms',type=int,default=50)
    parser.add_argument('--stale-seconds',type=float,default=5.)
    # CHANGE: Add default values
    parser.add_argument('--gain', type=float, default=1.0)
    parser.add_argument('--bias-adu', type=float, default=1000.0)
    parser.add_argument('--read-noise-e', type=float, default=100.0)
    
    # NEW: Add simulation flag
    parser.add_argument('--simulation', action='store_true', help='Use simulation MDS endpoint defaults')
    parser.add_argument('--background-e',type=float,default=0.)
    parser.add_argument('--demo',action='store_true')
    parser.add_argument('--mds',help='MDS ZMQ endpoint; enables automatic clear mask-offset/return')
    parser.add_argument('--mask-axis',choices=('BMX','BMY'),default='BMY')
    parser.add_argument('--mask-offset-um',type=float,default=200.)
    parser.add_argument('--mask-settle-s',type=float,default=1.)
    parser.add_argument('--capture-timeout',type=float,default=120.)
    parser.add_argument('--motion-timeout',type=float,default=15.)
    parser.add_argument('--position-tolerance',type=float,default=.1)



    args = parser.parse_args(argv)
    
    # NEW: Dynamically set the MDS default if the user didn't explicitly provide one
    if args.mds is None:
        args.mds = 'tcp://127.0.0.1:5555' if args.simulation else 'tcp://mimir:5555'
        
    if args.demo:
        # Override with specific demo values if --demo is used and standard defaults are still present
        if args.read_noise_e == 100.0:
            args.read_noise_e = 8.0

    # if args.demo:
    #     for key,value in dict(gain=1.,bias_adu=1000.,read_noise_e=8.).items():
    #         if getattr(args,key) is None:
    #             setattr(args,key,value)
    # if any(getattr(args,k) is None for k in ('gain','bias_adu','read_noise_e')):
    #     parser.error('Live mode requires --gain, --bias-adu and --read-noise-e (editable per beam).')
    if len(set(args.beams))!=len(args.beams):
        parser.error('Choose each beam once')
    values=[args.gain,args.bias_adu,args.read_noise_e,args.background_e,args.stale_seconds]
    if not np.all(np.isfinite(values)) or args.gain<=0 or args.read_noise_e<0 or args.background_e<0 or args.frames<3 or args.history<3 or args.poll_ms<1 or args.stale_seconds<=0:
        parser.error('Invalid detector or timing settings')
    try:
        paths=[args.shm_template.format(beam=b) for b in args.beams]
    except (KeyError,ValueError) as exc:
        parser.error(str(exc))
    if len(set(paths))!=len(paths):
        parser.error('Shared-memory template must give each beam a distinct path; use {beam}.')
    motion=[args.mask_offset_um,args.mask_settle_s,args.capture_timeout,args.motion_timeout,args.position_tolerance]
    if not np.all(np.isfinite(motion)) or args.mask_offset_um==0 or args.mask_settle_s<0 or min(motion[2:])<=0:
        parser.error('Invalid motion/settling/timeout settings')
    return args


def main():
    args=parse_args()
    pg.setConfigOptions(imageAxisOrder='row-major')
    app=pg.mkQApp('Baldr halo Strehl')
    window=Window(args)
    window.show()
    app.exec()


if __name__=='__main__':
    main()
